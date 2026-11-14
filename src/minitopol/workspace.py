"""Workspace lifecycle and serialized DuckDB catalog access."""

from contextlib import contextmanager
import json
import hashlib
import os
import sqlite3
import threading
from pathlib import Path, PurePosixPath

import duckdb
from filelock import FileLock

from .storage import digest, identifier
from .windows_paths import filesystem_path


_query_session_local = threading.local()


class Workspace:
    def __init__(self, path=None):
        requested = path if path is not None else os.environ.get("MINITOPOL_WORKSPACE")
        record = Path.home() / ".minitopol" / "storage-location.json"
        registered = requested is None and record.is_file()
        self.path = filesystem_path(
            Path(requested if requested is not None else self.default_path()).expanduser(),
            threshold=160).resolve()
        self._write_nesting = threading.local()
        self._tier_access_lock = threading.Lock()
        self._tier_access = None
        if registered and (not self.path.is_dir() or not (self.path / "integrated.duckdb").is_file()):
            raise FileNotFoundError(f"Registered MiniTopol Storage or its catalog is missing: {self.path}")
        if requested is None and not registered and not self.path.exists():
            project = Path(__file__).resolve().parents[2]
            legacy = project / "BaseTraj_Workspace"
            if (project / "pyproject.toml").is_file() and (legacy / "integrated.duckdb").is_file():
                raise RuntimeError(
                    f"Existing legacy Workspace detected at {legacy}; "
                    "run 'minitopol migrate-workspace' before creating the new default storage")
        self.path.mkdir(parents=True, exist_ok=True)
        for directory in ("trajectories", "descriptors", "locks"):
            (self.path / directory).mkdir(exist_ok=True)
        with self.catalog() as con:
            con.execute("CREATE TABLE IF NOT EXISTS metadata(version INTEGER PRIMARY KEY)")
            versions = con.execute("SELECT version FROM metadata").fetchall()
            if versions and versions != [(1,)]:
                raise ValueError("Unsupported workspace schema version")
            con.execute("INSERT INTO metadata VALUES (1) ON CONFLICT DO NOTHING")
            con.execute("""CREATE TABLE IF NOT EXISTS trajectories(
                id VARCHAR PRIMARY KEY, path VARCHAR NOT NULL, frames BIGINT NOT NULL,
                atoms BIGINT NOT NULL, identity VARCHAR NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS representations(
                id VARCHAR PRIMARY KEY, definition VARCHAR NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS results(
                id VARCHAR PRIMARY KEY, trajectory VARCHAR NOT NULL,
                representation VARCHAR NOT NULL, descriptor VARCHAR NOT NULL,
                definition VARCHAR NOT NULL, start_frame BIGINT NOT NULL,
                stop_frame BIGINT NOT NULL, path VARCHAR UNIQUE NOT NULL,
                rows BIGINT NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS descriptor_definitions(
                id VARCHAR PRIMARY KEY, definition VARCHAR NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS result_parts(
                result_id VARCHAR, part_index BIGINT, path VARCHAR UNIQUE NOT NULL,
                rows BIGINT NOT NULL, PRIMARY KEY(result_id, part_index))""")
            # Independent bond graph catalog; no change to A/C or result schemas.
            con.execute("""CREATE TABLE IF NOT EXISTS bond_graphs(
                trajectory VARCHAR PRIMARY KEY, info VARCHAR NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS hydrogen_bond_sets(
                trajectory VARCHAR PRIMARY KEY, info VARCHAR NOT NULL)""")
        if requested is None and not registered:
            record.parent.mkdir(parents=True, exist_ok=True)
            temporary = record.with_suffix(".tmp")
            temporary.write_text(json.dumps({"path": str(self.path)}), encoding="utf-8")
            os.replace(temporary, record)

    @staticmethod
    def default_path():
        """Anchor source installs to the project, independently of shell cwd."""
        record = Path.home() / ".minitopol" / "storage-location.json"
        if record.is_file():
            location = json.loads(record.read_text(encoding="utf-8"))
            if not isinstance(location, dict) or not isinstance(location.get("path"), str):
                raise ValueError("Invalid registered MiniTopol Storage path")
            return Path(location["path"])
        project = Path(__file__).resolve().parents[2]
        if (project / "pyproject.toml").is_file() and (project / "src" / "minitopol").is_dir():
            return project.parent / "MiniTopol Storage"
        # The installed wheel also keeps mutable data outside package files.
        return Path.home() / "MiniTopol Storage"

    @contextmanager
    def query_session(self):
        """Reuse one serialized catalog connection for explicit repeated reads.

        Holds the workspace catalog lock for the whole with-block. Other
        processes wait until it exits; do not use for unbounded idle sessions.
        Every query still verifies the same result files, with no science cache.
        """
        sessions = getattr(_query_session_local, 'connections', None)
        if sessions is None:
            sessions = {}; _query_session_local.connections = sessions
        key = str(self.path)
        if key in sessions:
            yield self
            return
        with self.catalog() as con:
            sessions[key] = con
            try:
                yield self
            finally:
                del sessions[key]

    @contextmanager
    def catalog(self):
        existing = getattr(_query_session_local, 'connections', {}).get(str(self.path))
        if existing is not None:
            yield existing
            return
        # Readers take the same lock: DuckDB connections in other processes
        # must not overlap a write connection.
        with FileLock(str(self.path / "locks" / "catalog.lock")):
            con = duckdb.connect(str(self.path / "integrated.duckdb"))
            try:
                yield con
            finally:
                con.close()

    def lock(self, key):
        return FileLock(str(self.path / "locks" / f"{digest(key)}.lock"))

    def storage_writer_lock(self, relative, key):
        """A writer-scoped cross-process lock shared with tier offload/pruning.

        No scientific writer may mutate a logical dataset after external
        promotion. A missing placement DB is an error if a registry marker
        exists, rather than permission to recreate an empty catalog.
        """
        from .storage_tiers import logical_path
        relative = logical_path(relative)
        identifier_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        tier = FileLock(str(self.path / "locks" / "storage-tiers" /
                            (identifier_hash + ".lock")))
        original = self.lock(key)
        return _ManagedWriteLock(self, relative, tier, original)

    @contextmanager
    def storage_write(self, relative):
        guard = self.storage_writer_lock(relative, "storage-write-" + relative)
        guard.acquire()
        try:
            yield
        finally:
            guard.release()

    @contextmanager
    def storage_read_scope(self, overrides):
        """Per-thread, read-only locations for oversized external primaries."""
        contexts = getattr(_query_session_local, "storage_overrides", None)
        if contexts is None:
            contexts = {}
            _query_session_local.storage_overrides = contexts
        key = str(self.path)
        previous = contexts.get(key)
        contexts[key] = dict(overrides)
        try:
            yield self
        finally:
            if previous is None:
                contexts.pop(key, None)
            else:
                contexts[key] = previous

    def storage_read_overrides(self):
        return dict(getattr(_query_session_local, "storage_overrides", {}).get(
            str(self.path), {}))

    def resolve_path(self, relative):
        """Resolve read-only scientific paths against published external primaries.

        A Provider get() supplies an explicit storage_read_scope and thus retains
        its choice of staged cache versus direct read. Unscoped Registry and
        Workspace queries consult the durable placement catalog themselves.
        This does NOT change the internal physical writer paths.
        """
        path = PurePosixPath(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Data path must be relative to Workspace")
        key = str(self.path)
        scopes = getattr(_query_session_local, "storage_overrides", {})
        if key in scopes:
            for object_name, external in sorted(
                    scopes[key].items(), key=lambda pair: len(pair[0]), reverse=True):
                root = PurePosixPath(object_name)
                if path == root or root in path.parents:
                    return Path(external) / path.relative_to(root)
            return self.path / str(path)

        parts = path.parts
        if len(parts) >= 2 and parts[0] == "trajectories":
            object_name = "/".join(parts[:2])
        elif len(parts) >= 3 and parts[0] in ("descriptors", "sources"):
            object_name = "/".join(parts[:3])
        else:
            return self.path / str(path)
        if not (self.path / ".minitopol-tier-registry").exists():
            return self.path / str(path)
        if not (self.path / "storage_index.sqlite3").is_file():
            raise FileNotFoundError("Storage placement catalog is missing")
        if self._tier_access is None:
            with self._tier_access_lock:
                if self._tier_access is None:
                    from .storage_tiers import TieredStorage
                    self._tier_access = TieredStorage(self.path)
        record = self._tier_access._object(object_name)
        if record is not None and record[0] == "external":
            try:
                external = self._tier_access._direct_external(object_name, record)
            except (FileNotFoundError, RuntimeError):
                # If an external drive is unplugged, keep serving a verified
                # local replica. No eviction can discard it without first
                # verifying the missing external primary again.
                from .storage_tiers import _tree_digest
                local_root = self.path / object_name
                if not local_root.exists():
                    raise
                digest, size = _tree_digest(local_root)
                if digest != record[1] or size != record[2]:
                    raise IOError(
                        f"Local fallback differs from verified primary: {object_name}")
                return local_root / path.relative_to(PurePosixPath(object_name))
            return external / path.relative_to(PurePosixPath(object_name))
        return self.path / str(path)

    def trajectory_path(self, trajectory):
        """Use the Provider's explicit overlay, else the physical writer path.

        Background Provider kernels still obtain paths with this established
        API. An explicit storage_read_scope must therefore take precedence.
        Unscoped scientific writes retain the original internal path.
        """
        name = identifier(trajectory)
        if str(self.path) in getattr(_query_session_local, "storage_overrides", {}):
            return self.resolve_path("trajectories/" + name)
        return self.path / "trajectories" / name

    def read_trajectory_path(self, trajectory):
        """Return the readable internal or verified external trajectory path."""
        return self.resolve_path("trajectories/" + identifier(trajectory))

    def readonly_overrides(self, trajectory):
        """Paths for lightweight scientific read-only Workspace adapters."""
        overrides = self.storage_read_overrides()
        name = identifier(trajectory)
        external = self.read_trajectory_path(name)
        if external != self.trajectory_path(name):
            overrides["trajectories/" + name] = external
        return overrides

    def list_trajectories(self):
        with self.catalog() as con:
            return con.execute("SELECT id, frames, atoms FROM trajectories ORDER BY id").fetchall()

    def query_results(self, *, descriptor=None, trajectory=None, representation=None,
                      definition_id=None, columns=None, start=None, stop=None):
        """Read published logical results, including native counts and recipes.

        Returns an Arrow table. Use narrow filters for large datasets.
        Frame filtering applies to per-frame rows; interval-valued descriptors
        should encode their interval explicitly in their result schema.
        """
        if definition_id is not None:
            import json
            with self.catalog() as con:
                row = con.execute("SELECT definition FROM descriptor_definitions WHERE id=?", [definition_id]).fetchone()
            if row is None:
                raise LookupError("Unknown descriptor definition")
            defined_descriptor = json.loads(row[0])["name"]
            if descriptor is None:
                descriptor = defined_descriptor
            elif descriptor != defined_descriptor:
                raise LookupError("Descriptor and definition do not match")
        if descriptor is None:
            with self.catalog() as con:
                names = con.execute("SELECT DISTINCT descriptor FROM results" +
                                    (" WHERE trajectory=?" if trajectory is not None else "") +
                                    " ORDER BY descriptor", [trajectory] if trajectory is not None else []).fetchall()
            if not names:
                raise LookupError("No completed descriptor results match the request")
            tables = [self.query_results(descriptor=name, trajectory=trajectory,
                                        representation=representation, columns=columns, start=start, stop=stop)
                      for (name,) in names]
            # Preserve DuckDB's UNION BY NAME coercion for heterogeneous
            # descriptors (for example numeric values and classification text).
            with duckdb.connect() as con:
                for index, table in enumerate(tables):
                    con.register(f"logical_{index}", table)
                return con.sql(" UNION ALL BY NAME ".join(
                    f"SELECT * FROM logical_{index}" for index in range(len(tables)))).to_arrow_table()
        if descriptor in ('ramachandran', 'sasa') and representation not in (None, 'A'):
            raise LookupError("No completed descriptor results match the representation")
        if descriptor == 'ramachandran':
            from .descriptors.virtual_rama_layout import query
            return query(self,trajectory=trajectory,definition_id=definition_id,columns=columns,start=start,stop=stop)
        if descriptor == 'sasa':
            from .descriptors.count_store import query
            return query(self, trajectory=trajectory, definition_id=definition_id,
                         columns=columns, start=start, stop=stop)
        clauses, args = [], []
        for field, value in (("descriptor", descriptor), ("trajectory", trajectory),
                             ("representation", representation)):
            if value is not None:
                clauses.append(f"r.{field} = ?")
                args.append(value)
        with self.catalog() as con:
            if definition_id is not None:
                row = con.execute("SELECT definition FROM descriptor_definitions WHERE id=?", [definition_id]).fetchone()
                if row is None:
                    raise LookupError("Unknown descriptor definition")
                clauses.append("r.definition = ?")
                args.append(row[0])
            paths = [str(self.resolve_path(row[0])) for row in con.execute(
                "SELECT COALESCE(p.path,r.path) FROM results r LEFT JOIN result_parts p ON p.result_id=r.id" +
                (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY r.id,p.part_index", args
            ).fetchall()]
            if not paths:
                raise LookupError("No completed descriptor results match the request")
            relation = con.read_parquet(paths, union_by_name=True, hive_partitioning=False)
            # Whole-trajectory descriptors, such as RMSF, have no frame column.
            # A frame range does not constrain those aggregate results.
            if "frame" in relation.columns:
                if start is not None:
                    relation = relation.filter(f"frame >= {int(start)}")
                if stop is not None:
                    relation = relation.filter(f"frame < {int(stop)}")
            if columns:
                relation = relation.project(",".join('"' + c.replace('"', '""') + '"' for c in columns))
            return relation.to_arrow_table()


class _ManagedWriteLock:
    """FileLock-compatible exclusive lock with an offloaded-primary check."""

    def __init__(self, workspace, relative, tier_lock, writer_lock):
        self.workspace, self.relative = workspace, relative
        self.tier_lock, self.writer_lock = tier_lock, writer_lock
        self.held = False
        self.owns_tier = False

    def acquire(self):
        (self.workspace.path / "locks" / "storage-tiers").mkdir(parents=True, exist_ok=True)
        depths = getattr(self.workspace._write_nesting, "depths", None)
        if depths is None:
            depths = {}
            self.workspace._write_nesting.depths = depths
        self.owns_tier = depths.get(self.relative, 0) == 0
        if self.owns_tier:
            self.tier_lock.acquire()
        try:
            directory = self.workspace.path
            marker = directory / ".minitopol-tier-registry"
            index = directory / "storage_index.sqlite3"
            if marker.exists() and not index.is_file():
                raise FileNotFoundError("Storage placement catalog is missing")
            if index.is_file():
                with sqlite3.connect(index, timeout=30) as db:
                    row = db.execute(
                        "SELECT primary_tier FROM tier_objects WHERE relative=?",
                        (self.relative,)).fetchone()
                if row is not None and row[0] == "external":
                    raise PermissionError(
                        f"Dataset is externally authoritative; mutation rejected: {self.relative}")
            self.writer_lock.acquire()
            depths[self.relative] = depths.get(self.relative, 0) + 1
            self.held = True
        except BaseException:
            if self.owns_tier:
                self.tier_lock.release()
            raise
        return self

    def release(self):
        if self.held:
            try:
                self.writer_lock.release()
            finally:
                self.held = False
                depths = self.workspace._write_nesting.depths
                depths[self.relative] -= 1
                if not depths[self.relative]:
                    del depths[self.relative]
                if self.owns_tier:
                    self.tier_lock.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()
