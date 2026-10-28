"""Workspace lifecycle and serialized DuckDB catalog access."""

from contextlib import contextmanager
import os
from pathlib import Path

import duckdb
from filelock import FileLock

from .storage import digest, identifier


class Workspace:
    def __init__(self, path=None):
        requested = path if path is not None else os.environ.get("MINITOPOL_WORKSPACE")
        self.path = Path(requested if requested is not None else self.default_path()).resolve()
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

    @staticmethod
    def default_path():
        """Anchor source installs to the project, independently of shell cwd."""
        project = Path(__file__).resolve().parents[2]
        if (project / "pyproject.toml").is_file() and (project / "src" / "minitopol").is_dir():
            return project / "BaseTraj_Workspace"
        # Wheels have no source project directory. Keep their default writable
        # and stable, while preserving the explicit path/environment overrides.
        return Path.home() / "MiniTopol" / "BaseTraj_Workspace"

    @contextmanager
    def catalog(self):
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

    def trajectory_path(self, trajectory):
        return self.path / "trajectories" / identifier(trajectory)

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
            from .descriptors.virtual_rama_store import query
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
            paths = [str(self.path / row[0]) for row in con.execute(
                "SELECT COALESCE(p.path,r.path) FROM results r LEFT JOIN result_parts p ON p.result_id=r.id" +
                (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY r.id,p.part_index", args
            ).fetchall()]
            if not paths:
                raise LookupError("No completed descriptor results match the request")
            relation = con.read_parquet(paths, union_by_name=True, hive_partitioning=False)
            if start is not None:
                relation = relation.filter(f"frame >= {int(start)}")
            if stop is not None:
                relation = relation.filter(f"frame < {int(stop)}")
            if columns:
                relation = relation.project(",".join('"' + c.replace('"', '""') + '"' for c in columns))
            return relation.to_arrow_table()
