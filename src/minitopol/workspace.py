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
                      columns=None, start=None, stop=None):
        """Read published Parquet only; filtering/projection stays in DuckDB.

        Returns an Arrow table. Use narrow filters for large datasets.
        Frame filtering applies to per-frame rows; interval-valued descriptors
        should encode their interval explicitly in their result schema.
        """
        clauses, args = [], []
        for field, value in (("descriptor", descriptor), ("trajectory", trajectory),
                             ("representation", representation)):
            if value is not None:
                clauses.append(f"{field} = ?")
                args.append(value)
        with self.catalog() as con:
            paths = [str(self.path / row[0]) for row in con.execute(
                "SELECT path FROM results" + (" WHERE " + " AND ".join(clauses) if clauses else ""), args
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
