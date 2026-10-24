"""Resumable multipart Parquet, published only when the whole job completes."""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..registry import Registry
from ..storage import atomic_json, atomic_parquet, canonical, digest, fingerprint, read_json
from .results import _payload_hash, with_result_metadata


def verify_part(path, part, info):
    if not path.is_file():
        raise IOError("Completed descriptor part file is missing")
    if fingerprint(path)["sha256"] != part["sha256"]:
        raise IOError("Descriptor part checksum mismatch")
    metadata = pq.read_metadata(path)
    if metadata.num_rows != part["rows"] or (metadata.metadata or {}).get(b"minitopol") != canonical(info).encode():
        raise IOError("Descriptor part metadata/row count mismatch")


def verify_dataset(path, completed):
    if digest(completed["info"]) != path.stem or not completed["parts"]:
        raise IOError("Descriptor dataset completion identity/parts mismatch")
    rows, keys = 0, set()
    for index, part in enumerate(completed["parts"]):
        expected = f"{path.stem}/part-{index:08d}.parquet"
        if part["file"] != expected or part["key"] in keys:
            raise IOError("Descriptor dataset part identity mismatch")
        keys.add(part["key"])
        verify_part(path.parent / expected, part, completed["info"])
        rows += part["rows"]
    if rows != completed["rows"]:
        raise IOError("Descriptor dataset total row count mismatch")


def validate_table(table, stop):
    if not isinstance(table, pa.Table) or "value" not in table.column_names:
        raise ValueError("A descriptor part must be an Arrow table with a value column")
    reserved = {"trajectory", "representation", "descriptor", "definition_id"}
    if reserved.intersection(table.column_names) or len(set(table.column_names)) != len(table.column_names):
        raise ValueError("Duplicate or reserved result column")
    for field in table.schema:
        if pa.types.is_floating(field.type) and pc.all(pc.fill_null(pc.is_finite(table[field.name]), True)).as_py() is False:
            raise ValueError("Non-finite descriptor output must be represented by null with a reason")
    if "frame" in table.column_names:
        col = table["frame"]
        if not pa.types.is_integer(col.type) or (col.null_count and "level" not in table.column_names):
            raise ValueError("Frame indices must be integers; mixed aggregate rows need an explicit level")
        data = col.drop_null().to_numpy()
        if len(data) and (data.min() < 0 or data.max() >= stop):
            raise ValueError("Result frame is outside the trajectory")


class DatasetWriter:
    def __init__(self, store, trajectory, definition, *, layout):
        self.store, self.workspace, self.definition = store, store.workspace, definition
        manifest = Registry(self.workspace).manifest(trajectory)
        if definition.representation != "A":
            raise ValueError("The built-in multipart calculators currently target A only")
        with self.workspace.catalog() as con:
            saved = con.execute("SELECT definition FROM descriptor_definitions WHERE id=?", [definition.identity]).fetchone()
        if saved != (canonical(definition.as_dict()),):
            raise ValueError("Register the scientific descriptor definition before running a Calculator")
        self.info = {"trajectory": trajectory, "source_identity": manifest["identity"],
                     "definition": definition.as_dict(), "start": 0, "stop": manifest["frames"]}
        self.result_id = digest(self.info)
        self.relative = f"descriptors/{definition.name}/{trajectory}/A/{self.result_id}.parquet"
        self.path = self.workspace.path / self.relative
        self.directory = self.path.with_suffix("")
        self.marker = self.path.with_suffix(".json")
        self.progress = self.directory / "progress"
        self.layout = layout
        self.lock = self.workspace.lock("result-" + digest([trajectory, definition.identity]))
        self.complete = False
        self.journal = False

    def __enter__(self):
        self.lock.acquire()
        try:
            if self.marker.exists():
                completed = read_json(self.marker)
                if completed["info"] != self.info:
                    raise IOError("Dataset definition differs from completion record")
                self.store._verify_completed(self.path, completed)
                self.store._register(self.result_id, self.relative, completed)
                self.complete = True
                self.state = completed
            elif self.progress.exists():
                from .progress_journal import load
                self.state, self.journal = load(self.progress)
                if self.state["info"] != self.info or self.state["layout"] != self.layout:
                    raise ValueError("Resume the unfinished dataset with its original processing layout")
                for index, part in enumerate(self.state["parts"]):
                    if part["file"] != f"{self.result_id}/part-{index:08d}.parquet":
                        raise IOError("Unfinished dataset part identity mismatch")
                    verify_part(self.path.parent / part["file"], part, self.info)
            else:
                self.directory.mkdir(parents=True, exist_ok=True)
                self.state = {"schema": 3, "info": self.info, "layout": self.layout, "parts": []}
                from .progress_journal import start
                start(self.progress, self.state)
                self.journal = True
            self.by_key = {part["key"]: part for part in self.state["parts"]}
            return self
        except BaseException:
            self.lock.release()
            raise

    def has(self, key):
        return str(key) in self.by_key

    def append(self, key, table):
        if self.complete:
            raise ValueError("Completed datasets cannot be overwritten")
        validate_table(table, self.info["stop"])
        key, checksum = str(key), _payload_hash(table)
        undefined = table["value"].null_count
        if self.has(key):
            if self.by_key[key]["payload_hash"] != checksum:
                raise ValueError("Resumed descriptor part produced different values")
            return
        part_file = f"{self.result_id}/part-{len(self.state['parts']):08d}.parquet"
        table = with_result_metadata(table, self.info)
        child = self.path.parent / part_file
        atomic_parquet(child, table)
        part = {"key": key, "file": part_file, "rows": len(table),
                "sha256": fingerprint(child)["sha256"], "payload_hash": checksum, "undefined_values": undefined}
        self.state["parts"].append(part)
        if self.journal:
            from .progress_journal import append
            append(self.progress, len(self.state['parts'])-1, part)
        else:
            atomic_json(self.progress, self.state)
        self.by_key[key] = part

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None and not self.complete:
                if not self.state["parts"]:
                    raise ValueError("A completed dataset needs an explicitly typed (possibly empty) result part")
                completed = {**self.state, "rows": sum(part["rows"] for part in self.state["parts"])}
                verify_dataset(self.path, completed)
                atomic_json(self.marker, completed)
                self.store._register(self.result_id, self.relative, completed)
                self.state = completed
                self.progress.unlink(missing_ok=True)
        finally:
            self.lock.release()
