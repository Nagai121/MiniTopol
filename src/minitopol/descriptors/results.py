"""Verified Parquet publication followed by idempotent catalog registration."""

import hashlib,io
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..registry import Registry
from ..storage import atomic_json, atomic_parquet, canonical, digest, fingerprint, read_json
from ..storage import _basic_result_schema, _BUILTIN_RESULTS

class _SHA256Sink(io.RawIOBase):
    def __init__(self):
        super().__init__();self.hasher=hashlib.sha256();self.position=0
    def writable(self):return True
    def tell(self):return self.position
    def write(self,data):
        self.hasher.update(data);self.position+=len(data);return len(data)


def _payload_hash(table):
    # Same combined-table IPC stream, including EOS and alignment bytes. Only
    # its destination changes; legacy payload hashes must remain identical.
    if isinstance(table,pa.Table) and table.nbytes>=256*1024:
        sink=_SHA256Sink()
        with pa.ipc.new_stream(sink,table.schema) as stream:
            stream.write_table(table.combine_chunks())
        return sink.hasher.hexdigest()
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, table.schema) as stream:
        stream.write_table(table.combine_chunks())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def with_result_metadata(table, info):
    """Attach the common result columns in their established order."""
    definition = info["definition"]
    for name, value in (("trajectory", info["trajectory"]), ("representation", definition["representation"]),
                        ("descriptor", definition["name"]), ("definition_id", digest(definition))):
        table = table.append_column(name, pa.repeat(pa.scalar(value, type=pa.string()), len(table)))
    return table.replace_schema_metadata({b"minitopol": canonical(info).encode()})

def _with_compact_result_metadata(table, info):
    # This private writer path never changes the existing public helper's
    # StringArray return type. Arbitrary/extension/field-metadata schemas retain
    # the original serialized Arrow schema and metadata construction.
    if info['definition']['name'] not in _BUILTIN_RESULTS or not _basic_result_schema(table.schema):
        return with_result_metadata(table,info)
    definition=info['definition']
    for name,value in (('trajectory',info['trajectory']),('representation',definition['representation']),
                       ('descriptor',definition['name']),('definition_id',digest(definition))):
        indices=pa.repeat(pa.scalar(0,type=pa.int8()),len(table))
        values=pa.array([value],type=pa.string())
        table=table.append_column(name,pa.DictionaryArray.from_arrays(indices,values))
    return table.replace_schema_metadata({b'minitopol':canonical(info).encode()})


class ResultStore:
    def __init__(self, workspace):
        self.workspace = workspace

    def publish(self, trajectory, definition, start, stop, table):
        manifest = Registry(self.workspace).manifest(trajectory)
        representation = Registry(self.workspace).validate_representation(trajectory, definition.representation)
        if definition.representation != "A":
            expected = representation["descriptor_definitions"].get(definition.name)
            if expected != definition.as_dict():
                raise ValueError("Calculation differs from the descriptor definition registered for this C")
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= manifest["frames"]:
            raise ValueError("Invalid descriptor frame interval")
        if not isinstance(table, pa.Table) or "value" not in table.column_names:
            raise ValueError("Results must be an Arrow table with a value column")
        reserved = {"trajectory", "representation", "descriptor", "definition_id"}
        if reserved.intersection(table.column_names):
            raise ValueError("Result table contains reserved metadata columns")
        if "frame" in table.column_names:
            frame = table["frame"].to_numpy()
            if not pa.types.is_integer(table.schema.field("frame").type) or table["frame"].null_count:
                raise ValueError("Frame values must be non-null integers")
            if len(frame) and (min(frame) < start or max(frame) >= stop):
                raise ValueError("Frame rows lie outside the published interval")
        # Match DatasetWriter: invalid floating results must not be published.
        for field in table.schema:
            if pa.types.is_floating(field.type):
                finite = pc.fill_null(pc.is_finite(table[field.name]), True)
                if pc.all(finite).as_py() is False:
                    raise ValueError("Non-finite descriptor output must be null with a reason")
        payload_hash = _payload_hash(table)
        info = {"trajectory": trajectory, "source_identity": manifest["identity"],
                "definition": definition.as_dict(), "start": start, "stop": stop}
        result_id = digest(info)
        relative = ("descriptors/" + definition.name + "/" + trajectory + "/"
                    + definition.representation + "/" + result_id + ".parquet")
        path = self.workspace.path / relative
        # Stage data and completion sidecar before catalog insertion. The
        # sidecar permits recovery if the process stops before DB registration.
        with self.workspace.lock("result-" + digest([trajectory, definition.identity])):
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                completed = read_json(sidecar)
                if completed["info"] != info or not path.exists():
                    raise IOError("Published descriptor files are missing or inconsistent")
                if completed["payload_hash"] != payload_hash:
                    raise ValueError("Retry produced different values for the same calculation identity")
                self._verify_completed(path, completed)
                # Never overwrite an already completed scientific result.
                self._register(result_id, relative, completed)
                return result_id
            with self.workspace.catalog() as con:
                overlap = con.execute("""SELECT id FROM results WHERE trajectory = ?
                    AND representation = ? AND definition = ?
                    AND start_frame < ? AND stop_frame > ?""",
                    [trajectory, definition.representation, canonical(definition.as_dict()), stop, start]).fetchone()
            if overlap:
                raise ValueError("Overlapping results for the same calculation definition")
            table = with_result_metadata(table, info)
            atomic_parquet(path, table)
            completed = {"schema": 2, "info": info, "rows": len(table), "payload_hash": payload_hash,
                         "file_sha256": fingerprint(path)["sha256"]}
            atomic_json(sidecar, completed)
            self._register(result_id, relative, completed)
        return result_id

    @staticmethod
    def _verify_completed(path, completed):
        if completed.get("schema") in (3, 4, 5, 6):
            if completed.get("schema") == 5:
                from .virtual_rama_store import verify_dataset
            elif completed.get("schema") in (4,6):
                from .count_store import verify_dataset
            else:
                from .datasets import verify_dataset
            verify_dataset(path, completed)
            return
        if completed.get("schema") not in (1, 2):
            raise IOError("Unsupported descriptor completion schema")
        if digest(completed["info"]) != path.stem:
            raise IOError("Descriptor completion record identity mismatch")
        if not path.is_file():
            raise IOError("Completed descriptor file is missing")
        metadata = pq.read_metadata(path)
        if metadata.num_rows != completed["rows"]:
            raise IOError("Descriptor completion record row count mismatch")
        saved = (metadata.metadata or {}).get(b"minitopol")
        if saved != canonical(completed["info"]).encode():
            raise IOError("Descriptor metadata differs from completion record")
        if completed["schema"] == 2:
            if fingerprint(path)["sha256"] != completed["file_sha256"]:
                raise IOError("Descriptor file checksum mismatch")
        else:
            # Schema 1 stored only an IPC hash of the caller's original table.
            # Verify that when possible. Never trust row counts as a substitute
            # for value validation; ambiguous legacy data must be reported.
            table = pq.ParquetFile(path).read()
            fields = [c for c in table.column_names if c not in
                      {"trajectory", "representation", "descriptor", "definition_id"}]
            original = table.select(fields).replace_schema_metadata(None)
            if _payload_hash(original) != completed["payload_hash"]:
                raise IOError("Legacy descriptor payload checksum mismatch; cannot verify stored values")

    def is_complete(self, result_id):
        """Only skip a catalogued partition after checking its persisted data."""
        with self.workspace.catalog() as con:
            row = con.execute("SELECT path FROM results WHERE id = ?", [result_id]).fetchone()
        if row is None:
            return False
        path = self.workspace.path / row[0]
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise IOError("Completed descriptor completion record is missing")
        completed = read_json(sidecar)
        with self.workspace.lock("result-" + digest([
                completed["info"]["trajectory"], digest(completed["info"]["definition"])])):
            self._verify_completed(path, completed)
        return True

    def _register(self, result_id, relative, completed):
        info = completed["info"]
        definition = info["definition"]
        with self.workspace.catalog() as con:
            overlap = con.execute("""SELECT id FROM results WHERE trajectory = ?
                AND representation = ? AND definition = ? AND id != ?
                AND start_frame < ? AND stop_frame > ?""",
                [info["trajectory"], definition["representation"], canonical(definition), result_id,
                 info["stop"], info["start"]]).fetchone()
            if overlap:
                raise ValueError("Overlapping results for the same calculation definition")
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute("INSERT INTO descriptor_definitions VALUES (?, ?) ON CONFLICT DO NOTHING",
                            [digest(definition), canonical(definition)])
                con.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                            [result_id, info["trajectory"], definition["representation"], definition["name"],
                             canonical(definition), info["start"], info["stop"], relative, completed["rows"]])
                if completed.get("schema") in (3, 4, 5, 6):
                    from pathlib import Path
                    # Bound temporary memory and retain the single publication
                    # transaction, ordering, keys and conflict policy.
                    for start in range(0, len(completed["parts"]), 4096):
                        batch = completed["parts"][start:start + 4096]
                        table = pa.table({
                            "result_id": pa.array([result_id] * len(batch), type=pa.string()),
                            "part_index": pa.array(range(start, start + len(batch)), type=pa.int64()),
                            "path": pa.array([(Path(relative).parent / part["file"]).as_posix()
                                              for part in batch], type=pa.string()),
                            "rows": pa.array([part["rows"] for part in batch], type=pa.int64()),
                        })
                        con.register("_minitopol_result_parts", table)
                        try:
                            con.execute("INSERT INTO result_parts SELECT * FROM _minitopol_result_parts "
                                        "ON CONFLICT DO NOTHING")
                        finally:
                            con.unregister("_minitopol_result_parts")
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise

    def dataset(self, trajectory, definition, *, layout):
        if definition.name == 'ramachandran':
            from .virtual_rama_store import RecipeWriter
            return RecipeWriter(self, trajectory, definition, layout=layout)
        if definition.name == 'sasa':
            from .count_store import CountWriter
            return CountWriter(self, trajectory, definition, layout=layout)
        from .datasets import DatasetWriter
        return DatasetWriter(self, trajectory, definition, layout=layout)

    def recover(self):
        """Re-register completed files; incomplete files are never exposed."""
        count = 0
        for sidecar in (self.workspace.path / "descriptors").rglob("*.json"):
            completed = read_json(sidecar)
            with self.workspace.lock("result-" + digest([
                    completed["info"]["trajectory"], digest(completed["info"]["definition"])])):
                path = sidecar.with_suffix(".parquet")
                self._verify_completed(path, completed)
                self._register(sidecar.stem, path.relative_to(self.workspace.path).as_posix(), completed)
                count += 1
        return count
