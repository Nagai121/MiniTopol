"""Verified Parquet publication followed by idempotent catalog registration."""

import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

from ..registry import Registry
from ..storage import atomic_json, atomic_parquet, canonical, digest, fingerprint, read_json


def _payload_hash(table):
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, table.schema) as stream:
        stream.write_table(table.combine_chunks())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


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
            for key, value in (("trajectory", trajectory), ("representation", definition.representation),
                               ("descriptor", definition.name), ("definition_id", definition.identity)):
                table = table.append_column(key, pa.array([value] * len(table), type=pa.string()))
            table = table.replace_schema_metadata({b"minitopol": canonical(info).encode()})
            atomic_parquet(path, table)
            completed = {"schema": 2, "info": info, "rows": len(table), "payload_hash": payload_hash,
                         "file_sha256": fingerprint(path)["sha256"]}
            atomic_json(sidecar, completed)
            self._register(result_id, relative, completed)
        return result_id

    @staticmethod
    def _verify_completed(path, completed):
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
            con.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                        [result_id, info["trajectory"], definition["representation"], definition["name"],
                         canonical(definition), info["start"], info["stop"], relative, completed["rows"]])

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
