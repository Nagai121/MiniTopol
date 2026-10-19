"""Verified Parquet publication followed by idempotent catalog registration."""

import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

from ..registry import Registry
from ..storage import atomic_json, atomic_parquet, canonical, digest, read_json


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
        buffer = pa.BufferOutputStream()
        with pa.ipc.new_stream(buffer, table.schema) as stream:
            stream.write_table(table.combine_chunks())
        payload_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
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
                if pq.read_metadata(path).num_rows != completed["rows"]:
                    raise IOError("Published descriptor row count differs from completion record")
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
            completed = {"schema": 1, "info": info, "rows": len(table), "payload_hash": payload_hash}
            atomic_json(sidecar, completed)
            self._register(result_id, relative, completed)
        return result_id

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
                if digest(completed["info"]) != sidecar.stem:
                    raise IOError("Descriptor completion record identity mismatch")
                if pq.read_metadata(path).num_rows != completed["rows"]:
                    raise IOError("Descriptor completion record row count mismatch")
                self._register(sidecar.stem, path.relative_to(self.workspace.path).as_posix(), completed)
                count += 1
        return count
