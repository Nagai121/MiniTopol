"""Independent, verified persistence for one static TPR-derived bond graph per A.

The bond database is a dedicated DuckDB catalog table and an independent
Parquet/JSON pair. A.zarr, A's manifest and descriptor tables are never edited.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .registry import Registry
from .storage import atomic_json, atomic_parquet, canonical, digest, fingerprint, read_json


class BondStore:
    def __init__(self, workspace):
        self.workspace = workspace

    def _paths(self, trajectory):
        directory = self.workspace.trajectory_path(trajectory)
        return directory / "bonds.parquet", directory / "bonds.json"

    def _verify(self, trajectory, record):
        manifest = Registry(self.workspace).manifest(trajectory)
        if record.get("schema") != 1 or record["a_identity"] != manifest["identity"]:
            raise IOError("Bond record does not match the completed A trajectory")
        data, _ = self._paths(trajectory)
        if not data.is_file() or fingerprint(data)["sha256"] != record["sha256"]:
            raise IOError("Bond data missing or checksum mismatch")
        table = pq.read_table(data)
        if (table.schema.names != ["point_i", "point_j"]
            or table.schema.types != [pa.int64(), pa.int64()]
            or table.num_rows != record["edges"]):
            raise IOError("Bond graph schema or edge count mismatch")
        endpoints = table.to_pydict()
        edges = list(zip(endpoints["point_i"], endpoints["point_j"]))
        if (any(not 0 <= x < y < manifest["atoms"] for x, y in edges)
            or edges != sorted(set(edges))):
            raise IOError("Bond graph indices are invalid or unordered")
        return table

    def _register(self, trajectory, record):
        with self.workspace.catalog() as con:
            existing = con.execute("SELECT info FROM bond_graphs WHERE trajectory = ?",
                                   [trajectory]).fetchone()
            serialized = canonical(record)
            if existing and existing[0] != serialized:
                raise ValueError("Different bond graph already registered for this A")
            con.execute("INSERT INTO bond_graphs VALUES (?, ?) ON CONFLICT DO NOTHING",
                        [trajectory, serialized])

    def publish(self, trajectory, edges, *, tpr_sha256: str, mapping_sha256: str):
        manifest = Registry(self.workspace).manifest(trajectory)
        edge_list = sorted(set((int(a), int(b)) for a, b in edges))
        if any(a < 0 or a >= b or b >= manifest["atoms"] for a, b in edge_list):
            raise ValueError("Bond edges must be distinct ordered A point pairs")
        data, marker = self._paths(trajectory)
        record = {"schema": 1, "a_identity": manifest["identity"],
                  "tpr_sha256": tpr_sha256, "mapping_sha256": mapping_sha256,
                  "edges": len(edge_list)}
        table = pa.table({
            "point_i": pa.array([x for x, _ in edge_list], type=pa.int64()),
            "point_j": pa.array([y for _, y in edge_list], type=pa.int64()),
        })
        with self.workspace.lock("trajectory-bonds-" + trajectory):
            if marker.is_file():
                old = read_json(marker)
                if any(old.get(k) != v for k, v in record.items()):
                    raise ValueError("Different bond graph already exists for this A")
                self._verify(trajectory, old)
                if old["edges"] != len(edge_list):
                    raise ValueError("Conflicting existing bond count")
                old_data = pq.read_table(data)
                if not old_data.equals(table):
                    raise ValueError("Conflicting existing bond connectivity")
                self._register(trajectory, old)
                return old["edges"]
            if data.exists():
                # Crash between atomic Parquet publication and marker publication.
                # Reconcile the original file, never overwrite it blindly.
                old_data = pq.read_table(data)
                if not old_data.equals(table):
                    raise IOError("Uncommitted bond file has different contents")
            else:
                atomic_parquet(data, table)
            record["sha256"] = fingerprint(data)["sha256"]
            atomic_json(marker, record)
            self._verify(trajectory, record)
            self._register(trajectory, record)
        return len(edge_list)

    def get(self, trajectory):
        data, marker = self._paths(trajectory)
        if not marker.is_file():
            raise LookupError("No completed bond graph for this trajectory")
        with self.workspace.lock("trajectory-bonds-" + trajectory):
            record = read_json(marker)
            table = self._verify(trajectory, record)
            self._register(trajectory, record)
            return table

    def info(self, trajectory):
        _, marker = self._paths(trajectory)
        if not marker.is_file():
            return None
        with self.workspace.lock("trajectory-bonds-" + trajectory):
            record = read_json(marker)
            self._verify(trajectory, record)
            self._register(trajectory, record)
        return {"trajectory": trajectory, "edges": record["edges"],
                "a_identity": record["a_identity"], "tpr_sha256": record["tpr_sha256"]}
