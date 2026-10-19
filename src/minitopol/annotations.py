"""Explicit residue continuity and atomic overrides, without chemical bonds."""

import numpy as np
import re
import pyarrow as pa
import pyarrow.parquet as pq

from .storage import atomic_parquet, canonical, digest


def annotate(registry, trajectory, *, chains, alpha_carbons=None, masses=None, radii=None):
    atoms = registry.atoms(trajectory)
    residues = sorted(set(atoms["residue_index"].to_pylist()))
    rows = {r: {"residue_index": r, "chain_id": None, "run": None, "sequence_position": None} for r in residues}
    seen, names, run = set(), set(), 0
    for chain in chains:
        name = chain["id"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Chain IDs must be nonempty and unique")
        names.add(name)
        for segment in chain["segments"]:
            if not segment:
                raise ValueError("Continuous segments cannot be empty")
            for sequence, residue in enumerate(segment):
                if type(residue) is not int or residue not in rows or residue in seen:
                    raise ValueError("Each annotated residue must exist and occur only once")
                seen.add(residue)
                rows[residue].update(chain_id=name, run=run, sequence_position=sequence)
            run += 1
    overrides = {}
    for key, values in (("masses", masses), ("radii", radii)):
        normalized = {}
        for point, value in (values or {}).items():
            if not ((type(point) is int and point >= 0) or
                    (isinstance(point, str) and re.fullmatch(r"[0-9]+", point))):
                raise ValueError("Atomic override indices must be nonnegative integer indices, not fractional values")
            target = str(int(point))
            if target in normalized or type(value) is bool:
                raise ValueError("Atomic overrides must be unambiguous numeric values")
            normalized[target] = float(value)
        values = normalized
        if any(not 0 <= int(k) < len(atoms) or not np.isfinite(v) or v <= 0 for k, v in values.items()):
            raise ValueError("Atomic overrides require valid A point indices and positive finite values")
        overrides[key] = values
    if alpha_carbons is not None:
        if not alpha_carbons or len(set(alpha_carbons)) != len(alpha_carbons) or any(type(i) is not int or not 0 <= i < len(atoms) for i in alpha_carbons):
            raise ValueError("Explicit C-alpha indices must be nonempty, unique A point indices")
        selected = atoms.take(pa.array(alpha_carbons))
        if any(e != "C" for e in selected["element"].to_pylist()) or len(set(selected["residue_index"].to_pylist())) != len(alpha_carbons):
            raise ValueError("Explicit C-alpha atoms must be carbon, with at most one per residue")
    spec = {"schema": 1, "chains": chains, "alpha_carbons": alpha_carbons, **overrides}
    table = pa.Table.from_pylist(list(rows.values()), schema=pa.schema([
        ("residue_index", pa.int64()), ("chain_id", pa.string()),
        ("run", pa.int64()), ("sequence_position", pa.int64())]))
    table = table.replace_schema_metadata({b"minitopol_annotations": canonical(spec).encode()})
    path = registry.workspace.trajectory_path(trajectory) / "annotations.parquet"
    with registry.workspace.lock("trajectory-" + trajectory):
        if path.exists():
            if pq.read_metadata(path).metadata[b"minitopol_annotations"] != canonical(spec).encode():
                raise ValueError("Existing annotations are immutable; import under a new trajectory ID to change them")
        else:
            atomic_parquet(path, table)
    return digest(spec)


def annotations(registry, trajectory):
    path = registry.workspace.trajectory_path(trajectory) / "annotations.parquet"
    if not path.exists():
        return {"schema": 1, "chains": [], "alpha_carbons": None, "masses": {}, "radii": {}}, {}
    import json
    table = pq.read_table(path)
    spec = json.loads(table.schema.metadata[b"minitopol_annotations"])
    return spec, {row["residue_index"]: row for row in table.to_pylist()}
