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
    from .descriptors.context import PROTEIN_NAMES
    details = atoms.to_pydict()
    points_by_residue = {}
    for point, residue in enumerate(details["residue_index"]):
        points_by_residue.setdefault(residue, []).append(point)

    if alpha_carbons is not None:
        if (not alpha_carbons or len(set(alpha_carbons)) != len(alpha_carbons)
                or any(type(i) is not int or not 0 <= i < len(atoms) for i in alpha_carbons)):
            raise ValueError("Explicit C-alpha indices must be nonempty, unique A point indices")
        selected = atoms.take(pa.array(alpha_carbons))
        if (any(e != "C" for e in selected["element"].to_pylist())
                or len(set(selected["residue_index"].to_pylist())) != len(alpha_carbons)):
            raise ValueError("Explicit C-alpha atoms must be carbon, with at most one per residue")
        uncertain_alpha = []
        for point in alpha_carbons:
            residue_name = details["residue_name"][point].upper()
            atom_name = details["name"][point].upper()
            if residue_name in PROTEIN_NAMES and atom_name != "CA":
                raise ValueError(f"Inconsistent C-alpha annotation at point {point}: expected CA atom name")
            if residue_name not in PROTEIN_NAMES:
                uncertain_alpha.append(int(point))
        alpha_check = {"status": "unverified" if uncertain_alpha else "verified",
                       "method": "element + residue name + CA atom name for recognized amino acids",
                       "unverified_points": uncertain_alpha}
    else:
        alpha_check = {"status": "unverified", "reason": "no explicit C-alpha annotation"}

    # A user-supplied segment is a proposed covalent path, not evidence of
    # peptide/other backbone connectivity by itself.
    neighbors = [(a, b) for chain in chains for segment in chain["segments"]
                 for a, b in zip(segment, segment[1:])]
    graph_info = registry.bond_info(trajectory)
    if graph_info is None and neighbors:
        segment_check = {"status": "unverified", "reason": "no TPR-derived bond graph"}
    elif not neighbors:
        segment_check = {"status": "verified", "method": "no adjacent residues specified",
                         "tested_links": 0}
    else:
        bond_table = registry.bonds(trajectory).to_pydict()
        bonds = {tuple(sorted((i, j))) for i, j in
                 zip(bond_table["point_i"], bond_table["point_j"])}
        tested = 0
        for left, right in neighbors:
            a, b = points_by_residue[left], points_by_residue[right]
            names_a = {details["name"][p]: p for p in a}
            names_b = {details["name"][p]: p for p in b}
            amino = (details["residue_name"][a[0]].upper() in PROTEIN_NAMES
                     and details["residue_name"][b[0]].upper() in PROTEIN_NAMES)
            if amino and "C" in names_a and "N" in names_b:
                candidates = [(names_a["C"], names_b["N"])]
            else:
                # Non-standard units: any covalent inter-residue link suffices.
                candidates = [(i, j) for i in a for j in b]
            if not any(tuple(sorted(pair)) in bonds for pair in candidates):
                raise ValueError(f"Inconsistent continuous segment: residues {left} and {right} lack a supporting covalent bond")
            tested += 1
        segment_check = {"status": "verified", "method": "TPR bond-graph covalent adjacency",
                         "tested_links": tested, "tpr_sha256": graph_info["tpr_sha256"]}

    spec = {"schema": 1, "chains": chains, "alpha_carbons": alpha_carbons,
            "validation": {"alpha_carbons": alpha_check,
                           "continuous_segments": segment_check},
            **overrides}
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
