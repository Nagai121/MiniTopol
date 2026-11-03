"""Source topology/trajectory and chemical element evidence.

A verified match means the ordered coordinate arrays agree at a sampled
snapshot; it does NOT assert that every frame has the same atom identities.
Unknown provenance is recorded as unverified, never silently verified.
"""
from __future__ import annotations

from MDAnalysis.exceptions import NoDataError
import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree


MATCH_TOLERANCE_ANGSTROM = 0.06
_PROTEIN = frozenset((
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "HID HIE HIP HSD HSE HSP ASH GLH CYX CYM LYN MSE"
).split())


def _native_elements(atoms):
    try:
        raw = list(atoms.elements)
    except (AttributeError, NoDataError):
        return None
    if len(raw) != len(atoms):
        raise ValueError("Native element array has the wrong atom count")
    return [str(value).strip().capitalize() for value in raw]


def resolve_elements(universe, supplied=None):
    """Prefer native elements, then recognized protein atom roles, then guesses.

    When a supplied element contradicts an explicit native element, stop.
    Unverified guesses and declarations are retained with per-atom provenance.
    """
    atoms = universe.atoms
    count = len(atoms)
    native = _native_elements(atoms)
    if supplied is not None:
        if len(supplied) != count:
            raise ValueError("Explicit element array must cover every source atom")
        values = [str(v).strip().capitalize() for v in supplied]
        if any(not v for v in values):
            raise ValueError("Explicit element identities must be nonempty")
        if native is not None:
            for i, (given, known) in enumerate(zip(values, native)):
                if known and given != known:
                    raise ValueError(f"Explicit element differs from topology at atom {i}")
        states = ["verified" if native and native[i] else "unverified" for i in range(count)]
        provenance = ["native+supplied" if states[i] == "verified" else "supplied" for i in range(count)]
    else:
        native = native or [""] * count
        values = list(native)
        states = ["verified" if e else "unverified" for e in native]
        provenance = ["native" if e else "" for e in native]
        for i in range(count):
            if values[i]:
                continue
            residue = str(atoms.resnames[i]).strip().upper()
            name = str(atoms.names[i]).strip().upper()
            if residue in _PROTEIN and name[:1] in {"C", "N", "O", "S", "H", "P"}:
                values[i] = name[0].capitalize()
                provenance[i] = "protein-atom-name"
                # A known protein naming convention is evidence, not a
                # chemically confirmed element assignment.
                continue
        if any(not e for e in values):
            # MDAnalysis may legitimately need to guess missing elements.
            # Never re-guess an already explicit native or user-supplied value.
            universe.guess_TopologyAttrs(to_guess=["elements"])
            guessed = _native_elements(atoms)
            if guessed is None:
                raise ValueError("Unable to infer source atom elements")
            for i in range(count):
                if not values[i]:
                    values[i] = guessed[i]
                    provenance[i] = "name-guess"
    unresolved = [i for i, e in enumerate(values) if not e or e == "X"]
    if unresolved:
        examples = [(int(i), str(atoms.names[i]), str(atoms.resnames[i]),
                     str(values[i])) for i in unresolved[:12]]
        raise ValueError(f"Unknown source atom element at {len(unresolved)} atoms: {examples}; supply explicit elements")
    report = {
        "status": "verified" if all(x == "verified" for x in states) else "unverified",
        "native_count": sum(x.startswith("native") for x in provenance),
        "protein_name_count": provenance.count("protein-atom-name"),
        "guess_count": provenance.count("name-guess"),
        "supplied_count": provenance.count("supplied"),
        "reason": ("all elements present in source topology" if all(x == "verified" for x in states)
                   else "one or more elements supplied/inferred without independent confirmation")
    }
    return np.asarray(values, dtype=str), provenance, states, report


def validate_coordinate_pairing(universe, topology):
    """Conservatively probe topology coordinates against first/last XTC frames.

    Direct ordered-coordinate agreement proves an aligned snapshot.
    An overwhelming unordered match but failed ordered match is strong
    evidence of atom permutation, so conversion must stop.
    Distinct snapshots cannot establish order and are *unverified*, not bad.
    """
    from pathlib import Path
    try:
        reference = mda.Universe(str(topology))
        try:
            xyz = np.asarray(reference.atoms.positions, dtype=np.float64).copy()
        finally:
            if getattr(reference, "trajectory", None) is not None:
                reference.trajectory.close()
    except (AttributeError, NoDataError, ValueError, OSError):
        return {"status": "unverified", "method": "topology coordinates unavailable",
                "reason": "XTC carries positions without atom identity"}
    if xyz.shape != (len(universe.atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError("Topology reference coordinates and trajectory atom axes disagree")
    frames = len(universe.trajectory)
    for frame in sorted({0, frames - 1}):
        snapshot = np.asarray(universe.trajectory[frame].positions, dtype=np.float64)
        if np.max(np.linalg.norm(snapshot - xyz, axis=1)) <= MATCH_TOLERANCE_ANGSTROM:
            return {"status": "verified", "method": "ordered-coordinate-snapshot",
                    "frame": int(frame), "tolerance_angstrom": MATCH_TOLERANCE_ANGSTROM}
        # Only a near-identical *unordered* coordinate set supports an
        # order-mismatch conclusion. Moving trajectories fail this test and
        # remain unverified. Avoid quadratic all-pairs distance calculations.
        if len(xyz) > 1:
            distances, indices = cKDTree(snapshot).query(xyz, k=1)
            matched = distances <= MATCH_TOLERANCE_ANGSTROM
            if (np.mean(matched) >= 0.999
                    and len(np.unique(indices[matched])) == int(np.sum(matched))
                    and np.mean(indices[matched] == np.flatnonzero(matched)) < 0.95):
                raise ValueError("Topology/XTC atom order mismatch: coordinate set agrees but indices differ")
    return {"status": "unverified", "method": "coordinate snapshots differ",
            "reason": "topology may represent a different frame; ordering cannot be established"}
