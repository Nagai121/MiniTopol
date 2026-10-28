"""Static MD system information; no trajectory duplication or guessed chemistry."""
from __future__ import annotations

from pathlib import Path
import math
import re
import shutil
import subprocess
import tempfile

import MDAnalysis as mda
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .storage import atomic_json, atomic_parquet, read_json
from .registry import Registry
from .descriptors.bond_reconstruction import _covalent_edges_from_tpr


FIELDS = ("temperature_k", "pressure_bar", "ensemble", "integrator", "time_step_ps",
          "thermostat", "barostat", "force_field_name", "force_field_version",
          "water_model", "ion_composition", "ph")


def _mdp(path):
    """Read explicit key=value MD settings, never infer absent settings."""
    values = {}
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.split(";", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key and value:
                values[key.lower().replace("_", "-")] = value
    return values


def _tpr_mdp(tpr):
    """Optional GROMACS extraction: -om produces a readable MDP-like dump.

    Never capture the potentially enormous gmx dump stdout in RAM.
    """
    binary = shutil.which("gmx") or shutil.which("gmx_mpi")
    if binary is None:
        return {}
    with tempfile.TemporaryDirectory(prefix="minitopol-gmx-") as directory:
        path = Path(directory) / "input.mdp"
        try:
            run = subprocess.run([binary, "dump", "-s", str(Path(tpr).resolve()),
                                  "-om", str(path)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=90, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        return _mdp(path) if run.returncode == 0 and path.is_file() else {}


def _number(value):
    if not value:
        return None
    parts = value.split()
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        return None
    if not nums or not all(math.isfinite(x) for x in nums):
        return None
    return nums[0] if len(set(nums)) == 1 else nums


def _settings_to_conditions(settings):
    out = {key: None for key in FIELDS}
    out["temperature_k"] = _number(settings.get("ref-t"))
    out["pressure_bar"] = _number(settings.get("ref-p"))
    out["integrator"] = settings.get("integrator")
    out["time_step_ps"] = _number(settings.get("dt"))
    out["thermostat"] = settings.get("tcoupl")
    out["barostat"] = settings.get("pcoupl")
    # Only infer an ensemble when both coupling controls are explicitly known.
    tc, pc = out["thermostat"], out["barostat"]
    off = {"no", "none", "off"}
    if tc is not None and pc is not None:
        out["ensemble"] = ("NPT" if pc.lower() not in off else
                           "NVT" if tc.lower() not in off else "NVE")
    return out


def publish_conditions(workspace, trajectory, *, tpr=None, mdp=None, supplied=None):
    """Persist all nine agreed condition groups; missing values are JSON null.

    Existing non-null values cannot be contradicted. A later call may enrich
    fields that were previously unavailable.
    """
    Registry(workspace).manifest(trajectory)
    path = workspace.trajectory_path(trajectory) / "system_conditions.json"
    parsed = _mdp(mdp) if mdp is not None else (_tpr_mdp(tpr) if tpr else {})
    incoming = _settings_to_conditions(parsed)
    if supplied is not None:
        if not isinstance(supplied, dict) or set(supplied) - set(FIELDS):
            raise ValueError("Unknown or invalid system condition fields")
        incoming.update(supplied)
    for key in ("temperature_k", "pressure_bar", "time_step_ps", "ph"):
        value = incoming[key]
        if value is not None and (not isinstance(value, (int, float, list)) or
                                  isinstance(value, bool) or
                                  not all(math.isfinite(x) for x in
                                          (value if isinstance(value, list) else [value]))):
            raise ValueError("Invalid numeric system condition: " + key)
    with workspace.lock("system-conditions-" + trajectory):
        if path.exists():
            old = read_json(path)
            if old.get("schema") != 1 or set(old) != set(FIELDS) | {"schema"}:
                raise ValueError("Unsupported system conditions schema")
            for key in FIELDS:
                if incoming[key] is not None and old[key] not in (None, incoming[key]):
                    raise ValueError("Conflicting system condition: " + key)
                if incoming[key] is None:
                    incoming[key] = old[key]
        atomic_json(path, {"schema": 1, **incoming})
    return incoming


def publish_molecules(workspace, trajectory, tpr, source_to_tpr):
    """Publish TPR-wide atoms and molecules with validated source/A indices.

    TPR may describe more atoms than a protein-only XTC; absent coordinate
    indices remain null. No ligand/cofactor or protonation classifications.
    """
    manifest = Registry(workspace).manifest(trajectory)
    from .hydrogen_bonds import _validate_mapping
    tpr = Path(tpr)
    u = mda.Universe(str(tpr))
    try:
        n = len(u.atoms)
        charge = np.asarray(u.atoms.charges, dtype=np.float64)
        types = list(map(str, u.atoms.types))
        names = list(map(str, u.atoms.names))
        try:
            moltypes = list(map(str, u.atoms.moltypes))
        except (AttributeError, ValueError):
            moltypes = [None] * n
        if len(charge) != n or not np.isfinite(charge).all() or len(types) != n:
            raise ValueError("TPR charges or atom types incomplete")
    finally:
        if getattr(u, "trajectory", None) is not None:
            u.trajectory.close()
    count, _, molecule_ids = _covalent_edges_from_tpr(tpr, molecule_ids=True)
    if count != n or len(molecule_ids) != n:
        raise ValueError("TPR atom count/molecule boundaries disagree")
    # Mapping must cover each *source* atom, but TPR may include excluded atoms.
    # Use the converter's saved A source indices to verify their mapping.
    mapping = _validate_mapping(source_to_tpr, len(source_to_tpr), n)
    selected = Registry(workspace).atoms(trajectory).to_pydict()
    point_by_source = {int(src): int(point) for src, point in
                       zip(selected["source_atom"], selected["point"])}
    if len(point_by_source) != manifest["atoms"] or any(s >= len(mapping) for s in point_by_source):
        raise ValueError("Source/A atom correspondence is invalid")
    source_by_tpr = {int(tp): i for i, tp in enumerate(mapping)}
    atoms = pa.Table.from_pylist([
        {"tpr_atom": i, "source_atom": source_by_tpr.get(i),
         "a_point": point_by_source.get(source_by_tpr.get(i)),
         "molecule_id": int(molecule_ids[i]), "name": names[i],
         "atom_type": types[i], "partial_charge": float(charge[i])}
        for i in range(n)
    ])
    grouped = {}
    for i, mol in enumerate(molecule_ids):
        info = grouped.setdefault(int(mol), {"molecule_id": int(mol),
                            "molecule_type": moltypes[i], "atom_count": 0,
                            "net_partial_charge": 0.0})
        if info["molecule_type"] != moltypes[i]:
            raise ValueError("TPR molecule contains conflicting molecule types")
        info["atom_count"] += 1
        info["net_partial_charge"] += float(charge[i])
    molecules = pa.Table.from_pylist(list(grouped.values()))
    # Explicitly count one-atom charged molecule types as ionic composition.
    # No inference of ligands, cofactors or polyatomic ions is attempted.
    ions = {}
    for info in grouped.values():
        if (info["atom_count"] == 1 and info["molecule_type"]
                and abs(info["net_partial_charge"]) >= 0.5):
            name = info["molecule_type"]
            ions[name] = ions.get(name, 0) + 1
    directory = workspace.trajectory_path(trajectory) / "system"
    with workspace.lock("system-molecules-" + trajectory):
        directory.mkdir(parents=True, exist_ok=True)
        for path, table in ((directory / "atom_properties.parquet", atoms),
                            (directory / "molecules.parquet", molecules)):
            if path.exists():
                if not pq.read_table(path).equals(table):
                    raise ValueError("Different molecule data already stored: " + str(path))
            else:
                atomic_parquet(path, table)
    if ions:
        publish_conditions(workspace, trajectory, supplied={"ion_composition": ions})
    return {"molecules": len(grouped), "atoms": n}
