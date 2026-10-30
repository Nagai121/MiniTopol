"""Validate extracted ATLAS replica inputs and their provenance before conversion.

XTC contains only ordered coordinates, not atom identities. Thus consistency
with an ATLAS-provided topology is established by the provider pairing contract,
file identity and atom count, not by inventing atom identities from positions.
"""
from __future__ import annotations

from pathlib import Path
import re
import zipfile

import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCReader

from .client import _dataset_id, _kind, _sha256, _crc32
from .ledger import DownloadLedger
from .paths import source_directory

_REPLICA = re.compile(r"(?:^|[_.-])R([1-9][0-9]*)(?=$|[_.-])", re.IGNORECASE)
_SUFFIXES = {".gro", ".pdb", ".tpr", ".xtc"}


def _replica(path: Path) -> str | None:
    matches = _REPLICA.findall(path.stem)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous replica markers: {path.name}")
    return f"R{int(matches[0])}" if matches else None


def _choose(files: list[Path], replica: str, suffix: str) -> Path | None:
    matching = [p for p in files if p.suffix.lower() == suffix and _replica(p) == replica]
    if not matching:
        matching = [p for p in files if p.suffix.lower() == suffix and _replica(p) is None]
    if len(matching) > 1:
        raise ValueError(f"Ambiguous {suffix} for {replica}: {[p.name for p in matching]}")
    return matching[0] if matching else None


def _labels(path: Path) -> tuple[tuple[str, str, int], ...]:
    """Label residue occurrences independent of PDB/TPR residue numbering.

    Atom names and residue names alone cannot distinguish repeated residues.
    Ordinal by residue *name* makes repeated copies distinguishable without
    trusting PDB/TPR absolute residue ids, which need not agree.
    """
    universe = mda.Universe(str(path))
    try:
        ordinal = {}
        seen = {}
        result = []
        for atom in universe.atoms:
            residue_index = int(atom.resindex)
            residue = str(atom.resname).upper()
            if residue_index not in ordinal:
                seen[residue] = seen.get(residue, 0) + 1
                ordinal[residue_index] = seen[residue]
            result.append((residue, str(atom.name).upper(), ordinal[residue_index]))
        return tuple(result)
    finally:
        if getattr(universe, "trajectory", None) is not None:
            universe.trajectory.close()


def unique_subsequence(source: tuple, target: tuple) -> list[int]:
    """Require a unique, order-preserving source-to-TPR atom mapping."""
    if not source:
        raise ValueError("Empty atom topology")
    forward, position = [], 0
    for label in source:
        while position < len(target) and target[position] != label:
            position += 1
        if position == len(target):
            raise ValueError("Source topology is not an ordered subset of TPR")
        forward.append(position)
        position += 1
    backward, position = [], len(target) - 1
    for label in reversed(source):
        while position >= 0 and target[position] != label:
            position -= 1
        if position < 0:
            raise ValueError("Source topology is not an ordered subset of TPR")
        backward.append(position)
        position -= 1
    if forward != list(reversed(backward)):
        raise ValueError("Ambiguous atom-to-TPR correspondence")
    return forward


def _verify_acquisition(workspace, dataset: str, kind: str, files: list[Path]) -> str:
    """Recheck downloaded archive/ledger and all extracted inputs, if tracked.

    Manually staged inputs without a ledger remain supported; they are marked
    'manual' and cannot be mistaken for provider-verified downloads.
    """
    record = DownloadLedger(workspace).get(dataset, kind)
    directory = source_directory(workspace, dataset, kind)
    archive = directory / f"{dataset}_{kind}.zip"
    if record is None:
        if archive.exists():
            raise ValueError("Untracked ATLAS archive cannot authenticate extracted inputs")
        return "manual"
    if (record["status"] != "complete"
            or record["archive"] != archive.relative_to(workspace.path).as_posix()
            or not archive.is_file() or archive.is_symlink()
            or int(record["bytes"]) != archive.stat().st_size
            or record["sha256"] != _sha256(archive)):
        raise ValueError("ATLAS acquisition ledger/archive mismatch")
    with zipfile.ZipFile(archive) as zf:
        entries = {}
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename.replace("\\", "/")).suffix.lower() not in _SUFFIXES:
                continue
            name = Path(info.filename.replace("\\", "/")).name
            if name in entries:
                raise ValueError("Ambiguous input member names in ATLAS archive")
            entries[name] = info
        for path in files:
            info = entries.get(path.name)
            if (info is None or not path.is_file() or path.is_symlink()
                    or path.stat().st_size != info.file_size or _crc32(path) != info.CRC):
                raise ValueError(f"Extracted ATLAS input differs from verified archive: {path.name}")
    return "verified_archive"


def check_atlas(workspace, dataset: str, kind: str = "analysis") -> list[dict]:
    """Check source data while holding the migration/extraction exclusion lock."""
    with workspace.lock("atlas-sources"):
        return _check_atlas_unlocked(workspace, dataset, kind)


def _check_atlas_unlocked(workspace, dataset: str, kind: str = "analysis") -> list[dict]:
    """Return checker-validated independent plans or explicit per-replica errors."""
    dataset, kind = _dataset_id(dataset), _kind(kind)
    source = source_directory(workspace, dataset, kind)
    if not source.is_dir():
        raise FileNotFoundError(f"No extracted ATLAS inputs in {source}")
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in _SUFFIXES)
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("ATLAS inputs must be ordinary, non-symlink files")
    trajectories = [p for p in files if p.suffix.lower() == ".xtc"]
    if not trajectories:
        raise ValueError("ATLAS input has no extracted XTC")
    tagged = [_replica(p) for p in trajectories]
    if len(trajectories) != len(set(tagged)):
        raise ValueError("Multiple XTC files with the same or missing replica identifier")
    if len(trajectories) > 1 and None in tagged:
        raise ValueError("Multiple XTC files require explicit replica identifiers")
    provenance = _verify_acquisition(workspace, dataset, kind, files)
    plans = []
    for xtc in trajectories:
        replica = _replica(xtc) or "R1"
        try:
            with XTCReader(str(xtc)) as reader:
                atom_count = reader.n_atoms
            candidates = [_choose(files, replica, suffix) for suffix in (".gro", ".pdb")]
            valid = []
            for topology in (p for p in candidates if p is not None):
                try:
                    labels = _labels(topology)
                except Exception:
                    continue
                if len(labels) == atom_count:
                    valid.append((topology, labels))
            if not valid:
                raise ValueError(f"No GRO/PDB topology has {atom_count} atoms for {replica}")
            # A matching GRO and PDB must describe the same atom order.
            if len(valid) == 2 and valid[0][1] != valid[1][1]:
                raise ValueError("Conflicting GRO/PDB atom orders for the same replica")
            topology, labels = valid[0]
            tpr = _choose(files, replica, ".tpr")
            mapping, bond_error = None, None
            if tpr is not None:
                try:
                    mapping = unique_subsequence(labels, _labels(tpr))
                except Exception as error:
                    bond_error = f"{type(error).__name__}: {error}"
            plans.append({"dataset": dataset, "kind": kind, "replica": replica,
                          "topology": str(topology), "xtc": str(xtc),
                          "tpr": str(tpr) if tpr else None,
                          "source_to_tpr": mapping, "bond_error": bond_error,
                          "atoms": atom_count, "provenance": provenance})
        except Exception as error:
            plans.append({"dataset": dataset, "kind": kind, "replica": replica,
                          "error": f"{type(error).__name__}: {error}",
                          "provenance": provenance})
    replicas = {p["replica"] for p in plans}
    orphans = [p.name for p in files if p.suffix.lower() == ".tpr"
               and _replica(p) is not None and _replica(p) not in replicas]
    if orphans:
        raise ValueError(f"TPR without a corresponding replica XTC: {orphans}")
    return sorted(plans, key=lambda p: int(p["replica"][1:]))
