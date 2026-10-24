"""ATLAS input inspection: one validated file and atom correspondence plan per replica.

XTC has no atom identities. Atom count and an ATLAS-named topology establish
a supported pairing, not an independent proof of atom order inside an XTC.
Never silently choose between ambiguous file or topology matches.
"""
from __future__ import annotations

from pathlib import Path
import re

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader

from .client import _dataset_id, _kind

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


def _labels(path: Path) -> tuple[tuple[str, str], ...]:
    universe = mda.Universe(str(path))
    try:
        return tuple((str(res).upper(), str(atom).upper())
                     for res, atom in zip(universe.atoms.resnames, universe.atoms.names))
    finally:
        if getattr(universe, "trajectory", None) is not None:
            universe.trajectory.close()


def unique_subsequence(source: tuple, target: tuple) -> list[int]:
    """Map each source atom to an atom in target; require one ordered solution.

    Forward/backward greedy alignments agree exactly iff the monotone
    subsequence correspondence is unique. This is linear even with solvent.
    """
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


def check_atlas(workspace, dataset: str, kind: str = "analysis") -> list[dict]:
    """Inspect extracted ATLAS inputs; return per-replica plans shared by both consumers.

    An XTC is coupled to a topology only by unambiguous ATLAS replica naming
    and equal atom counts. TPR matching additionally checks ordered atom labels.
    A failed optional TPR check never invalidates an otherwise usable A input.
    """
    dataset, kind = _dataset_id(dataset), _kind(kind)
    source = workspace.trajectory_path(f"atlas-{dataset}-{kind}") / "source"
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
    plans = []
    for xtc in trajectories:
        replica = _replica(xtc) or "R1"
        # Prefer GRO when both are supplied, without changing GRO converter behavior.
        topology = (_choose(files, replica, ".gro")
                    or _choose(files, replica, ".pdb"))
        if topology is None:
            raise ValueError(f"No GRO or PDB topology for {replica}")
        labels = _labels(topology)
        with XTCReader(str(xtc)) as reader:
            atom_count = reader.n_atoms
        if atom_count != len(labels):
            raise ValueError(f"{replica}: {xtc.name} has {atom_count} atoms; "
                             f"{topology.name} has {len(labels)}")
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
                      "atoms": atom_count})
    # One TPR per replica or an unlabelled shared TPR is valid.
    # Orphan replica-specific TPRs indicate a misassembled input archive.
    replicas = {p["replica"] for p in plans}
    orphans = [p.name for p in files if p.suffix.lower() == ".tpr"
               and _replica(p) is not None and _replica(p) not in replicas]
    if orphans:
        raise ValueError(f"TPR without a corresponding replica XTC: {orphans}")
    return sorted(plans, key=lambda p: int(p["replica"][1:]))
