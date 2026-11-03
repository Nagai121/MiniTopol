"""Independent real-trajectory compatibility probes for strict source validation.

These intentionally assert acceptance for physically paired real datasets.
A failure documents a concrete possible false rejection, not a synthetic case.
MDAnalysisTests data are installed from the independent published test suite.
"""
import gzip
import json
import shutil
import warnings
from pathlib import Path

import MDAnalysis as mda
import pytest

from MDAnalysisTests.datafiles import (
    PDB, XTC, GRO_MEMPROT, XTC_MEMPROT,
    PDB_sub_sol, XTC_sub_sol, TPR, GRO, TPR_xvf,
)
from minitopol import Workspace, Registry, convert
from minitopol.atom_mapping import validate_source_mapping


@pytest.mark.parametrize("label,topology,xtc,compressed", [
    ("adk-pdb", PDB, XTC, False),
    ("cobrotoxin-pdb", PDB_sub_sol, XTC_sub_sol, False),
    ("yiip-membrane-gro", GRO_MEMPROT, XTC_MEMPROT, True),
    ("cobrotoxin-tpr", TPR_xvf, XTC_sub_sol, False),
])
def test_real_trajectories_from_distinct_molecular_systems(
    label, topology, xtc, compressed, tmp_path,
):
    if compressed:
        path = tmp_path / (label + ".gro")
        with gzip.open(topology, "rb") as original, path.open("wb") as output:
            shutil.copyfileobj(original, output)
        topology = str(path)
    # Confirm MDAnalysis itself can combine the independent source pair.
    universe = mda.Universe(topology, xtc)
    try:
        atom_count = len(universe.atoms)
        frames = len(universe.trajectory)
    finally:
        universe.trajectory.close()
    print("REAL_SOURCE", json.dumps({
        "label":label, "source_atom_count":atom_count, "frames":frames,
        "topology":str(Path(topology).name),
        "trajectory":str(Path(xtc).name),
    }), flush=True)
    workspace = Workspace(tmp_path / "workspace")
    try:
        tid = convert(workspace, topology, xtc, chunk_frames=2,
                      solvent_resnames=["SOL", "WAT", "HOH", "TIP3", "SPC"])
    except Exception as error:
        print("REAL_REJECTION", json.dumps({
            "label":label, "error":f"{type(error).__name__}: {error}"
        }), flush=True)
        raise
    manifest = Registry(workspace).manifest(tid)
    print("REAL_ACCEPTANCE", json.dumps({
        "label":label, "frames":manifest["frames"], "A_atoms":manifest["atoms"],
        "input_validation":manifest["input_validation"],
        "format":Path(topology).suffix.lower(),
    }, sort_keys=True), flush=True)
    assert manifest["status"] == "complete"
    assert manifest["frames"] == frames
    assert manifest["atoms"] > 0


def test_real_adk_full_gro_to_tpr_mapping(tmp_path):
    source, target = mda.Universe(GRO), mda.Universe(TPR)
    try:
        print("FULL_TPR_MAPPING", json.dumps({
            "source_atoms":len(source.atoms), "target_atoms":len(target.atoms),
        }), flush=True)
        assert len(source.atoms) == len(target.atoms)
        mapping = list(range(len(source.atoms)))
    finally:
        source.trajectory.close()
        target.trajectory.close()
    assert validate_source_mapping(GRO, TPR, mapping) == mapping


def test_real_adk_middle_residue_subset_to_full_tpr_mapping(tmp_path):
    """A real, physically correct subset must not be rejected by renumbering.

    The source was extracted in the original TPR order; target indices are
    taken from the *original actual atoms*, not guessed from residue names.
    """
    universe = mda.Universe(TPR, XTC)
    try:
        fragment = universe.residues[40:48].atoms
        exact_indices = [int(x) for x in fragment.indices]
        assert len(fragment) > 15
        fragment_gro = tmp_path / "real-subset.gro"
        fragment_xtc = tmp_path / "real-subset.xtc"
        universe.trajectory[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fragment.write(str(fragment_gro))
        with mda.Writer(str(fragment_xtc), n_atoms=len(fragment)) as writer:
            for frame in range(min(len(universe.trajectory), 3)):
                universe.trajectory[frame]
                writer.write(fragment)
    finally:
        universe.trajectory.close()
    # This step confirms that a real physical residue subset is still a
    # valid GRO+XTC input before evaluating its explicit TPR correspondence.
    ws = Workspace(tmp_path / "fragment-workspace")
    tid = convert(ws, fragment_gro, fragment_xtc, chunk_frames=2, solvent_resnames=[])
    assert Registry(ws).manifest(tid)["status"] == "complete"
    try:
        result = validate_source_mapping(fragment_gro, TPR, exact_indices)
    except Exception as error:
        print("REAL_SUBSET_REJECTION", json.dumps({
            "source_atoms":len(exact_indices),
            "first_real_tpr_atom":exact_indices[0],
            "last_real_tpr_atom":exact_indices[-1],
            "error":f"{type(error).__name__}: {error}"
        }), flush=True)
        raise
    print("REAL_SUBSET_ACCEPTED", json.dumps({
        "source_atoms":len(exact_indices), "first_real_tpr_atom":exact_indices[0],
    }), flush=True)
    assert result == exact_indices
