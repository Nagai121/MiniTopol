"""Independent real MDAnalysisTests acceptance tests for the early TPR feature baseline.

Real GROMACS topologies/trajectories, no fake topology mock or atom labels.
This deliberately runs on commit 8 before the strict source mapping additions.
"""
import gzip
import shutil
import json
from pathlib import Path

import pytest
import MDAnalysis as mda
import numpy as np

pytest.importorskip("MDAnalysisTests")
from MDAnalysisTests.datafiles import (
    TPR, XTC, GRO, PDB, TPR_xvf, XTC_sub_sol, PDB_sub_sol,
    GRO_MEMPROT, XTC_MEMPROT
)
from minitopol import Registry, Workspace, convert
from minitopol.descriptors import analyze


@pytest.mark.parametrize("label,topology,trajectory,compressed,full_tpr", [
    ("adk-tpr", TPR, XTC, False, True),
    ("cobrotoxin-tpr", TPR_xvf, XTC_sub_sol, False, True),
    ("adk-gro-legacy", GRO, XTC, False, False),
    ("adk-pdb-legacy", PDB, XTC, False, False),
    ("cobrotoxin-pdb-legacy", PDB_sub_sol, XTC_sub_sol, False, False),
    ("yiip-gro-legacy", GRO_MEMPROT, XTC_MEMPROT, True, False),
])
def test_real_molecular_trajectory_accepted(tmp_path,label,topology,trajectory,compressed,full_tpr):
    if compressed:
        dest=tmp_path/"uncompressed.gro"
        with gzip.open(topology,"rb") as src, dest.open("wb") as out:
            shutil.copyfileobj(src,out)
        topology=str(dest)
    u=mda.Universe(topology,trajectory)
    try:
        frames=len(u.trajectory)
        source_atoms=len(u.atoms)
    finally:
        u.trajectory.close()
    ws=Workspace(tmp_path/"ws")
    tid=convert(ws,topology,trajectory,solvent_resnames=["SOL","WAT","HOH","TIP3","SPC"],chunk_frames=2)
    registry=Registry(ws)
    manifest=registry.manifest(tid)
    assert manifest["status"]=="complete"
    assert manifest["frames"]==frames
    assert 0<manifest["atoms"]<=source_atoms
    if full_tpr:
        molecules,atoms=registry.molecule_data(tid)
        assert atoms.num_rows==source_atoms
        assert molecules.num_rows>0
        assert registry.bond_info(tid) is not None
        result=analyze(ws,tid,descriptors=["gyration","electrostatic_score"],
                       coordinate_mode="nonperiodic",chunk_frames=2)
        assert result["status"]=="complete",result
        assert all(x["status"]=="complete" for x in result["results"].values())
        assert len(ws.query_results(descriptor="gyration").to_pylist())==frames
    print("BASELINE_REAL",json.dumps({"label":label,"frames":frames,
          "source_atoms":source_atoms,"a_atoms":manifest["atoms"],
          "direct_tpr":full_tpr}),flush=True)


def test_mismatched_real_tpr_and_xtc_rejected(tmp_path):
    with pytest.raises((ValueError, IndexError, OSError)):
        convert(Workspace(tmp_path/"wrong"),TPR,XTC_sub_sol,
                solvent_resnames=["SOL","WAT","HOH"],chunk_frames=2)
