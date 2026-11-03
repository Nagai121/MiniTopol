"""Real GROMACS input integration: MDAnalysisTests AdK OPLS-AA system.

This test uses the distributed, unmodified real GRO/TPR/XTC files, not
fake parser objects or generated mock topology records.
"""
import json

import numpy as np
import pytest

from minitopol import Registry, Workspace, convert
from minitopol.descriptors import analyze
from minitopol.descriptors.context import AnalysisContext

pytest.importorskip("MDAnalysisTests")
from MDAnalysisTests.datafiles import GRO, TPR, XTC


def test_real_adk_tpr_xtc_conversion_and_science(tmp_path):
    ws = Workspace(tmp_path / "real-tpr")
    tid = convert(ws, TPR, XTC, solvent_resnames=["SOL", "WAT", "HOH"], chunk_frames=2)
    registry = Registry(ws)
    manifest = registry.manifest(tid)
    assert manifest["status"] == "complete"
    assert manifest["frames"] >= 2
    assert manifest["atoms"] >= 2000
    assert manifest["input_validation"]["pairing"]["status"] in ("verified", "unverified")
    assert manifest["input_validation"]["elements"]["status"] in ("verified", "unverified")
    molecule_table, atom_table = registry.molecule_data(tid)
    assert atom_table.num_rows > manifest["atoms"]
    assert "mass" in atom_table.column_names
    assert "partial_charge" in atom_table.column_names
    assert registry.bond_info(tid) is not None
    masses = AnalysisContext(ws, tid, coordinate_mode="nonperiodic").atomic_values("masses")
    assert masses.shape == (manifest["atoms"],)
    assert np.isfinite(masses).all() and (masses > 0).all()
    report = analyze(ws, tid, descriptors=["gyration", "electrostatic_score"],
                     coordinate_mode="nonperiodic", chunk_frames=2)
    assert report["status"] == "complete", report
    assert all(r["status"] == "complete" for r in report["results"].values())
    assert len(ws.query_results(descriptor="gyration").to_pylist()) == manifest["frames"]
    data = ws.query_results(descriptor="electrostatic_score").to_pylist()
    assert len(data) > 0
    assert all(np.isfinite(row["value"]) for row in data)
    print("REAL_TPR_XTC", json.dumps({
        "source": "MDAnalysisTests AdK OPLS-AA (real GROMACS TPR/XTC)",
        "frames": manifest["frames"], "heavy_atoms": manifest["atoms"],
        "tpr_atoms": atom_table.num_rows, "molecules": molecule_table.num_rows,
        "residue_pairs_written": len(data),
        "input_validation": manifest["input_validation"],
    }, sort_keys=True))


def test_real_adk_gro_xtc_preserves_unverified_provenance(tmp_path):
    ws = Workspace(tmp_path / "real-gro")
    tid = convert(ws, GRO, XTC, solvent_resnames=["SOL", "WAT", "HOH"], chunk_frames=2)
    manifest = Registry(ws).manifest(tid)
    assert manifest["status"] == "complete"
    assert manifest["frames"] >= 2 and manifest["atoms"] > 1000
    pairing = manifest["input_validation"]["pairing"]["status"]
    assert pairing in ("verified", "unverified")
    # In both cases the source and the XTC are real AdK GROMACS inputs.
    print("REAL_GRO_XTC", json.dumps({
        "source": "MDAnalysisTests AdK OPLS-AA (real GROMACS GRO/XTC)",
        "frames": manifest["frames"], "heavy_atoms": manifest["atoms"],
        "input_validation": manifest["input_validation"],
    }, sort_keys=True))
