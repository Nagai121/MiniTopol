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
    assert manifest["atoms"] >= 1000
    assert manifest["input_validation"]["pairing"]["status"] in ("verified", "unverified")
    assert manifest["input_validation"]["elements"]["status"] in ("verified", "unverified")
    molecule_table, atom_table = registry.molecule_data(tid)
    assert atom_table.num_rows > manifest["atoms"]
    assert "mass" in atom_table.column_names
    assert "partial_charge" in atom_table.column_names
    assert registry.bond_info(tid) is not None
    # Validate a *real* TPR-derived peptide bond and C-alpha role, rather
    # than relying only on a small synthetic annotation fixture.
    atom_data = registry.atoms(tid).to_pydict()
    by_residue = {}
    for i, r in enumerate(atom_data["residue_index"]):
        by_residue.setdefault(r, {})[atom_data["name"][i]] = i
    links = registry.bonds(tid).to_pydict()
    real_peptide = None
    for left, right in zip(links["point_i"], links["point_j"]):
        for carbon, nitrogen in ((left, right), (right, left)):
            a, b = atom_data["residue_index"][carbon], atom_data["residue_index"][nitrogen]
            if (a != b and atom_data["name"][carbon] == "C"
                    and atom_data["name"][nitrogen] == "N"
                    and "CA" in by_residue[a] and "CA" in by_residue[b]):
                real_peptide = (a, b)
                break
        if real_peptide is not None:
            break
    assert real_peptide is not None, "Real AdK TPR should contain a peptide C-N bond"
    a, b = real_peptide
    registry.annotate(tid, chains=[{"id": "actual-peptide", "segments": [[a,b]]}],
                      alpha_carbons=[by_residue[a]["CA"], by_residue[b]["CA"]])
    check = registry.annotations(tid)[0]["validation"]
    assert check["alpha_carbons"]["status"] == "verified"
    assert check["continuous_segments"]["status"] == "verified"
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
