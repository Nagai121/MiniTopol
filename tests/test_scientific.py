import json
from pathlib import Path
import warnings
import shutil
import subprocess

import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals
import numpy as np
import pyarrow as pa
import pytest

from minitopol import Registry, Workspace, convert
from minitopol.coordinates import minimum_vectors
from minitopol.descriptors import analyze
from minitopol.descriptors.kernels import sasa_atoms, secondary_labels


def import_frames(root, xyz, *, names=None, residue_indices=None, residue_names=None, times=None, boxes=None, chunk=3):
    root.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    atoms = xyz.shape[1]
    names = names or ["C"] * atoms
    residue_indices = residue_indices if residue_indices is not None else [0] * atoms
    residue_names = residue_names or ["MOL"] * (max(residue_indices) + 1)
    u = mda.Universe.empty(atoms, n_residues=len(residue_names), atom_resindex=residue_indices, trajectory=True)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("resnames", residue_names)
    u.add_TopologyAttr("resids", np.arange(len(residue_names)) + 1)
    u.atoms.positions = xyz[0]
    u.dimensions = [100, 100, 100, 90, 90, 90] if boxes is None else boxes[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u.atoms.write(str(root / "input.gro"))
    with mda.Writer(str(root / "input.xtc"), n_atoms=atoms, precision=5) as writer:
        for frame in range(len(xyz)):
            u.atoms.positions = xyz[frame]
            u.trajectory.ts.time = frame * 2.5 if times is None else times[frame]
            u.dimensions = [100, 100, 100, 90, 90, 90] if boxes is None else boxes[frame]
            writer.write(u.atoms)
    ws = Workspace(root / "workspace")
    tid = convert(ws, root / "input.gro", root / "input.xtc", solvent_resnames=[], chunk_frames=chunk)
    return ws, tid


@pytest.fixture
def protein(tmp_path):
    centers = np.array([[0, 0, 0], [3, 1, 0], [5, 4, 2], [2, 5, 3], [0, 2, 2]], dtype=float)
    offsets = np.array([[-1, .4, -.4], [0, 0, 0], [1.1, .5, .2], [1.4, 1.4, .4]])
    base = (centers[:, None] + offsets).reshape(-1, 3)
    xyz = []
    rates = np.random.default_rng(9).normal(size=base.shape) * .06
    for frame in range(8):
        angle = frame * .19
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        xyz.append((base + rates * frame) @ rotation + np.array([2*frame, -frame, .4*frame]))
    kwargs = dict(names=["N", "CA", "C", "O"] * 5, residue_indices=np.repeat(np.arange(5), 4).tolist(), residue_names=["ALA"] * 5)
    ws, tid = import_frames(tmp_path / "protein", xyz, **kwargs)
    Registry(ws).annotate(tid, chains=[{"id": "A", "segments": [[0, 1, 2, 3, 4]]}])
    return ws, tid, kwargs


def rows(ws, name):
    return ws.query_results(descriptor=name).to_pylist()


def test_all_ten_match_direct_reference_and_parquet_catalog(protein):
    ws, tid, _ = protein
    report = analyze(ws, tid, coordinate_mode="nonperiodic", chunk_frames=3, pair_block=2, fft_block=3)
    assert report["status"] == "complete", report
    assert len(report["results"]) == 10
    registry = Registry(ws)
    assert registry.representations() == ["A"]
    raw = np.concatenate([b.positions for b in registry.blocks(tid)]).astype(float)
    ca = np.arange(1, 20, 4)
    aligned = []
    # Independent least-squares reference (MDTraj), compared after fitting on CA.
    md = pytest.importorskip("mdtraj")
    top = md.Topology()
    chain = top.add_chain()
    for residue in range(5):
        r = top.add_residue("ALA", chain)
        for name in ("N", "CA", "C", "O"):
            top.add_atom(name, md.element.get_by_symbol(name[0]), r)
    traj = md.Trajectory(raw.astype(np.float32)/10, top)
    traj.superpose(traj, 0, atom_indices=ca, parallel=False)
    aligned = traj.xyz.astype(float) * 10
    expected_rmsd = np.sqrt(((aligned - aligned[0])**2).sum(axis=(1, 2)) / 20)
    found = sorted(rows(ws, "rmsd"), key=lambda r: r["frame"])
    np.testing.assert_allclose([r["value"] for r in found], expected_rmsd, atol=3e-5)
    mean = aligned.mean(axis=0)
    delta = aligned - mean
    variance = (delta**2).sum(axis=2).mean(axis=0)
    found = sorted(rows(ws, "rmsf"), key=lambda r: r["point"])
    np.testing.assert_allclose([r["value"] for r in found], np.sqrt(variance), atol=3e-5)
    expected_msd = [0.] + [np.mean(np.sum((aligned[k:] - aligned[:-k])**2, axis=2)) for k in range(1, 8)]
    found = sorted(rows(ws, "msd"), key=lambda r: r["lag_frames"])
    np.testing.assert_allclose([r["value"] for r in found], expected_msd, atol=3e-5)
    assert [r["origins"] for r in found] == list(range(8, 0, -1))
    for r in rows(ws, "time_correlation"):
        point, k = r["point"], r["lag_frames"]
        expected = np.mean(np.sum(delta[:8-k, point] * delta[k:, point], axis=1)) / variance[point]
        assert r["value"] == pytest.approx(expected, abs=4e-4)
    for r in rows(ws, "dynamic_correlation"):
        i, j = ca[r["residue_i"]], ca[r["residue_j"]]
        expected = np.mean(np.sum(delta[:, i] * delta[:, j], axis=1)) / np.sqrt(variance[i] * variance[j])
        assert r["value"] == pytest.approx(expected, abs=4e-4)
    masses = np.tile([14.007, 12.011, 12.011, 15.999], 5)
    center = np.einsum("faj,a->fj", raw, masses) / masses.sum()
    rg = np.sqrt(np.einsum("faj,faj,a->f", raw-center[:, None], raw-center[:, None], masses) / masses.sum())
    found = sorted(rows(ws, "gyration"), key=lambda r: r["frame"])
    np.testing.assert_allclose([r["value"] for r in found], rg, atol=1e-10)
    for r in rows(ws, "ramachandran"):
        i, f = r["residue"], r["frame"]
        if (r["angle"] == "phi" and i == 0) or (r["angle"] == "psi" and i == 4):
            assert r["value"] is None and r["reason"] == "segment_terminal"
        else:
            quad = [4*(i-1)+2, 4*i, 4*i+1, 4*i+2] if r["angle"] == "phi" else [4*i, 4*i+1, 4*i+2, 4*(i+1)]
            expected = (np.rad2deg(calc_dihedrals(*raw[f, quad])) + 180) % 360 - 180
            assert r["value"] == pytest.approx(expected, abs=1e-5)
    contacts = rows(ws, "contacts")
    expected_counts = {}
    for i in range(5):
        for j in range(i+3, 5):
            count = 0
            for f in range(8):
                distances = np.linalg.norm(raw[f, i*4:(i+1)*4, None] - raw[f, None, j*4:(j+1)*4], axis=2)
                count += int(distances.min() <= 4.5)
            if count:
                expected_counts[i, j] = count
    found = {(r["residue_i"], r["residue_j"]): r["contact_frames"] for r in contacts if r["level"] == "frequency"}
    assert found == expected_counts
    assert sum(r["level"] == "state" for r in contacts) == sum(expected_counts.values())
    sasa = rows(ws, "sasa")
    for frame in range(8):
        atom_sum = sum(r["value"] for r in sasa if r["frame"] == frame and r["level"] == "atom")
        residue_sum = sum(r["value"] for r in sasa if r["frame"] == frame and r["level"] == "residue")
        total = next(r["value"] for r in sasa if r["frame"] == frame and r["level"] == "total")
        assert atom_sum == pytest.approx(total)
        assert residue_sum == pytest.approx(total)
    raw_traj = md.Trajectory(raw.astype(np.float32)/10, top)
    reference_sasa = md.shrake_rupley(raw_traj, probe_radius=.14, n_sphere_points=960).sum(axis=1) * 100
    ours = [next(r["value"] for r in sasa if r["frame"] == frame and r["level"] == "total") for frame in range(8)]
    np.testing.assert_allclose(ours, reference_sasa, rtol=.025)
    assert len(rows(ws, "secondary_structure")) == 8*5
    assert not list((ws.path / "work").iterdir())
    for name, entry in report["results"].items():
        assert ws.query_results(definition_id=entry["definition_id"]).num_rows == len(rows(ws, name))
    assert ws.query_results().num_rows == sum(entry["rows"] for entry in report["results"].values())


def test_secondary_boundaries_and_sasa_known_spheres():
    labels = secondary_labels(np.array([-100, -30, -120, -120, 20, np.nan]),
                              np.array([-80, -5, 120, -170, 20, -40]))
    assert list(labels) == ["helix_like", "helix_like", "extended_like", "extended_like", "other", None]
    radius = 1.7
    assert sasa_atoms(np.zeros((1, 3)), np.array([radius]))[0] == pytest.approx(4*np.pi*(radius+1.4)**2)
    # Two equal spheres separated by d have one buried cap each.
    d, expanded = 3., radius+1.4
    expected = 4*np.pi*expanded**2 - 2*np.pi*expanded*(expanded-d/2)
    low = sasa_atoms(np.array([[0.,0,0],[d,0,0]]), np.array([radius,radius]))
    high = sasa_atoms(np.array([[0.,0,0],[d,0,0]]), np.array([radius,radius]), sphere_count=3840)
    np.testing.assert_allclose(low, expected, atol=1.)
    np.testing.assert_allclose(high, expected, atol=.5)


def test_nojump_and_triclinic_minimum_image(tmp_path):
    base = np.array([[1.,1,1],[2,2,1],[2,1,2]])
    xyz = np.array([(base + [3*f,0,0]) % 10 for f in range(5)])
    ws, tid = import_frames(tmp_path, xyz, boxes=[[10,10,10,90,90,90]]*5, chunk=2)
    registry = Registry(ws)
    blocks = list(registry.coordinate_blocks(tid, mode="nojump", initial_whole=True, half_box_steps=True, chunk_frames=2))
    np.testing.assert_allclose(np.concatenate([b.positions for b in blocks]), np.array([base+[3*f,0,0] for f in range(5)]), atol=1e-4)
    np.testing.assert_allclose(next(registry.blocks(tid)).positions, xyz[:2], atol=1e-4)
    with pytest.raises(ValueError, match="half-box"):
        list(registry.coordinate_blocks(tid, mode="nojump", initial_whole=True))
    with pytest.raises(ValueError, match="initial"):
        list(registry.coordinate_blocks(tid, mode="nojump", half_box_steps=True))
    box = np.array([[10.,0,0],[3,9,0],[2,1,8]])
    vector = np.array([[8.,8,7],[-9,5,1],[4,-6,5]])
    actual = minimum_vectors(vector, box)
    images = np.array([[i,j,k] for i in range(-2,3) for j in range(-2,3) for k in range(-2,3)]) @ box
    expected = [np.min(np.linalg.norm(v-images,axis=1)) for v in vector]
    np.testing.assert_allclose(np.linalg.norm(actual,axis=1), expected, atol=2e-6)


def test_all_ten_partition_invariance(protein, tmp_path):
    ws, tid, _ = protein
    other = Workspace(shutil.copytree(ws.path, tmp_path / "copy"))
    first = analyze(ws, tid, coordinate_mode="nonperiodic", chunk_frames=2, fft_block=2, pair_block=2)
    second = analyze(other, tid, coordinate_mode="nonperiodic", chunk_frames=5, fft_block=5, pair_block=3)
    assert first["status"] == second["status"] == "complete"
    for name in first["results"]:
        assert first["results"][name]["definition_id"] == second["results"][name]["definition_id"]
        def key(row):
            return json.dumps({k:v for k,v in row.items() if k not in ("value", "distance_angstrom")}, sort_keys=True)
        a, b = sorted(rows(ws, name), key=key), sorted(rows(other, name), key=key)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            for field in x:
                if isinstance(x[field], float):
                    assert x[field] == pytest.approx(y[field], abs=1e-10)
                else:
                    assert x[field] == y[field]


def test_static_zero_variance_and_all_lags(protein, tmp_path):
    ws, tid, kwargs = protein
    base = next(Registry(ws).blocks(tid)).positions[0]
    ws, tid = import_frames(tmp_path / "static", np.repeat(base[None], 6, axis=0), **kwargs)
    report = analyze(ws, tid, descriptors=["rmsf", "msd", "time_correlation", "dynamic_correlation"],
                     coordinate_mode="nonperiodic", fft_block=2)
    assert report["status"] == "complete", report
    assert all(r["value"] == 0 for r in rows(ws, "rmsf") + rows(ws, "msd"))
    assert len(rows(ws, "msd")) == 6
    assert all(r["value"] is None and r["reason"] == "zero_variance"
               for r in rows(ws, "time_correlation") + rows(ws, "dynamic_correlation"))


def test_missing_chain_continuity_is_not_invented(protein, tmp_path):
    ws, tid, kwargs = protein
    raw = np.concatenate([b.positions for b in Registry(ws).blocks(tid)])
    ws, tid = import_frames(tmp_path / "unannotated", raw, **kwargs)
    report = analyze(ws, tid, descriptors=["ramachandran", "secondary_structure", "contacts"], coordinate_mode="nonperiodic")
    assert report["results"]["ramachandran"]["status"] == "complete"
    assert report["results"]["contacts"]["status"] == "failed"
    assert all(r["value"] is None and r["reason"] == "unconfirmed_chain_continuity"
               for r in rows(ws, "ramachandran") + rows(ws, "secondary_structure"))
    with pytest.raises(LookupError):
        ws.query_results(descriptor="contacts")


def test_irregular_time_is_rejected_without_dropping_frames(protein, tmp_path):
    ws, tid, kwargs = protein
    raw = np.concatenate([b.positions for b in Registry(ws).blocks(tid)])
    ws, tid = import_frames(tmp_path / "irregular", raw, times=[0,1,2,4,5,6,7,8], **kwargs)
    report = analyze(ws, tid, descriptors=["msd", "time_correlation", "rmsd"], coordinate_mode="nonperiodic")
    assert report["results"]["rmsd"]["status"] == "complete"
    assert len(rows(ws, "rmsd")) == 8
    for name in ("msd", "time_correlation"):
        assert report["results"][name]["status"] == "failed"
        assert "Irregular timestamps" in report["results"][name]["reason"]
        with pytest.raises(LookupError):
            ws.query_results(descriptor=name)


def test_unsupported_radius_requires_override(tmp_path):
    ws, tid = import_frames(tmp_path, [[[0.,0,0]]], names=["ZN"])
    report = analyze(ws, tid, descriptors=["sasa", "gyration"], coordinate_mode="nonperiodic")
    assert report["results"]["sasa"]["status"] == "failed"
    assert "Undefined radii" in report["results"]["sasa"]["reason"]
    assert report["results"]["gyration"]["status"] == "complete"
    Registry(ws).annotate(tid, chains=[], radii={"0": 1.39})
    report = analyze(ws, tid, descriptors=["sasa"], coordinate_mode="nonperiodic")
    assert report["status"] == "complete", report
    assert rows(ws, "sasa")[0]["value"] == pytest.approx(4*np.pi*(1.39+1.4)**2)


def test_periodic_contact_distinguishes_distance_from_unwrapping(tmp_path):
    xyz = np.array([[[.5,1,1],[.7,1.2,1],[.9,1,1],[9.,1,1],[9.2,1.2,1],[9.4,1,1]]])
    ws, tid = import_frames(tmp_path, xyz, names=["N","CA","C"]*2, residue_indices=[0]*3+[1]*3,
                            residue_names=["ALA"]*2, boxes=[[10,10,10,90,90,90]])
    Registry(ws).annotate(tid, chains=[{"id":"A","segments":[[0]]},{"id":"B","segments":[[1]]}])
    report = analyze(ws, tid, descriptors=["contacts"], coordinate_mode="continuous", initial_whole=True)
    assert report["status"] == "complete", report
    found = [r for r in rows(ws,"contacts") if r["level"] == "state"]
    assert len(found) == 1
    assert found[0]["distance_angstrom"] == pytest.approx(1.1, abs=1e-5)
    with pytest.raises(ValueError, match="invertible"):
        minimum_vectors(np.array([[1.,2,3]]), np.zeros((3,3)))


def test_partial_dataset_and_completed_marker_recovery(protein, monkeypatch):
    from minitopol.descriptors.datasets import DatasetWriter
    from minitopol.descriptors import ResultStore
    ws, tid, _ = protein
    append = DatasetWriter.append
    calls = []
    def interrupted(self, key, table):
        append(self, key, table)
        raise RuntimeError("injected interruption after first committed part")
    monkeypatch.setattr(DatasetWriter, "append", interrupted)
    report = analyze(ws, tid, descriptors=["rmsd"], coordinate_mode="nonperiodic", chunk_frames=3)
    assert report["status"] == "incomplete"
    with pytest.raises(LookupError):
        ws.query_results(descriptor="rmsd")
    def tracked(self,key,table):
        calls.append(key)
        return append(self,key,table)
    monkeypatch.setattr(DatasetWriter, "append", tracked)
    report = analyze(ws, tid, descriptors=["rmsd"], coordinate_mode="nonperiodic", chunk_frames=3)
    assert report["status"] == "complete", report
    assert calls == [3,6]
    assert len(rows(ws,"rmsd")) == 8
    result = report["results"]["rmsd"]["result_id"]
    with ws.catalog() as con:
        con.execute("DELETE FROM result_parts WHERE result_id=?", [result])
        con.execute("DELETE FROM results WHERE id=?", [result])
    assert ResultStore(ws).recover() == 1
    assert len(rows(ws,"rmsd")) == 8
    report = analyze(ws, tid, descriptors=["rmsd"], coordinate_mode="nonperiodic", chunk_frames=5)
    assert report["status"] == "complete"
    with ws.catalog() as con:
        assert con.execute("SELECT count(*) FROM result_parts").fetchone() == (3,)


def test_changing_box_nojump_matches_mdanalysis(tmp_path):
    from MDAnalysis.lib.mdamath import triclinic_vectors
    from MDAnalysis.transformations.nojump import NoJump
    dimensions = [[10+.2*f,11+.1*f,12+.15*f,85,80,75] for f in range(6)]
    initial = np.array([[.1,.1,.1],[.2,.2,.1],[.2,.1,.2]])
    xyz = np.array([((initial+[.3*f,0,0])%1) @ triclinic_vectors(d) for f,d in enumerate(dimensions)])
    ws, tid = import_frames(tmp_path, xyz, boxes=dimensions, chunk=2)
    found = np.concatenate([b.positions for b in Registry(ws).coordinate_blocks(
        tid,mode="nojump",initial_whole=True,half_box_steps=True,chunk_frames=2)])
    u = mda.Universe(str(tmp_path/'input.gro'), str(tmp_path/'input.xtc'))
    u.trajectory.add_transformations(NoJump())
    expected = np.array([ts.positions.copy() for ts in u.trajectory])
    u.trajectory.close()
    np.testing.assert_allclose(found,expected,atol=1e-4)


def test_msd_accumulator_is_not_double_counted_after_interruption(protein, monkeypatch):
    from minitopol.descriptors import msd
    from minitopol.descriptors.context import AnalysisContext
    ws, tid, _ = protein
    original = msd.atomic_json
    def interrupt(path, data):
        if data.get("atoms_done") == 2:
            raise RuntimeError("interrupt before accumulator generation commit")
        return original(path,data)
    monkeypatch.setattr(msd,"atomic_json",interrupt)
    report=analyze(ws,tid,descriptors=["msd"],coordinate_mode="nonperiodic",fft_block=3)
    assert report["status"] == "incomplete"
    with pytest.raises(LookupError):
        ws.query_results(descriptor="msd")
    monkeypatch.setattr(msd,"atomic_json",original)
    report=analyze(ws,tid,descriptors=["msd"],coordinate_mode="nonperiodic",fft_block=3)
    assert report["status"] == "complete",report
    context=AnalysisContext(ws,tid,coordinate_mode="nonperiodic",fft_block=3)
    with context.cache_lock:
        xyz=np.concatenate([p for _,_,p,_ in context.aligned_blocks()])
    expected=[0.]+[np.mean(np.sum((xyz[k:]-xyz[:-k])**2,axis=2)) for k in range(1,len(xyz))]
    found=sorted(rows(ws,"msd"),key=lambda r:r["lag_frames"])
    np.testing.assert_allclose([r["value"] for r in found],expected,atol=1e-11)


def test_dataset_registration_transaction_rolls_back_parent_and_parts(protein, monkeypatch):
    from contextlib import contextmanager
    from minitopol.descriptors import ResultStore
    ws,tid,_=protein
    catalog=ws.catalog
    class Connection:
        def __init__(self,con): self.con=con
        def execute(self,sql,*args):
            if sql.startswith("INSERT INTO result_parts"):
                raise RuntimeError("injected database part registration failure")
            return self.con.execute(sql,*args)
    @contextmanager
    def broken():
        with catalog() as con:
            yield Connection(con)
    monkeypatch.setattr(ws,"catalog",broken)
    report=analyze(ws,tid,descriptors=["rmsd"],coordinate_mode="nonperiodic")
    assert report["status"] == "incomplete"
    monkeypatch.setattr(ws,"catalog",catalog)
    with ws.catalog() as con:
        assert con.execute("SELECT count(*) FROM results").fetchone() == (0,)
        assert con.execute("SELECT count(*) FROM result_parts").fetchone() == (0,)
    assert ResultStore(ws).recover() == 1
    assert len(rows(ws,"rmsd")) == 8


def test_missing_backbone_atom_does_not_create_dihedrals(protein,tmp_path):
    ws,tid,kwargs=protein
    raw=np.concatenate([b.positions for b in Registry(ws).blocks(tid)])
    keep=np.array([i for i in range(20) if i!=6])  # missing C in residue 1
    kwargs={**kwargs,"names":[kwargs['names'][i] for i in keep],
            "residue_indices":[kwargs['residue_indices'][i] for i in keep]}
    ws,tid=import_frames(tmp_path/'missing',raw[:,keep],**kwargs)
    Registry(ws).annotate(tid,chains=[{"id":"A","segments":[[0,1,2,3,4]]}])
    report=analyze(ws,tid,descriptors=['ramachandran','secondary_structure'],coordinate_mode='nonperiodic')
    assert report['status']=='complete'
    affected=[r for r in rows(ws,'ramachandran') if r['residue']==1 or (r['residue']==2 and r['angle']=='phi')]
    assert all(r['value'] is None and r['reason']=='missing_or_ambiguous_backbone_atom' for r in affected)
    assert report['results']['ramachandran']['undefined_values']>0


def test_kabsch_does_not_apply_a_reflection():
    from minitopol.coordinates import fit_positions
    reference=np.array([[0.,0,0],[2,0,0],[0,2,0],[0,0,2]])
    reflected=reference*np.array([-1,1,1])
    fitted=fit_positions(reflected,reference,np.arange(4))
    assert np.linalg.norm(fitted-reference)>1
    assert np.linalg.det(fitted[1:]-fitted[0])<0


def test_cli_runs_all_ten_and_reports_completion(protein,tmp_path):
    import sys
    ws,tid,_=protein
    result=subprocess.run([sys.executable,'-m','minitopol','--workspace',str(ws.path),
                           'calculate',tid,'--coordinates','nonperiodic'],
                          text=True,capture_output=True,check=True)
    report=json.loads(result.stdout)
    assert report['status']=='complete'
    assert len(report['results'])==10
    assert all(r['status']=='complete' for r in report['results'].values())


def test_nonprotein_atom_name_ca_does_not_block_contact_distances(tmp_path):
    ws,tid=import_frames(tmp_path,[[[0.,0,0],[1.,0,0]]],names=['CA','CA'],
                         residue_indices=[0,1],residue_names=['MOL','MOL'])
    report=analyze(ws,tid,descriptors=['contacts'],coordinate_mode='nonperiodic')
    assert report['status']=='complete',report
    assert [r['value'] for r in rows(ws,'contacts') if r['level']=='frequency']==[1.]


@pytest.mark.parametrize('point,value',[(1.5,12.),(True,12.),(1,float('nan'))])
def test_invalid_atomic_overrides_do_not_change_existing_annotations(protein,point,value):
    ws,tid,_=protein
    registry=Registry(ws)
    original=registry.annotations(tid)
    with pytest.raises(ValueError):
        registry.annotate(tid,chains=[{"id":"A","segments":[[0,1,2,3,4]]}],masses={point:value})
    assert registry.annotations(tid)==original


def test_cache_cleanup_preserves_unrecognized_user_file(protein,monkeypatch):
    from minitopol.descriptors.context import AnalysisContext
    ws,tid,_=protein
    original=AnalysisContext.prepare_aligned
    saved=[]
    def wrapped(ctx):
        original(ctx)
        path=ctx.cache_path/'user-note.txt'
        path.write_text('keep this user data')
        saved.append(path)
    monkeypatch.setattr(AnalysisContext,'prepare_aligned',wrapped)
    report=analyze(ws,tid,descriptors=['rmsd'],coordinate_mode='nonperiodic')
    assert report['status']=='complete'
    assert 'cache_retained' in report
    assert saved[0].read_text()=='keep this user data'


@pytest.mark.parametrize('damage', ['positions', 'time', 'changed_values', 'missing_check', 'legacy'])
def test_damaged_alignment_cache_is_never_published(protein, damage):
    from minitopol.descriptors.context import AnalysisContext
    from minitopol.storage import atomic_json, read_json
    ws, tid, _ = protein
    options = dict(coordinate_mode='nonperiodic', chunk_frames=3)
    ctx = AnalysisContext(ws, tid, **options)
    with ctx.cache_lock:
        ctx.prepare_aligned()
        if damage in ('positions', 'time'):
            chunks = sorted(p for p in (ctx.cache_path / 'aligned.zarr' / damage / 'c').rglob('*') if p.is_file())
            chunks[1].unlink()
        elif damage == 'changed_values':
            ctx.cache['positions'][3, 0, 0] += 1.0
        elif damage == 'missing_check':
            (ctx.cache_path / 'aligned-checksums' / '3').unlink()
        else:
            state = read_json(ctx.cache_path / 'state')
            state.pop('schema')
            atomic_json(ctx.cache_path / 'state', state)
    report = analyze(ws, tid, descriptors=['rmsf'], **options)
    assert report['status'] == 'incomplete'
    assert 'cache' in report['results']['rmsf']['reason']
    with ws.catalog() as con:
        assert con.execute('SELECT count(*) FROM results').fetchone() == (0,)
    assert not list((ws.path / 'descriptors').rglob('*.json'))
    assert ctx.cache_path.exists()  # No damaged or legacy input is deleted.


def test_alignment_cache_commit_interruption_resumes_exactly(protein, monkeypatch):
    from minitopol.coordinates import fit_positions
    from minitopol.descriptors import alignment_cache
    from minitopol.descriptors.context import AnalysisContext
    ws, tid, _ = protein
    options = dict(coordinate_mode='nonperiodic', chunk_frames=3)
    ctx = AnalysisContext(ws, tid, **options)
    original = alignment_cache.atomic_json
    def interrupt(path, data):
        if path.name == 'state' and data.get('committed') == 6:
            raise RuntimeError('interrupt after cache checksum but before commit')
        return original(path, data)
    monkeypatch.setattr(alignment_cache, 'atomic_json', interrupt)
    with ctx.cache_lock, pytest.raises(RuntimeError, match='before commit'):
        ctx.prepare_aligned()
    monkeypatch.setattr(alignment_cache, 'atomic_json', original)
    resumed = AnalysisContext(ws, tid, **options)
    with resumed.cache_lock:
        actual = np.concatenate([p for _, _, p, _ in resumed.aligned_blocks()])
    raw = np.concatenate([block.positions for block in Registry(ws).blocks(tid)]).astype(float)
    ca = resumed.ca_indices()
    expected = np.array([fit_positions(frame, raw[0], ca) for frame in raw])
    np.testing.assert_array_equal(actual, expected)
    # An intact completed cache is reusable without reading source coordinates.
    reused = AnalysisContext(ws, tid, **options)
    def no_source_reads():
        raise AssertionError('completed cache should be verified and reused')
    monkeypatch.setattr(reused, 'coordinate_blocks', no_source_reads)
    with reused.cache_lock:
        reused.prepare_aligned()
    report = analyze(ws, tid, descriptors=['rmsf'], **options)
    assert report['status'] == 'complete', report
    mean = np.zeros_like(expected[0])
    m2 = np.zeros_like(mean)
    for count, frame in enumerate(expected, 1):
        delta = frame - mean
        mean += delta / count
        m2 += delta * (frame - mean)
    actual_rmsf = [r['value'] for r in sorted(rows(ws, 'rmsf'), key=lambda r: r['point'])]
    np.testing.assert_array_equal(actual_rmsf, np.sqrt((np.maximum(m2, 0) / len(expected)).sum(axis=1)))
