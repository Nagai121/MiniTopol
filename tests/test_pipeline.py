import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pyarrow as pa
import pytest

from minitopol import Registry, Workspace, convert
from minitopol.descriptors import DescriptorDefinition, ResultStore


def _initialize(path):
    return Workspace(path).list_trajectories()


def _publish_once(args):
    path, tid = args
    definition = DescriptorDefinition("concurrent", "1", "A", "test-only", "dimensionless")
    return ResultStore(Workspace(path)).publish(tid, definition, 0, 1,
                                               pa.table({"frame": [0], "value": [1.]}))


def imported(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "workspace")
    gro, xtc, _ = trajectory
    tid = convert(workspace, gro, xtc, solvent_resnames=["SOL"], chunk_frames=4)
    return workspace, tid


def test_empty_workspace_and_concurrent_initialization(tmp_path):
    path = str(tmp_path / "ws")
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert list(pool.map(_initialize, [path] * 8)) == [[]] * 8
    assert Registry(Workspace(path)).representations() == ["A"]
    assert not list((tmp_path / "ws" / "descriptors").iterdir())


def test_real_gro_xtc_roundtrip_keeps_drift_time_and_box(tmp_path, trajectory):
    workspace, tid = imported(tmp_path, trajectory)
    registry = Registry(workspace)
    blocks = list(registry.blocks(tid, chunk_frames=3))
    actual = np.concatenate([block.positions for block in blocks])
    expected = np.stack([trajectory[2][:2] + [i * 5, 0, 0] for i in range(9)])
    np.testing.assert_allclose(actual, expected, atol=0.002)
    np.testing.assert_allclose(np.concatenate([block.time for block in blocks]), np.arange(9) * 2.5)
    assert blocks[0].box.shape == (3, 3, 3)
    assert abs(blocks[0].box[0, 1, 0]) > 1  # nonorthogonal box retained
    assert registry.atoms(tid)["source_atom"].to_pylist() == [0, 1]
    assert workspace.list_trajectories() == [(tid, 9, 2)]
    assert json.loads((workspace.trajectory_path(tid) / "A.zarr" / "zarr.json").read_text())["zarr_format"] == 3


def test_converter_resume_after_interrupted_chunk(tmp_path, trajectory, monkeypatch):
    from minitopol import converter
    workspace = Workspace(tmp_path / "ws")
    original = converter.atomic_json

    def interrupt(path, data):
        if data.get("committed_frames") == 8:
            raise RuntimeError("simulated interruption before commit marker")
        original(path, data)

    monkeypatch.setattr(converter, "atomic_json", interrupt)
    with pytest.raises(RuntimeError):
        convert(workspace, *trajectory[:2], solvent_resnames=["SOL"], trajectory_id="test", chunk_frames=4)
    assert workspace.list_trajectories() == []
    with pytest.raises(ValueError, match="not completed"):
        Registry(workspace).manifest("test")
    monkeypatch.setattr(converter, "atomic_json", original)
    convert(workspace, *trajectory[:2], solvent_resnames=["SOL"], trajectory_id="test", chunk_frames=4)
    actual = np.concatenate([b.positions for b in Registry(workspace).blocks("test")])
    np.testing.assert_allclose(actual[-1], trajectory[2][:2] + [40, 0, 0], atol=0.002)
    with pytest.raises(ValueError, match="different sources/options"):
        convert(workspace, *trajectory[:2], solvent_resnames=[], trajectory_id="test", chunk_frames=4)


def test_registration_does_not_generate_coordinates(tmp_path, trajectory):
    workspace, tid = imported(tmp_path, trajectory)
    registry = Registry(workspace)
    groups = [{"name": "bead", "atoms": [0, 1], "weights": [0.25, 0.75]}]
    descriptor = DescriptorDefinition("example", "1", "C1", "test-only", "dimensionless")
    kwargs = dict(trajectory=tid, version="1", groups=groups,
                  descriptor_definitions={"example": descriptor.as_dict()})
    registry.register("C1", **kwargs)
    registry.register("C1", **kwargs)
    assert registry.representations() == ["A", "C1"]
    assert not list(workspace.path.rglob("C1.zarr"))
    a = next(registry.blocks(tid))
    c = next(registry.blocks(tid, "C1"))
    np.testing.assert_allclose(c.positions[:, 0], a.positions[:, 0] * 0.25 + a.positions[:, 1] * 0.75)
    with pytest.raises(ValueError, match="immutable"):
        registry.register("C1", **dict(kwargs, version="2"))
    with pytest.raises(ValueError, match="sum to one"):
        registry.register("C2", **dict(kwargs, groups=[dict(groups[0], weights=[1, 1])]))
    wrong = DescriptorDefinition("example", "2", "C1", "different test-only method", "dimensionless")
    with pytest.raises(ValueError, match="registered for this C"):
        ResultStore(workspace).publish(tid, wrong, 0, 1, pa.table({"value": [1.]}))
    other = convert(workspace, *trajectory[:2], solvent_resnames=[], chunk_frames=4)
    with pytest.raises(ValueError, match="different A topology"):
        list(registry.blocks(other, "C1"))


def test_publish_query_recover_and_no_duplicate(tmp_path, trajectory, monkeypatch):
    workspace, tid = imported(tmp_path, trajectory)
    definition = DescriptorDefinition("test_value", "1", "A", "test-only known values", "dimensionless")
    table = pa.table({"frame": [0, 1, 2], "value": [2., 4., 6.]})
    store = ResultStore(workspace)
    real_register = store._register

    def interrupt(*args):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(store, "_register", interrupt)
    with pytest.raises(RuntimeError):
        store.publish(tid, definition, 0, 3, table)
    with pytest.raises(LookupError):
        workspace.query_results()
    monkeypatch.setattr(store, "_register", real_register)
    assert store.recover() == 1
    store.publish(tid, definition, 0, 3, table)
    result = workspace.query_results(descriptor="test_value", trajectory=tid, columns=["frame", "value"], start=1, stop=3)
    assert result.to_pydict() == {"frame": [1, 2], "value": [4., 6.]}
    with workspace.catalog() as con:
        assert con.execute("SELECT count(*) FROM results").fetchone() == (1,)
    assert len(list((workspace.path / "descriptors").rglob("*.parquet"))) == 1
    with pytest.raises(ValueError, match="different values"):
        store.publish(tid, definition, 0, 3, pa.table({"frame": [0], "value": [999.]}))
    with pytest.raises(ValueError, match="Overlapping"):
        store.publish(tid, definition, 2, 4, pa.table({"frame": [2, 3], "value": [6., 8.]}))


@pytest.mark.parametrize("value", ["../outside", "/tmp/outside", "a/b", "a'b", ""])
def test_invalid_ids_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        Workspace(tmp_path).trajectory_path(value)


def test_invalid_frame_ranges(tmp_path, trajectory):
    workspace, tid = imported(tmp_path, trajectory)
    for params in ({"start": -1}, {"stop": 10}, {"chunk_frames": 0}):
        with pytest.raises(ValueError):
            list(Registry(workspace).blocks(tid, **params))


def test_calculator_resume_and_definition_snapshot(tmp_path, trajectory):
    from minitopol.descriptors.calculators import calculate_frames
    workspace, tid = imported(tmp_path, trajectory)
    parameters = {"purpose": "test"}
    definition = DescriptorDefinition("frame_index", "1", "A", "test-only frame indices", "dimensionless", parameters)
    identity = definition.identity
    parameters["purpose"] = "mutated"
    assert definition.identity == identity
    called = []

    def calculator(block):
        called.append(block.start)
        return pa.table({"frame": list(range(block.start, block.stop)),
                         "value": [float(x) for x in range(block.start, block.stop)]})

    assert calculate_frames(workspace, tid, definition, calculator) == 3
    assert called == [0, 4, 8]
    called.clear()
    assert calculate_frames(workspace, tid, definition, calculator) == 3
    assert called == []
    assert workspace.query_results(descriptor="frame_index").num_rows == 9


def test_completed_results_with_different_chunking_cannot_overlap(tmp_path, trajectory, monkeypatch):
    workspace, tid = imported(tmp_path, trajectory)
    definition = DescriptorDefinition("test", "1", "A", "test-only", "dimensionless")
    store = ResultStore(workspace)
    register = store._register
    monkeypatch.setattr(store, "_register", lambda *args: None)
    store.publish(tid, definition, 0, 4, pa.table({"value": [1.]}))
    store.publish(tid, definition, 0, 8, pa.table({"value": [2.]}))
    monkeypatch.setattr(store, "_register", register)
    with pytest.raises(ValueError, match="Overlapping"):
        store.recover()
    with workspace.catalog() as con:
        assert con.execute("SELECT count(*) FROM results").fetchone() == (1,)


def test_concurrent_result_publication(tmp_path, trajectory):
    workspace, tid = imported(tmp_path, trajectory)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        ids = list(pool.map(_publish_once, [(str(workspace.path), tid)] * 4))
    assert len(set(ids)) == 1
    assert workspace.query_results(descriptor="concurrent").num_rows == 1
