"""Native molecule-whole NoJump and large-GPU-budget regression tests."""
import numpy as np
import pytest

from minitopol.whole_nojump import bond_forest, reconstruct_frame
from minitopol.descriptors import gpu_sasa_batch


def _run(raw, box, edges, chunk):
    order, parent, offsets = bond_forest(raw.shape[1], edges)
    previous = np.zeros((len(offsets) - 1, 3), dtype=np.float64)
    frames = []
    first = True
    for left in range(0, len(raw), chunk):
        for positions in raw[left:left + chunk]:
            current = np.array(positions, dtype=np.float64, copy=True)
            reconstruct_frame(current, box, np.linalg.inv(box), order, parent,
                              offsets, previous, first)
            frames.append(current)
            first = False
    return np.stack(frames)


def test_whole_then_nojump_keeps_bonds_and_crosses_periodic_boundary():
    box = np.diag([10., 10., 10.])
    base = np.array([[9.7, 1., 1.], [10.3, 1., 1.], [10.9, 1., 1.]])
    expected = np.stack([base + [2.*t, 0., 0.] for t in range(8)])
    wrapped = expected % 10.
    for chunk in (1, 2, 5, 8):
        result = _run(wrapped, box, np.array([[0, 1], [1, 2]]), chunk)
        np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(wrapped, expected % 10.)


def test_triclinic_bond_image_and_isolated_atom():
    box = np.array([[10., 0., 0.], [3., 10., 0.], [2., 2., 10.]])
    base = np.array([[0.5, 1., 1.], [0.9, 1., 1.], [6., 7., 8.]])
    raw = base.copy()
    raw[1] += box[0]
    result = _run(raw[None], box, np.array([[0, 1]]), 1)
    np.testing.assert_allclose(result[0], base, atol=1e-12)


def test_gpu_planner_accepts_up_to_20480_subject_to_memory(monkeypatch):
    monkeypatch.setattr(gpu_sasa_batch, "_buffers", None)
    assert gpu_sasa_batch.fits_planned_reservation(1, 20480, 1 << 34, 1 << 33)
    assert not gpu_sasa_batch.fits_planned_reservation(1, 20481, 1 << 34, 1 << 33)
    assert not gpu_sasa_batch.fits_planned_reservation(1, 20480, 1 << 20, 1 << 20)


def test_invalid_bond_forest_is_rejected():
    with pytest.raises(ValueError):
        bond_forest(3, [[0, 3]])


def test_coordinate_blocks_use_native_whole_when_bonds_exist():
    import pyarrow as pa
    from minitopol.registry import FrameBlock
    from minitopol.coordinates import coordinate_blocks

    base = np.array([[9.7, 1., 1.], [10.3, 1., 1.], [10.9, 1., 1.]])
    expected = np.stack([base + [2.*t, 0., 0.] for t in range(5)])
    stored = np.asarray(expected % 10., dtype=np.float32)
    box = np.diag([10., 10., 10.])
    table = pa.table({"point_i": [0, 1], "point_j": [1, 2]})

    class RegistryStub:
        def bond_info(self, trajectory):
            return {"edges": 2}

        def bonds(self, trajectory):
            return table

        def manifest(self, trajectory):
            return {"atoms": 3}

        def blocks(self, trajectory, chunk_frames=None):
            for i in range(0, len(stored), chunk_frames or len(stored)):
                j = min(i + (chunk_frames or len(stored)), len(stored))
                yield FrameBlock(i, j, stored[i:j].copy(),
                                 np.arange(i, j, dtype=np.float64),
                                 np.repeat(box[None], j-i, axis=0))

    registry = RegistryStub()
    for chunk in (1, 2, 5):
        result = np.concatenate([b.positions for b in coordinate_blocks(
            registry, "test", mode="nojump", initial_whole=False,
            half_box_steps=True, chunk_frames=chunk)])
        np.testing.assert_allclose(result, expected, atol=1e-5)
    np.testing.assert_allclose(stored, expected % 10., atol=1e-5)
    with pytest.raises(ValueError, match="half-box"):
        list(coordinate_blocks(registry, "test", mode="nojump",
                               half_box_steps=False))
