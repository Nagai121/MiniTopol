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


def test_gpu_planner_accepts_up_to_20480_subject_to_memory():
    gpu_sasa_batch._buffers = None
    assert gpu_sasa_batch.fits_planned_reservation(1, 20480, 1 << 34, 1 << 33)
    assert not gpu_sasa_batch.fits_planned_reservation(1, 20481, 1 << 34, 1 << 33)
    assert not gpu_sasa_batch.fits_planned_reservation(1, 20480, 1 << 20, 1 << 20)


def test_invalid_bond_forest_is_rejected():
    with pytest.raises(ValueError):
        bond_forest(3, [[0, 3]])
