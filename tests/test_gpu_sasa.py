import numpy as np
import pytest

from minitopol.descriptors import gpu_sasa_batch as gpu
from minitopol.descriptors.kernels import sasa_atoms


def test_cpu_selection_never_initializes_cuda(monkeypatch):
    monkeypatch.setenv("MINITOPOL_SASA_DEVICE", "cpu")
    monkeypatch.setattr(gpu, "_evaluate", lambda *a, **k: pytest.fail("CPU worker entered CUDA"))
    assert gpu.evaluate_batch(np.zeros((2, 128, 3)), np.ones(128), 128 * 1024**2) is None


def test_invalid_or_unbudgeted_batches_keep_cpu_path(monkeypatch):
    monkeypatch.delenv("MINITOPOL_SASA_DEVICE", raising=False)
    xyz = np.ones((2, 128, 3));radii = np.full(128, 1.7)
    assert gpu.evaluate_batch(xyz, radii, 1) is None
    assert gpu.evaluate_batch(xyz, radii, 128 * 1024**2, available=1) is None
    assert gpu.evaluate_batch(xyz[:0], radii, 128 * 1024**2) is None
    assert gpu.evaluate_batch(xyz.astype("f4"), radii, 128 * 1024**2) is None
    assert gpu.evaluate_batch(xyz[:, :3], radii[:3], 128 * 1024**2) is None
    with np.errstate(over="raise"):
        assert gpu.evaluate_batch(xyz, radii, 128 * 1024**2) is None


@pytest.fixture
def cuda_ready(monkeypatch):
    monkeypatch.delenv("MINITOPOL_SASA_DEVICE", raising=False)
    from minitopol.acceleration import configure_cuda
    configure_cuda()
    cuda = pytest.importorskip("numba.cuda")
    if not cuda.is_available():
        pytest.skip("CUDA toolkit and device are not available")
    return cuda


def test_gpu_counts_match_independent_cpu_for_every_atom(cuda_ready):
    xyz = np.ascontiguousarray(np.random.default_rng(19).normal(size=(2, 128, 3)) * 8)
    radii = np.resize(np.array([1.7, 1.55, 1.52, 1.8]), 128)
    actual = gpu.evaluate_batch(xyz, radii, 128 * 1024**2)
    assert actual is not None
    for frame in range(2):
        expected = sasa_atoms(xyz[frame], radii)
        assert np.array_equal(actual[frame].view("u8"), expected.view("u8"))


def test_gpu_tangent_duplicate_and_dense_guards(cuda_ready):
    xyz = np.zeros((1, 130, 3));xyz[0, :, 0] = np.arange(130) * 10
    radii = np.full(130, 1.7)
    for separation in (np.nextafter(6.2, 0), 6.2, np.nextafter(6.2, np.inf)):
        xyz[0, 1, 0] = separation
        actual = gpu.evaluate_batch(xyz, radii, 128 * 1024**2)
        assert actual is not None
        assert np.array_equal(actual[0].view("u8"), sasa_atoms(xyz[0], radii).view("u8"))
    xyz[:] = 0
    assert gpu.evaluate_batch(xyz, radii, 128 * 1024**2) is None
    xyz[0, :, 0] = np.arange(130) * .0001
    assert gpu.evaluate_batch(xyz, radii, 128 * 1024**2) is None
