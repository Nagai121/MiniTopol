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


def test_gpu_preserves_float64_occlusion_boundary(cuda_ready):
    from minitopol.descriptors.kernels import sphere_points
    points = sphere_points(960)
    positions = np.zeros((1, 16, 3), dtype=np.float64)
    positions[0, 1] = points[0] * 3.1 + np.array([3.1 - 1e-8, 0., 0.])
    positions[0, 2:, 0] = np.arange(2, 16) * 20.
    radii = np.full(16, 1.7)
    actual = gpu.evaluate_batch(positions, radii, 128 * 1024**2)
    assert actual is not None
    np.testing.assert_array_equal(actual[0], sasa_atoms(positions[0], radii))


def test_filtered_predicate_matches_cpu_at_domain_boundaries(cuda_ready, tmp_path):
    from minitopol.descriptors.gpu_fused import fused_counts
    from minitopol.descriptors.kernels import sphere_points

    cuda = cuda_ready
    points = sphere_points(960)
    assert np.abs(points).max() <= 1
    cases = []
    # Probe individual occluder boundaries, both signs, large translations,
    # minimum/maximum inflated radii, and binary64 nextafter perturbations.
    for center in (0., 1e-20, -1e-20, 1000., -1000., 999950., -999950.):
        for radius, other_radius in ((1.5, 1.5), (10., 10.), (1.5, 10.), (10., 1.5)):
            for sample in (0, 17, 479, 959):
                target = np.full(3, center)
                occluder = target + points[sample] * radius + np.array([other_radius, 0., 0.])
                for direction in (-np.inf, np.inf):
                    shifted = occluder.copy()
                    shifted[0] = np.nextafter(shifted[0], direction)
                    cases.append((target, shifted, radius, other_radius))
    rng = np.random.default_rng(37)
    for _ in range(256):
        target = rng.uniform(-999950., 999950., 3)
        radius, other_radius = rng.uniform(1.5, 10., 2)
        occluder = target + rng.normal(size=3) * (radius + other_radius)
        cases.append((target, occluder, radius, other_radius))

    for left in range(0, len(cases), 32):
        batch = cases[left:left+32]
        xyz = np.asarray([[a, b] for a, b, _, _ in batch])
        radii = np.asarray([[r, s] for _, _, r, s in batch])
        lengths = np.full((len(batch), 2), 2, dtype=np.int32)
        neighbors = np.zeros((len(batch), 2, 128), dtype=np.int32)
        neighbors[:, :, 1] = 1
        for positions, radii_pair in zip(xyz, radii):
            output = cuda.device_array((1, 2), dtype=np.int64)
            fused_counts[2, 256](
                cuda.to_device(positions[None]), cuda.to_device(radii_pair),
                cuda.to_device(points), cuda.to_device(lengths[:1]),
                cuda.to_device(neighbors[:1]), output)
            actual = output.copy_to_host()[0]
            expected = []
            for atom in (0, 1):
                delta = positions[atom] + points * radii_pair[atom] - positions[1-atom]
                square = (delta[:, 0]*delta[:, 0] + delta[:, 1]*delta[:, 1]) + delta[:, 2]*delta[:, 2]
                expected.append(np.count_nonzero(square >= radii_pair[1-atom]**2))
            np.testing.assert_array_equal(actual, expected)
    ptx = next(iter(fused_counts.inspect_asm().values()))
    assert 'add.rn.f32' in ptx and 'mul.rn.f32' in ptx
    assert 'add.rn.f64' in ptx and 'mul.rn.f64' in ptx
    assert '.ftz' not in ptx
    # The error-envelope construction may fuse operations, which reduces its
    # rounding error. Explicit RN intrinsics keep both predicates unfused.
    assert 'fma.rn.f32' not in ptx
    (tmp_path / 'filtered-sasa.ptx').write_text(ptx)


def test_gpu_buffer_cache_tracks_changed_shape(cuda_ready):
    radii = np.full(32, 1.7)
    xyz = np.zeros((1, 32, 3), dtype=np.float64)
    xyz[0, :, 0] = np.arange(32) * 10.
    assert gpu.evaluate_counts(xyz, radii, 128 * 1024**2) is not None
    assert gpu._buffers['shape'] == (1, 32)
    bigger = np.repeat(xyz, 2, axis=0)
    assert gpu.evaluate_counts(bigger, radii, 128 * 1024**2) is not None
    assert gpu._buffers['shape'] == (2, 32)
    assert gpu.evaluate_counts(xyz, radii, 128 * 1024**2) is not None
    assert gpu._buffers['shape'] == (2, 32)
