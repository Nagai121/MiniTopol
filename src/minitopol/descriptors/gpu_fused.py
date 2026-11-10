"""Conservative float32 filter with CPU-order float64 boundary predicates."""
from numba import cuda, int32, float32
from numba.cuda import libdevice

@cuda.jit(fastmath=False)
def fused_counts(xyz, radii, points, lengths, neighbors, counts):
    q = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n = xyz.shape[1]
    frame, atom = q // n, q % n
    partial = cuda.shared.array(256, int32)
    relative = cuda.shared.array((128, 3), float32)
    squared_radii = cuda.shared.array(128, float32)
    bounds = cuda.shared.array(128, float32)
    ids = cuda.shared.array(128, int32)
    length = lengths[frame, atom]
    if tid < length:
        other = neighbors[frame, atom, tid]
        ids[tid] = other
        scale = radii[atom] + radii[other]
        absolute_scale = 10.0
        for axis in range(3):
            delta = libdevice.dadd_rn(xyz[frame, other, axis], -xyz[frame, atom, axis])
            relative[tid, axis] = float32(delta)
            scale += abs(delta)
            if abs(xyz[frame, atom, axis]) > absolute_scale:
                absolute_scale = abs(xyz[frame, atom, axis])
            if abs(xyz[frame, other, axis]) > absolute_scale:
                absolute_scale = abs(xyz[frame, other, axis])
        squared_radii[tid] = float32(libdevice.dmul_rn(radii[other], radii[other]))
        # Conservative absolute squared-distance error envelope.
        # 64 eps32 covers casts, products, differences and three-term reduction.
        # 128 eps64 covers the CPU absolute-coordinate construction as well.
        # Doubling leaves headroom for rounding the envelope to float32.
        bounds[tid] = float32(2.0 * (64.0 * 1.1920928955078125e-7 * scale * scale
                                  + 128.0 * 2.220446049250313e-16 * (absolute_scale + 10.0)**2))
    cuda.syncthreads()
    radius = float32(radii[atom])
    own = 0
    for point in range(tid, 960, 256):
        px = libdevice.fmul_rn(float32(points[point, 0]), radius)
        py = libdevice.fmul_rn(float32(points[point, 1]), radius)
        pz = libdevice.fmul_rn(float32(points[point, 2]), radius)
        accessible = 1
        for j in range(length):
            if ids[j] == atom:
                continue
            x = libdevice.fadd_rn(px, -relative[j, 0])
            y = libdevice.fadd_rn(py, -relative[j, 1])
            z = libdevice.fadd_rn(pz, -relative[j, 2])
            square = libdevice.fadd_rn(libdevice.fadd_rn(libdevice.fmul_rn(x, x), libdevice.fmul_rn(y, y)), libdevice.fmul_rn(z, z))
            margin = libdevice.fadd_rn(square, -squared_radii[j])
            if abs(margin) <= bounds[j]:
                other = ids[j]
                dx = libdevice.dadd_rn(libdevice.dadd_rn(xyz[frame, atom, 0], libdevice.dmul_rn(points[point, 0], radii[atom])), -xyz[frame, other, 0])
                dy = libdevice.dadd_rn(libdevice.dadd_rn(xyz[frame, atom, 1], libdevice.dmul_rn(points[point, 1], radii[atom])), -xyz[frame, other, 1])
                dz = libdevice.dadd_rn(libdevice.dadd_rn(xyz[frame, atom, 2], libdevice.dmul_rn(points[point, 2], radii[atom])), -xyz[frame, other, 2])
                exact_square = libdevice.dadd_rn(libdevice.dadd_rn(libdevice.dmul_rn(dx, dx), libdevice.dmul_rn(dy, dy)), libdevice.dmul_rn(dz, dz))
                blocked = exact_square < libdevice.dmul_rn(radii[other], radii[other])
            else:
                blocked = margin < 0.0
            if blocked:
                accessible = 0
                break
        own += accessible
    partial[tid] = own
    cuda.syncthreads()
    stride = 128
    while stride:
        if tid < stride:
            partial[tid] += partial[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        counts[frame, atom] = partial[0]
