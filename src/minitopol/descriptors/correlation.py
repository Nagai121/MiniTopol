"""Exact all-origin, all-lag correlations with bounded FFT time tiles."""

import numpy as np
from scipy.signal import fftconvolve
import zarr


class _MemoryArray(np.ndarray):
    @property
    def oindex(self):
        return self


def atom_correlation(ctx, point, mean, directory):
    """Return raw lag dot sums and squared-displacement prefix sums.

    Different time tiles are cross-correlated as well as each tile with itself.
    One-tile work uses budgeted RAM scratch; longer work uses on-disk tiles.
    Only two tiles and their FFT workspaces are in RAM. Neither long-lag pairs
    nor pairs across tile boundaries are dropped.
    """
    if ctx.fft_block >= ctx.n:
        # The original layout already budgets 512 bytes per time point.
        # Preserve its arithmetic, retaining this one-atom scratch in RAM.
        points = max(1, min(32, ctx.budget // (ctx.n * 3 * 8 * 8)))
        start = point // points * points
        stop = min(start + points, ctx.m)
        saved = getattr(ctx, '_correlation_block', None)
        if saved is None or saved[0:2] != (start, stop):
            saved = (start, stop, ctx.read_aligned(0, ctx.n, np.arange(start,stop)))
            ctx._correlation_block = saved
        a = saved[2][:, point-start] - mean
        prefix = np.empty(ctx.n + 1).view(_MemoryArray)
        prefix[0] = 0
        prefix[1:] = np.cumsum(np.einsum("ij,ij->i", a, a))
        convolution = fftconvolve(a, a[::-1], mode="full", axes=0).sum(axis=1)
        return (convolution[ctx.n-1:] + 0).view(_MemoryArray), prefix
    group = zarr.open_group(str(directory), mode="w", zarr_format=3)
    width = ctx.fft_block
    dots = group.create_array("dots", shape=(ctx.n,), chunks=(width,), dtype="f8", fill_value=0)
    prefix = group.create_array("squares", shape=(ctx.n+1,), chunks=(width,), dtype="f8", fill_value=0)
    cumulative = 0.0
    for left in range(0, ctx.n, width):
        a = ctx.read_aligned(left, min(left + width, ctx.n), [point])[:, 0] - mean
        squares = np.cumsum(np.einsum("ij,ij->i", a, a)) + cumulative
        prefix[left+1:left+len(a)+1] = squares
        cumulative = squares[-1]
        for right in range(left, ctx.n, width):
            b = a if right == left else ctx.read_aligned(right, min(right + width, ctx.n), [point])[:, 0] - mean
            convolution = fftconvolve(b, a[::-1], mode="full", axes=0).sum(axis=1)
            if right == left:
                values, start = convolution[len(a)-1:], 0
            else:
                values, start = convolution, right - left - (len(a) - 1)
            end = start + len(values)
            dots[start:end] = dots[start:end] + values
    return dots, prefix
