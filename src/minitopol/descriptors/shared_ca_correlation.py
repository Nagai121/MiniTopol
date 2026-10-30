"""Ephemeral, budgeted exact CA dot sums shared by MSD and time correlation."""
import numpy as np

def remember(ctx, point, mean, dots):
    if ctx.fft_block < ctx.n or not callable(getattr(ctx, 'ca_indices', None)):
        return
    cache = getattr(ctx, '_shared_ca_dots', None)
    if cache is None:
        points = frozenset(map(int, ctx.ca_indices()))
        needed = len(points) * (ctx.n * 8 + 24 + 256)
        cache = (points if needed <= ctx.budget // 8 else frozenset(), {})
        ctx._shared_ca_dots = cache
    if int(point) not in cache[0]:
        return
    if not isinstance(dots, np.ndarray) or dots.shape != (ctx.n,) or dots.dtype != np.dtype('f8'):
        return
    cache[1][int(point)] = (np.asarray(mean, dtype='f8').copy(), dots.copy())

def take(ctx, point, mean):
    cache = getattr(ctx, '_shared_ca_dots', None)
    if cache is None:
        return None
    item = cache[1].pop(int(point), None)
    if item is None:
        return None
    saved_mean, dots = item
    current_mean = np.asarray(mean, dtype='f8')
    if current_mean.shape != saved_mean.shape or not np.array_equal(current_mean.view('u8'), saved_mean.view('u8')):
        return None
    if dots.shape != (ctx.n,):
        return None
    return dots
