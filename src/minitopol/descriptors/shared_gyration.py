"""Optional scalar-only gyration reuse from SASA's exact coordinate pass."""
import numpy as np

def prepare(ctx):
    held = ctx.n * 16 + ctx.m * 8
    if held > ctx.budget // 128 or ctx.m * 3 * 8 * (ctx.chunk + 6) > ctx.budget // 4:
        return None
    try:
        mass = ctx.atomic_values('masses')
    except ValueError:
        return None
    return dict(mass=mass, total=mass.sum(), held=held, done=0,
                values=np.empty(ctx.n), times=np.empty(ctx.n))

def observe(shared, block):
    if shared is None:
        return
    if block.start != shared['done']:
        raise ValueError('Noncontiguous shared geometry pass')
    mass = shared['mass']
    center = np.einsum('faj,a->fj', block.positions, mass) / shared['total']
    delta = block.positions - center[:, None, :]
    value = np.sqrt(np.einsum('faj,faj,a->f', delta, delta, mass) / shared['total'])
    shared['values'][block.start:block.stop] = value
    shared['times'][block.start:block.stop] = block.time
    shared['done'] = block.stop

def finish(ctx, shared):
    if shared is None:
        return
    if shared['done'] != ctx.n:
        raise ValueError('Incomplete shared geometry pass')
    ctx._shared_gyration = shared

def cached_blocks(ctx, mass):
    shared = getattr(ctx, '_shared_gyration', None)
    if shared is None or shared['done'] != ctx.n:
        return None
    if mass.shape != shared['mass'].shape or not np.array_equal(mass.view('u8'), shared['mass'].view('u8')):
        return None
    def blocks():
        for start in range(0, ctx.n, ctx.chunk):
            stop = min(start + ctx.chunk, ctx.n)
            yield start, stop, shared['times'][start:stop], shared['values'][start:stop]
    return blocks()
