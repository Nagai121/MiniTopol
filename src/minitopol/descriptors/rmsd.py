import numpy as np
import pyarrow as pa


def calculate(ctx, writer):
    from .aligned_rmsd_handoff import consume
    if consume(ctx,writer):return
    # Share this full aligned pass with later RMSF/MSD/correlation statistics.
    # Match statistics() per-frame operation order exactly; publish only at EOF.
    collect = ctx._stats is None and ctx.m * 3 * 8 * 6 <= ctx.budget // 8
    if collect:
        mean = np.zeros((ctx.m, 3)); m2 = np.zeros_like(mean); count = 0
    from .shared_aligned_ca import prepare,observe,finish
    shared=prepare(ctx)
    reference = ctx.read_aligned(0, 1)[0]
    for start, stop, positions, time in ctx.aligned_blocks():
        observe(shared,start,stop,positions)
        if collect:
            for frame in positions:
                count += 1
                delta = frame - mean
                mean += delta / count
                m2 += delta * (frame - mean)
        if writer.has(start):
            continue
        value = np.sqrt(np.sum((positions - reference)**2, axis=(1, 2)) / ctx.m)
        writer.append(start, pa.table({"frame": np.arange(start, stop), "time_ps": time, "value": value}))
    if collect:
        if count != ctx.n:
            raise ValueError("Incomplete shared RMSD statistics pass")
        ctx._stats = mean, np.maximum(m2, 0) / ctx.n

    finish(ctx,shared)
