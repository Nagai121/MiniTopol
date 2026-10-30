import numpy as np
import pyarrow as pa


def calculate(ctx, writer):
    mass = ctx.atomic_values("masses")
    from .shared_gyration import cached_blocks
    shared = cached_blocks(ctx, mass)
    if shared is not None:
        for start, stop, times, values in shared:
            if writer.has(start):continue
            writer.append(start, pa.table({"frame": np.arange(start,stop), "time_ps":times,"value":values}))
        return
    total = mass.sum()
    for block in ctx.coordinate_blocks():
        if writer.has(block.start):
            continue
        center = np.einsum("faj,a->fj", block.positions, mass) / total
        delta = block.positions - center[:, None, :]
        value = np.sqrt(np.einsum("faj,faj,a->f", delta, delta, mass) / total)
        writer.append(block.start, pa.table({"frame": np.arange(block.start, block.stop),
                                             "time_ps": block.time, "value": value}))
