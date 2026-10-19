import numpy as np
import pyarrow as pa


def calculate(ctx, writer):
    reference = ctx.read_aligned(0, 1)[0]
    for start, stop, positions, time in ctx.aligned_blocks():
        if writer.has(start):
            continue
        value = np.sqrt(np.sum((positions - reference)**2, axis=(1, 2)) / ctx.m)
        writer.append(start, pa.table({"frame": np.arange(start, stop), "time_ps": time, "value": value}))
