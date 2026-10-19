import numpy as np
import pyarrow as pa


def calculate(ctx, writer):
    _, variance = ctx.statistics()
    for start in range(0, ctx.m, 4096):
        if writer.has(start):
            continue
        indices = np.arange(start, min(start + 4096, ctx.m))
        writer.append(start, pa.table({"point": indices, "residue": np.asarray(ctx.atoms["residue_index"])[indices],
                                       "frames": np.full(len(indices), ctx.n), "value": np.sqrt(variance[indices].sum(axis=1))}))
