import numpy as np
import pyarrow as pa

from .correlation import atom_correlation


def calculate(ctx, writer):
    step = ctx.timestep()
    mean, variance = ctx.statistics()
    for point in ctx.ca_indices():
        if writer.has(f"{point}-done"):
            continue
        residual = variance[point].sum()
        dots = None
        if residual > 0:
            dots, _ = atom_correlation(ctx, point, mean[point], ctx.cache_path / "time-correlation-scratch.zarr")
        for start in range(0, ctx.n, ctx.chunk):
            stop = min(start + ctx.chunk, ctx.n)
            lag = np.arange(start, stop)
            value = dots[start:stop] / ((ctx.n - lag) * residual) if residual > 0 else np.zeros(len(lag))
            if residual > 0:
                value[lag == 0] = 1
            table = pa.table({"point": np.full(len(lag), point),
                              "residue": np.full(len(lag), ctx.atoms["residue_index"][point]),
                              "lag_frames": lag, "lag_ps": lag * step, "origins": ctx.n - lag,
                              "value": pa.array(value, mask=np.full(len(lag), residual == 0)),
                              "reason": pa.array(["zero_variance" if residual == 0 else None] * len(lag), type=pa.string())})
            key = f"{point}-done" if stop == ctx.n else f"{point}-{start}"
            writer.append(key, table)
