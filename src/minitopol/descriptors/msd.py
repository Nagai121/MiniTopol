import numpy as np
import pyarrow as pa
import zarr

from ..storage import atomic_json, read_json
from .correlation import atom_correlation


def calculate(ctx, writer):
    step = ctx.timestep()
    mean, _ = ctx.statistics()
    directory = ctx.cache_path / "msd"
    directory.mkdir(exist_ok=True)
    marker = directory / "state"
    state = read_json(marker) if marker.exists() else {"atoms_done": 0, "slot": 0}
    group = zarr.open_group(str(directory / "sum.zarr"), mode="a", zarr_format=3)
    for name in ("slot0", "slot1"):
        if name not in group:
            if state["atoms_done"]:
                raise IOError("Committed MSD accumulator is missing")
            group.create_array(name, shape=(ctx.n,), chunks=(ctx.fft_block,), dtype="f8", fill_value=0)
    for point in range(state["atoms_done"], ctx.m):
        dots, prefix = atom_correlation(ctx, point, mean[point], directory / "scratch.zarr")
        current, next_slot = state["slot"], 1 - state["slot"]
        for start in range(0, ctx.n, ctx.fft_block):
            stop = min(start + ctx.fft_block, ctx.n)
            lag = np.arange(start, stop)
            squares = prefix.oindex[ctx.n-lag] + prefix[ctx.n] - prefix[start:stop]
            numerator = squares - 2 * dots[start:stop]
            tolerance = 256 * np.finfo(float).eps * max(float(prefix[ctx.n]), 1)
            if np.any(numerator < -tolerance):
                raise ArithmeticError("Negative MSD exceeds floating-point cancellation tolerance")
            value = np.maximum(numerator, 0) / (ctx.n - lag)
            value[lag == 0] = 0
            group[f"slot{next_slot}"][start:stop] = group[f"slot{current}"][start:stop] + value
        # Double-buffered commit: an interrupted atom never alters the previous
        # accumulator. Its uncommitted target slot is overwritten on restart.
        state = {"atoms_done": point + 1, "slot": next_slot}
        atomic_json(marker, state)
    for start in range(0, ctx.n, ctx.chunk):
        if writer.has(start):
            continue
        stop = min(start + ctx.chunk, ctx.n)
        lag = np.arange(start, stop)
        writer.append(start, pa.table({"lag_frames": lag, "lag_ps": lag * step,
                                       "origins": ctx.n - lag, "atoms": np.full(len(lag), ctx.m),
                                       "value": group[f"slot{state['slot']}"][start:stop] / ctx.m}))
