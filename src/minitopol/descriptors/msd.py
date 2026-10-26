import numpy as np
import pyarrow as pa
import zarr

from ..storage import atomic_json, read_json
from .correlation import atom_correlation


def _slot_encoding():
    try:
        import pcodec
        from zarr.codecs.numcodecs import PCodec
    except ImportError:
        return {}
    return {"serializer": PCodec(level=2, delta_spec="try_consecutive", delta_encoding_order=2),
            "compressors": []}


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
            group.create_array(name, shape=(ctx.n,), chunks=(ctx.fft_block,), dtype="f8", fill_value=0,
                               **_slot_encoding())
    slots = (group['slot0'], group['slot1'])
    # Keep the exact persisted accumulator in RAM only when two full vectors
    # fit a small reserved fraction, well below the one-tile FFT budget.
    # Long/tiled trajectories retain the existing bounded Zarr-only path.
    cached = state['atoms_done'] < ctx.m and ctx.fft_block >= ctx.n and ctx.n * 16 <= ctx.budget // 64
    accumulator = np.asarray(slots[state['slot']][:]) if cached else None
    next_accumulator = np.empty(ctx.n) if cached else None
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
            if cached:
                updated = next_accumulator[start:stop]
                np.add(accumulator[start:stop],value,out=updated)
                slots[next_slot][start:stop] = updated
            else:
                slots[next_slot][start:stop] = slots[current][start:stop] + value
        # Double-buffered commit: an interrupted atom never alters the previous
        # accumulator. Its uncommitted target slot is overwritten on restart.
        state = {"atoms_done": point + 1, "slot": next_slot}
        atomic_json(marker, state)
        if cached:
            accumulator, next_accumulator = next_accumulator, accumulator
    for start in range(0, ctx.n, ctx.chunk):
        if writer.has(start):
            continue
        stop = min(start + ctx.chunk, ctx.n)
        lag = np.arange(start, stop)
        writer.append(start, pa.table({"lag_frames": lag, "lag_ps": lag * step,
                                       "origins": ctx.n - lag, "atoms": np.full(len(lag), ctx.m),
                                       "value": slots[state['slot']][start:stop] / ctx.m}))
