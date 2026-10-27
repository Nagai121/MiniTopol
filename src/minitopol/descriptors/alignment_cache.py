"""Aligned coordinates and bounded, per-block integrity checks for restart."""

import hashlib

import numpy as np
import zarr

from ..coordinates import fit_positions
from ..fit_batch import fit_positions_batch
from ..storage import atomic_json, read_json


def block_checksum(positions, time):
    checksum = hashlib.sha256()
    for array in (positions, time):
        checksum.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return checksum.hexdigest()


def prepare_aligned_cache(ctx):
    ca = ctx.ca_indices()
    if len(ca) < 3:
        raise ValueError("At least three non-collinear C-alpha atoms are required for aligned descriptors")
    ctx.cache_path.mkdir(parents=True, exist_ok=True)
    marker = ctx.cache_path / "state"
    state = read_json(marker) if marker.exists() else {"schema": 2, "committed": 0, "layout": ctx.layout}
    if state["layout"] != ctx.layout:
        raise ValueError("Coordinate cache layout mismatch")
    if state.get("schema") != 2:
        raise IOError("Legacy coordinate cache has no integrity records; cannot safely resume this cache")
    committed = state["committed"]
    if type(committed) is not int or not 0 <= committed <= ctx.n or (committed != ctx.n and committed % ctx.chunk):
        raise IOError("Coordinate cache commit boundary mismatch")
    cache = zarr.open_group(str(ctx.cache_path / "aligned.zarr"), mode="a", zarr_format=3)
    for name, shape, chunks in (("positions", (ctx.n, ctx.m, 3), (ctx.chunk, min(32, ctx.m), 3)),
                                ("time", (ctx.n,), (ctx.chunk,))):
        if name not in cache:
            if committed:
                raise IOError("Committed coordinate cache data is missing")
            cache.create_array(name, shape=shape, chunks=chunks, dtype="f8")
        elif cache[name].shape != shape:
            raise IOError("Coordinate cache shape mismatch")
        elif cache[name].chunks != chunks or cache[name].dtype != np.dtype("f8"):
            raise IOError("Coordinate cache array schema mismatch")
    checks = ctx.cache_path / "aligned-checksums"
    # Zarr returns fill values for absent chunks. Check decoded data, not only
    # array shape or existence. Each check is bounded by one processing block;
    # records stay on disk rather than accumulating in the progress marker.
    for start in range(0, committed, ctx.chunk):
        stop = min(start + ctx.chunk, ctx.n)
        record = checks / str(start)
        if not record.is_file():
            raise IOError("Committed coordinate cache integrity record is missing")
        expected = read_json(record)
        actual = block_checksum(cache["positions"][start:stop], cache["time"][start:stop])
        if expected != {"stop": stop, "sha256": actual}:
            raise IOError("Committed coordinate cache checksum mismatch")
    if committed < ctx.n:
        reference = None
        for block in ctx.coordinate_blocks():
            if reference is None:
                reference = block.positions[0].copy()
                fit_positions(reference, reference, ca)
            if block.stop <= state["committed"]:
                continue
            if block.start != state["committed"]:
                raise IOError("Coordinate cache commit boundary mismatch")
            aligned = np.empty_like(block.positions)
            for index in range(0,len(block.positions),16):
                aligned[index:index+16] = fit_positions_batch(block.positions[index:index+16],reference,ca)
            cache["positions"][block.start:block.stop] = aligned
            cache["time"][block.start:block.stop] = block.time
            atomic_json(checks / str(block.start), {"stop": block.stop, "sha256": block_checksum(aligned, block.time)})
            state["committed"] = block.stop
            atomic_json(marker, state)
    return cache
