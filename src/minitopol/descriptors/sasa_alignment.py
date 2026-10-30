"""Fresh SASA input also feeds the existing durable alignment contract.

No transformed-coordinate cache is retained in RAM. Alignment errors are
recorded and deferred: independent SASA still sees its original full stream.
Existing/partial caches use the original preparation and verification path.
"""
import numpy as np
import zarr
from ..coordinates import fit_positions
from ..fit_batch import fit_positions_batch
from ..storage import atomic_json,read_json
from .alignment_cache import block_checksum

def _open(ctx):
    ca = ctx.ca_indices()
    if len(ca) < 3:
        raise ValueError("At least three non-collinear C-alpha atoms are required for aligned descriptors")
    ctx.cache_path.mkdir(parents=True, exist_ok=True)
    # Reuse the existing exact f64 PCodec policy, already used for MSD slots.
    from .msd import _slot_encoding
    encoding = _slot_encoding()
    codec_tag = encoding['serializer'].to_dict() if 'serializer' in encoding else {'name':'zarr-default'}
    marker = ctx.cache_path / "state"
    state = read_json(marker) if marker.exists() else {"schema": 3, "encoding": codec_tag, "committed": 0, "layout": ctx.layout}
    if state["layout"] != ctx.layout:
        raise ValueError("Coordinate cache layout mismatch")
    if state.get("schema") != 3 or state.get("encoding") != codec_tag:
        raise IOError("Aligned cache codec contract changed; resume with its original source")
    committed = state["committed"]
    if type(committed) is not int or not 0 <= committed <= ctx.n or (committed != ctx.n and committed % ctx.chunk):
        raise IOError("Coordinate cache commit boundary mismatch")
    cache = zarr.open_group(str(ctx.cache_path / "aligned.zarr"), mode="a", zarr_format=3)
    for name, shape, chunks in (("positions", (ctx.n, ctx.m, 3), (ctx.chunk, min(32, ctx.m), 3)),
                                ("time", (ctx.n,), (ctx.chunk,))):
        if name not in cache:
            if committed:
                raise IOError("Committed coordinate cache data is missing")
            cache.create_array(name, shape=shape, chunks=chunks, dtype="f8", **encoding)
        elif cache[name].shape != shape:
            raise IOError("Coordinate cache shape mismatch")
        elif cache[name].chunks != chunks or cache[name].dtype != np.dtype("f8"):
            raise IOError("Coordinate cache array schema mismatch")
        if encoding and [codec.to_dict() for codec in cache[name].metadata.codecs] != [codec_tag]:
            raise IOError("Aligned cache codec configuration mismatch")
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
    return ca,cache,checks,state,marker

def coordinate_blocks(ctx):
    stream=ctx.coordinate_blocks()
    audit=dict(attempted=False,used=False,reason='notrequested_or_cacheexists',frames=0)
    ctx._sasa_alignment_audit=audit
    if not getattr(ctx,'_sasa_alignment_requested',False) or ctx.cache is not None or (ctx.cache_path/'state').exists():
        yield from stream
        return
    raw=getattr(ctx,'_shared_raw_backbone',None)
    held=raw[-1] if isinstance(raw,tuple) and isinstance(raw[-1],int) else 0
    # Alignment runs before GPU evaluation; its output is released before yield.
    # Half remains available for SASA output/held bookkeeping and native work.
    working=ctx.chunk*ctx.m*3*8*3+ctx.m*3*8*64+ctx.n*16+ctx.m*8+held
    if working>ctx.budget//2:
        audit['reason']='combinedbudget';yield from stream;return
    audit['attempted']=True
    failed=False;reference=None;ctx._sasa_alignment_held=0
    try:
        try:ca,cache,checks,state,marker=_open(ctx)
        except Exception as error:
            failed=True;audit.update(reason='alignmentdeferred',error=f'{type(error).__name__}: {error}')
        for block in stream:
            aligned=None
            if not failed:
                try:
                    if reference is None:
                        reference=block.positions[0].copy()
                        ctx._sasa_alignment_held=reference.nbytes
                        fit_positions(reference,reference,ca)
                    if block.start!=state['committed']:raise IOError('Coordinate cache commit boundary mismatch')
                    aligned=np.empty_like(block.positions)
                    for index in range(0,len(block.positions),16):
                        aligned[index:index+16]=fit_positions_batch(block.positions[index:index+16],reference,ca)
                    cache['positions'][block.start:block.stop]=aligned
                    cache['time'][block.start:block.stop]=block.time
                    atomic_json(checks/str(block.start),dict(stop=block.stop,sha256=block_checksum(aligned,block.time)))
                    state['committed']=block.stop;atomic_json(marker,state)
                    audit['frames']=block.stop
                except Exception as error:
                    failed=True;audit.update(reason='alignmentdeferred',error=f'{type(error).__name__}: {error}')
                finally:aligned=None
            yield block
        if not failed and state['committed']==ctx.n:
            ctx.cache=cache;audit.update(used=True,reason='SASAcoordinateproducer',rawpass_avoided_logical_bytes=ctx.n*ctx.m*12)
    finally:ctx._sasa_alignment_held=0
