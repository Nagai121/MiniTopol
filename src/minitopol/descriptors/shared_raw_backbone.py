"""Ephemeral, bounded exact raw-coordinate input shared by two precision paths."""
import itertools
from types import SimpleNamespace
import numpy as np

def source(ctx,quads):
    if not getattr(ctx,'_share_raw_backbone',False):
        return quads,ctx.registry.blocks(ctx.trajectory,chunk_frames=ctx.chunk),0
    cached=getattr(ctx,'_shared_raw_backbone',None)
    points=np.asarray(sorted({int(i) for rows in quads.values() for q in rows if q is not None for i in q}),dtype=np.int64)
    if cached is None:
        stream=iter(ctx.registry.blocks(ctx.trajectory,chunk_frames=ctx.chunk))
        first=next(stream,None)
        if first is None:return quads,iter(()),0
        size=ctx.n*(len(points)*3*first.positions.dtype.itemsize+int(np.prod(first.box.shape[1:]))*first.box.dtype.itemsize+first.time.dtype.itemsize)
        # Retained raw input occupies at most half of this job's working budget.
        # Its construction also reserves the actual decoded input block + gather.
        transient=first.positions.nbytes+first.box.nbytes+first.time.nbytes+len(first.positions)*len(points)*3*first.positions.dtype.itemsize
        if size>ctx.budget//2 or size+transient>ctx.budget:
            return quads,itertools.chain([first],stream),0
        positions=np.empty((ctx.n,len(points),3),dtype=first.positions.dtype)
        boxes=np.empty((ctx.n,*first.box.shape[1:]),dtype=first.box.dtype)
        times=np.empty(ctx.n,dtype=first.time.dtype)
        end=0
        for block in itertools.chain([first],stream):
            if block.start!=end or block.stop>ctx.n:raise IOError('Shared backbone frame continuity')
            transient=block.positions.nbytes+block.box.nbytes+block.time.nbytes+len(block.positions)*len(points)*3*block.positions.dtype.itemsize
            if size+transient>ctx.budget:raise MemoryError('Shared backbone construction budget')
            positions[block.start:block.stop]=block.positions[:,points]
            boxes[block.start:block.stop]=block.box;times[block.start:block.stop]=block.time;end=block.stop
        if end!=ctx.n:raise IOError('Shared backbone incomplete input')
        for a in (points,positions,boxes,times):a.flags.writeable=False
        cached=(points,positions,boxes,times,size);ctx._shared_raw_backbone=cached
    ids,positions,boxes,times,size=cached
    if not np.array_equal(points,ids):raise IOError('Shared backbone topology mismatch')
    index={int(p):i for i,p in enumerate(ids)}
    mapped={name:[None if q is None else [index[int(i)] for i in q] for q in rows] for name,rows in quads.items()}
    def blocks():
        for start in range(0,ctx.n,ctx.chunk):
            stop=min(ctx.n,start+ctx.chunk)
            yield SimpleNamespace(start=start,stop=stop,positions=positions[start:stop],box=boxes[start:stop],time=times[start:stop])
    return mapped,blocks(),size
