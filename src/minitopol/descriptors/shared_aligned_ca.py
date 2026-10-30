"""One short-lived exact CA input shared by RMSD and immediately-following DCC."""
import numpy as np

def prepare(ctx):
    if not getattr(ctx,'_share_aligned_ca',False):return None
    ca=np.asarray(ctx.ca_indices(),dtype=np.int64)
    needed=ctx.n*len(ca)*3*8
    working=ctx.chunk*ctx.m*3*8*3+ctx.m*3*8*6
    if needed>ctx.budget//2 or needed+working>ctx.budget:return None
    return dict(ca=ca.copy(),values=np.empty((ctx.n,len(ca),3),dtype='f8'),next=0)
def observe(state,start,stop,positions):
    if state is None:return
    if start!=state['next'] or stop-start!=len(positions):raise IOError('Shared aligned CA continuity')
    state['values'][start:stop]=positions[:,state['ca']];state['next']=stop
def finish(ctx,state):
    if state is None:return
    if state['next']!=ctx.n:raise IOError('Shared aligned CA incomplete')
    for a in (state['ca'],state['values']):a.flags.writeable=False
    ctx._shared_aligned_ca=(state['ca'],state['values'])
def read(ctx,start,stop,ca):
    saved=getattr(ctx,'_shared_aligned_ca',None)
    if saved is None:return ctx.read_aligned(start,stop,ca)
    if not np.array_equal(np.asarray(ca,dtype='i8'),saved[0]) or saved[1].shape!=(ctx.n,len(ca),3):raise IOError('Shared aligned CA selection/shape changed')
    return saved[1][start:stop]
