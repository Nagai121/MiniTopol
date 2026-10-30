"""Fresh aligned blocks feed existing RMSD/stats/CA operations before reread."""
import numpy as np
import pyarrow as pa
from .shared_aligned_ca import prepare,observe as ca_observe,finish as ca_finish

def consume(ctx,writer):
    audit={'attempted':True,'used':False,'reason':'cachealreadyavailable'}
    ctx._aligned_handoff_audit=audit
    if ctx.cache is not None:return False
    if ctx._stats is not None or ctx.m*3*8*6>ctx.budget//8:
        audit['reason']='statisticsguard';return False
    aux=ctx.n*16
    if aux>ctx.budget//128:
        audit['reason']='scalarbudget';return False
    shared=prepare(ctx)
    ca_bytes=0 if shared is None else shared['values'].nbytes+shared['ca'].nbytes
    raw=getattr(ctx,'_shared_raw_backbone',None)
    held=raw[-1] if isinstance(raw,tuple) and isinstance(raw[-1],int) else 0
    shape=getattr(ctx,'_shared_gyration',None)
    if isinstance(shape,dict):held+=shape.get('held',0)
    working=ctx.chunk*ctx.m*3*8*3+ctx.m*3*8*6
    if ca_bytes+aux+working+held>ctx.budget:
        audit['reason']='combinedbudget';return False
    state=dict(done=0,count=0,reference=None,mean=np.zeros((ctx.m,3)),m2=np.zeros((ctx.m,3)),values=np.empty(ctx.n),times=np.empty(ctx.n),shared=shared,writer=writer,m=ctx.m)
    ctx._aligned_handoff_state=state
    try:ctx.prepare_aligned()
    finally:ctx._aligned_handoff_state=None
    if state['done']!=ctx.n:
        audit['reason']='existingcommittedcachefallback';return False
    for start in range(0,ctx.n,ctx.chunk):
        stop=min(start+ctx.chunk,ctx.n)
        if not writer.has(start):writer.append(start,pa.table({'frame':np.arange(start,stop),'time_ps':state['times'][start:stop],'value':state['values'][start:stop]}))
    ctx._stats=state['mean'],np.maximum(state['m2'],0)/ctx.n
    ca_finish(ctx,shared)
    audit.update(used=True,reason='freshproducerhandoff',frames=ctx.n,scalar_bytes=aux,CA_bytes=ca_bytes,readback_avoided_logical_bytes=ctx.n*ctx.m*3*8)
    return True

def observe(state,start,stop,positions,times):
    if start!=state['done'] or stop-start!=len(positions):raise IOError('Aligned handoff continuity')
    if state['reference'] is None:state['reference']=positions[0].copy()
    ca_observe(state['shared'],start,stop,positions)
    mean=state['mean'];m2=state['m2'];count=state['count']
    for frame in positions:
        count+=1;delta=frame-mean;mean+=delta/count;m2+=delta*(frame-mean)
    state['count']=count
    if not state['writer'].has(start):state['values'][start:stop]=np.sqrt(np.sum((positions-state['reference'])**2,axis=(1,2))/state['m'])
    state['times'][start:stop]=times;state['done']=stop
