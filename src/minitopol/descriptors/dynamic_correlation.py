"""Same pair/chunk reduction order with one aligned CA read per time chunk."""
import numpy as np
import pyarrow as pa
from .dynamic_correlation_legacy import SCHEMA,calculate as original

def calculate(ctx,writer):
    ca=ctx.ca_indices();size=len(ca)
    if size<3 or size*size*8>ctx.budget//4:return original(ctx,writer)
    mean,variance=ctx.statistics();pairs=[]
    for left in range(0,size,ctx.pair_block):
        for right in range(left,size,ctx.pair_block):
            key=f'{left}-{right}'
            if not writer.has(key):pairs.append((left,right,key))
    products=np.zeros((size,size))
    for start in range(0,ctx.n,ctx.chunk):
        if not pairs:break
        stop=min(start+ctx.chunk,ctx.n)
        from .shared_aligned_ca import read
        displacement=read(ctx,start,stop,ca)-mean[ca]
        for left,right,_ in pairs:
            end_left=min(left+ctx.pair_block,size);end_right=min(right+ctx.pair_block,size)
            # Match legacy contiguous operand shape, axis order and summation.
            x=np.ascontiguousarray(displacement[:,left:end_left,:])
            y=x if right==left else np.ascontiguousarray(displacement[:,right:end_right,:])
            products[left:end_left,right:end_right]+=np.einsum('tic,tjc->ij',x,y)
    for left,right,key in pairs:
        a=ca[left:left+ctx.pair_block];b=ca[right:right+ctx.pair_block];rows=[]
        for i,point_i in enumerate(a):
            for j,point_j in enumerate(b):
                if left+i>=right+j:continue
                denominator=np.sqrt(variance[point_i].sum()*variance[point_j].sum())
                value=products[left+i,right+j]/ctx.n/denominator if denominator>0 else None
                if value is not None:
                    if abs(value)>1+1e-10:raise ArithmeticError('Dynamic correlation violates its normalization bound')
                    value=float(np.clip(value,-1,1))
                r,s=sorted([ctx.atoms['residue_index'][point_i],ctx.atoms['residue_index'][point_j]])
                rows.append(dict(residue_i=r,residue_j=s,frames=ctx.n,value=value,reason='zero_variance' if value is None else None))
        writer.append(key,pa.Table.from_pylist(rows,schema=SCHEMA))
