"""Bounded atomic output batches preserving scientific values and row order."""
import numpy as np
import pyarrow as pa
from . import sasa
from .kernels import backbone_frames,secondary_labels
from .precision64 import backbone_frames64
from .correlation import atom_correlation

def _frames(ctx):
    # Conservative upper bound for simultaneous Python rows, Arrow tables,
    # concat metadata and coordinates; fallback to a single frame if needed.
    return max(1,min(16,ctx.budget//max(1,ctx.m*1024*2)))

class _CommittedFrames:
    def __init__(self,ctx,writer,batch):self.ctx,self.writer,self.batch=ctx,writer,batch
    def has(self,frame):
        start=(frame//self.batch)*self.batch
        return self.writer.has(f'frames-{start}-{min(start+self.batch,self.ctx.n)}')


def _backbone_table(name,module,ctx,frame,time,angles,why):
    if name=='ramachandran':
        n=len(ctx.residues)*2
        values=np.column_stack((angles['phi'],angles['psi'])).ravel()
        valid=np.isfinite(values)
        # Match the previous Arrow builder's zero-filled null slots as well.
        values=np.where(valid,values,0.)
        return pa.Table.from_arrays([
            pa.array(np.full(n,frame,dtype=np.int64)),pa.array(np.full(n,time,dtype=np.float64)),
            pa.array(np.repeat(ctx.residues,2),type=pa.int64()),
            pa.array(['phi','psi']*len(ctx.residues),type=pa.string()),
            pa.array(values,mask=~valid,type=pa.float64()),
            pa.array(np.column_stack((why['phi'],why['psi'])).ravel().tolist(),type=pa.string())],schema=module.SCHEMA)
    labels=secondary_labels(angles['phi'],angles['psi']);n=len(ctx.residues)
    return pa.Table.from_arrays([
        pa.array(np.full(n,frame,dtype=np.int64)),pa.array(np.full(n,time,dtype=np.float64)),
        pa.array(ctx.residues,type=pa.int64()),pa.array(labels,type=pa.string()),
        pa.array([(why['phi'][i] or why['psi'][i]) if labels[i] is None else None
                  for i in range(n)],type=pa.string())],schema=module.SCHEMA)

def _backbone(name,module,ctx,writer):
    batch=_frames(ctx);view=_CommittedFrames(ctx,writer,batch);tables=[];start=None
    frames=backbone_frames64(ctx,view) if name=='ramachandran' else backbone_frames(ctx,view)
    for frame,time,angles,why in frames:
        if start is None:start=frame
        table=_backbone_table(name,module,ctx,frame,time,angles,why)
        tables.append(table)
        if (frame+1)%batch==0 or frame+1==ctx.n:
            writer.append(f'frames-{start}-{frame+1}',pa.concat_tables(tables));tables=[];start=None
    assert not tables

def _sasa(ctx,writer):
    batch=_frames(ctx);radii=ctx.atomic_values('radii');tables=[];start=None
    for block in ctx.coordinate_blocks():
        for offset,xyz in enumerate(block.positions):
            frame=block.start+offset;first=(frame//batch)*batch;last=min(first+batch,ctx.n)
            if writer.has(f'frames-{first}-{last}'):continue
            if start is None:start=frame
            values=sasa._budgeted_sasa(xyz,radii,ctx.budget) if sasa._budgeted_sasa is not None else sasa.sasa_atoms(xyz,radii)
            tables.extend(table for _,table in sasa.output_tables(frame,block.time[offset],values,ctx.atoms['residue_index'],ctx.res_atoms))
            if frame+1==last:
                writer.append(f'frames-{start}-{last}',pa.concat_tables(tables));tables=[];start=None
    assert not tables

def _time_correlation(ctx,writer):
    step=ctx.timestep();mean,variance=ctx.statistics()
    # Output batching only; FFT kernels/accumulation and coordinate chunks unchanged.
    output_chunk=max(1,min(ctx.n,ctx.budget//1024))
    for point in ctx.ca_indices():
        if writer.has(f'{point}-done'):continue
        residual=variance[point].sum();dots=None
        if residual>0:dots,_=atom_correlation(ctx,point,mean[point],ctx.cache_path/'time-correlation-scratch.zarr')
        for start in range(0,ctx.n,output_chunk):
            stop=min(start+output_chunk,ctx.n);lag=np.arange(start,stop)
            value=dots[start:stop]/((ctx.n-lag)*residual) if residual>0 else np.zeros(len(lag))
            if residual>0:value[lag==0]=1
            table=pa.table({'point':np.full(len(lag),point),'residue':np.full(len(lag),ctx.atoms['residue_index'][point]),
                            'lag_frames':lag,'lag_ps':lag*step,'origins':ctx.n-lag,
                            'value':pa.array(value,mask=np.full(len(lag),residual==0)),
                            'reason':pa.array(['zero_variance' if residual==0 else None]*len(lag),type=pa.string())})
            writer.append(f'{point}-done' if stop==ctx.n else f'{point}-{start}',table)

def calculate(name,module,ctx,writer):
    if name=='sasa':return _sasa(ctx,writer)
    if name in ('ramachandran','secondary_structure'):return _backbone(name,module,ctx,writer)
    if name=='time_correlation':return _time_correlation(ctx,writer)
    return module.calculate(ctx,writer)
