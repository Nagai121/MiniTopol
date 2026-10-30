"""Exact frame-major SASA columns for an existing publication batch."""
import numpy as np
import pyarrow as pa
from .sasa import SCHEMA

def table(ctx,frames,times,values):
    m=ctx.m;residues=list(ctx.res_atoms);nr=len(residues);width=m+nr+1;count=len(frames)
    columns=np.empty((count,width),dtype=np.float64)
    for i,v in enumerate(values):
        columns[i,:m]=v
        columns[i,m:m+nr]=np.asarray([v[points].sum() for points in ctx.res_atoms.values()])
        columns[i,-1]=v.sum()
    point=np.r_[np.arange(m,dtype=np.int64),np.zeros(nr+1,dtype=np.int64)]
    residue=np.r_[np.asarray(ctx.atoms['residue_index'],dtype=np.int64),np.asarray(residues,dtype=np.int64),np.zeros(1,dtype=np.int64)]
    point_mask=np.r_[np.zeros(m,dtype=bool),np.ones(nr+1,dtype=bool)]
    residue_mask=np.r_[np.zeros(m+nr,dtype=bool),np.ones(1,dtype=bool)]
    return pa.Table.from_arrays([
        pa.array(np.repeat(np.asarray(frames,dtype=np.int64),width)),
        pa.array(np.repeat(np.asarray(times,dtype=np.float64),width)),
        pa.array((['atom']*m+['residue']*nr+['total'])*count,type=pa.string()),
        pa.array(np.tile(point,count),mask=np.tile(point_mask,count)),
        pa.array(np.tile(residue,count),mask=np.tile(residue_mask,count)),
        pa.array(columns.ravel())],schema=SCHEMA)
