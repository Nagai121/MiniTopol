"""A-1 backbone dihedrals: float64 vectors throughout, unchanged raw A."""
import numpy as np
from .kernels import backbone_quads
from ..coordinates import minimum_vectors

def backbone_angles64(positions,box,quads,reasons,*,periodic):
    positions=np.asarray(positions,dtype=np.float64)
    values={};why={name:list(items) for name,items in reasons.items()}
    for angle in ('phi','psi'):
        result=np.full(len(quads[angle]),np.nan)
        usable=[i for i,q in enumerate(quads[angle]) if q is not None]
        if usable:
            xyz=positions[np.asarray([quads[angle][i] for i in usable])]
            vectors=np.diff(xyz,axis=1)
            if periodic:vectors=minimum_vectors(vectors,box)
            a,b,c=vectors[:,0],vectors[:,1],vectors[:,2]
            n1=np.cross(a,b);n2=np.cross(b,c)
            x=np.einsum('ij,ij->i',n1,n2)
            with np.errstate(divide='ignore',invalid='ignore'):
                y=np.einsum('ij,ij->i',np.cross(n1,n2),b)/np.linalg.norm(b,axis=1)
                radians=np.arctan2(y,x)
            radians[(x==0)&(y==0)]=np.nan
            result[usable]=(np.rad2deg(radians)+180)%360-180
            for i in usable:
                if not np.isfinite(result[i]):why[angle][i]='degenerate_dihedral'
        values[angle]=result
    return values,why

def backbone_frames64(ctx,writer):
    quads,reasons=backbone_quads(ctx)
    for block in ctx.registry.blocks(ctx.trajectory,chunk_frames=ctx.chunk):
        for offset,xyz in enumerate(block.positions):
            frame=block.start+offset
            if writer.has(frame):continue
            values,why=backbone_angles64(xyz,block.box[offset],quads,reasons,periodic=ctx.mode!='nonperiodic')
            yield frame,block.time[offset],values,why
