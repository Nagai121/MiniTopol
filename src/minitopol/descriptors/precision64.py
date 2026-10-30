"""A-1 backbone dihedrals: float64 vectors throughout, unchanged raw A."""
import numpy as np
from .kernels import backbone_quads
from ..coordinates import minimum_vectors

def _index_plan(quads):
    plan={}
    for angle in ('phi','psi'):
        usable=[i for i,q in enumerate(quads[angle]) if q is not None]
        plan[angle]=(usable,np.asarray([quads[angle][i] for i in usable]))
    return plan


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


def _planned_angles64(positions,box,quads,reasons,plan,*,periodic):
    positions=np.asarray(positions,dtype=np.float64)
    values={};why={name:list(items) for name,items in reasons.items()}
    for angle in ('phi','psi'):
        result=np.full(len(quads[angle]),np.nan)
        usable,indices=plan[angle]
        if usable:
            xyz=positions[indices]
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

def _backbone_frames64_scalar(ctx,writer):
    quads,reasons=backbone_quads(ctx)
    plan=_index_plan(quads)
    for block in ctx.registry.blocks(ctx.trajectory,chunk_frames=ctx.chunk):
        for offset,xyz in enumerate(block.positions):
            frame=block.start+offset
            if writer.has(frame):continue
            values,why=_planned_angles64(xyz,block.box[offset],quads,reasons,plan,periodic=ctx.mode!='nonperiodic')
            yield frame,block.time[offset],values,why

def backbone_frames64(ctx,writer):
    """Bounded frame batching; every dihedral retains scalar float64 math."""
    quads,reasons=backbone_quads(ctx)
    from .shared_raw_backbone import source
    quads,blocks,held=source(ctx,quads);plan=_index_plan(quads)
    nr=len(quads['phi'])
    batch=max(1,min(16,(ctx.budget-held)//max(1,ctx.m*24+nr*1024)//16))
    for block in blocks:
        for start in range(0,len(block.positions),batch):
            offsets=[i for i in range(start,min(start+batch,len(block.positions))) if not writer.has(block.start+i)]
            if not offsets:continue
            positions=block.positions[offsets];count=len(offsets)
            values={};why={a:[list(reasons[a]) for _ in offsets] for a in ('phi','psi')}
            for angle in ('phi','psi'):
                result=np.full((count,len(quads[angle])),np.nan)
                usable,indices=plan[angle]
                if usable:
                    xyz=np.asarray(positions[:,indices],dtype=np.float64)
                    vectors=np.diff(xyz,axis=2)
                    if ctx.mode!='nonperiodic':
                        for f,offset in enumerate(offsets):vectors[f]=minimum_vectors(vectors[f],block.box[offset])
                    flat=vectors.reshape(-1,3,3)
                    a,b,c=flat[:,0],flat[:,1],flat[:,2]
                    n1=np.cross(a,b);n2=np.cross(b,c)
                    x=np.einsum('ij,ij->i',n1,n2)
                    with np.errstate(divide='ignore',invalid='ignore'):
                        y=np.einsum('ij,ij->i',np.cross(n1,n2),b)/np.linalg.norm(b,axis=1)
                        radians=np.arctan2(y,x)
                    radians[(x==0)&(y==0)]=np.nan
                    angles=((np.rad2deg(radians)+180)%360-180).reshape(count,len(usable))
                    result[:,usable]=angles
                    for f,j in np.argwhere(~np.isfinite(angles)):why[angle][f][usable[j]]='degenerate_dihedral'
                values[angle]=result
            for f,offset in enumerate(offsets):
                yield block.start+offset,block.time[offset],{a:values[a][f] for a in values},{a:why[a][f] for a in why}
