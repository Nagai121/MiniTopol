"""Strict point predicates; test nearest-first occluders and active point lists."""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from numba import njit,prange,get_num_threads,set_num_threads,get_thread_id
from .kernels import sphere_points
from .csr_sasa_baseline import _budgeted_sasa as original,_native_lock as _lock

@njit(cache=True,parallel=True,fastmath=False,nogil=True)
def counts(positions,inflated,points,offsets,neighbors,ordered,compact):
    result=np.empty(len(positions),np.int64)
    for atom in prange(len(positions)):
        center=positions[atom];surface=center+points*inflated[atom]
        nearby=neighbors[offsets[atom]:offsets[atom+1]]
        if ordered:
            distance=np.empty(len(nearby))
            for j in range(len(nearby)):
                x=positions[nearby[j],0]-center[0];y=positions[nearby[j],1]-center[1];z=positions[nearby[j],2]-center[2]
                distance[j]=x*x+y*y+z*z
            order=np.argsort(distance)
        else:order=np.arange(len(nearby))
        active=np.arange(len(points));exposed=np.ones(len(points),np.bool_);n=len(points)
        for j in order:
            other=nearby[j]
            if other==atom:continue
            radius2=inflated[other]**2
            if compact:
                kept=0
                for index in range(n):
                    p=active[index]
                    x=surface[p,0]-positions[other,0];y=surface[p,1]-positions[other,1];z=surface[p,2]-positions[other,2]
                    if not(x*x+y*y+z*z<radius2):
                        active[kept]=p;kept+=1
                n=kept
            else:
                for p in range(len(points)):
                    if exposed[p]:
                        x=surface[p,0]-positions[other,0];y=surface[p,1]-positions[other,1];z=surface[p,2]-positions[other,2]
                        if x*x+y*y+z*z<radius2:exposed[p]=False;n-=1
            if n==0:break
        result[atom]=n
    return result

@njit(cache=True,parallel=True,fastmath=False,nogil=True)
def _workspace_counts(positions,inflated,points,offsets,neighbors):
    workers=get_num_threads()
    surfaces=np.empty((workers,len(points),3),np.float64)
    active_points=np.empty((workers,len(points)),np.int64)
    result=np.empty(len(positions),np.int64)
    for atom in prange(len(positions)):
        tid=get_thread_id();surface=surfaces[tid];active=active_points[tid]
        center=positions[atom]
        for p in range(len(points)):
            active[p]=p
            for axis in range(3):
                surface[p,axis]=center[axis]+points[p,axis]*inflated[atom]
        nearby=neighbors[offsets[atom]:offsets[atom+1]]
        distance=np.empty(len(nearby))
        for j in range(len(nearby)):
            x=positions[nearby[j],0]-center[0];y=positions[nearby[j],1]-center[1];z=positions[nearby[j],2]-center[2]
            distance[j]=x*x+y*y+z*z
        order=np.argsort(distance);n=len(points)
        for j in order:
            other=nearby[j]
            if other==atom:continue
            radius2=inflated[other]**2;kept=0
            for index in range(n):
                p=active[index]
                x=surface[p,0]-positions[other,0];y=surface[p,1]-positions[other,1];z=surface[p,2]-positions[other,2]
                if not(x*x+y*y+z*z<radius2):
                    active[kept]=p;kept+=1
            n=kept
            if n==0:break
        result[atom]=n
    return result

def count_reusing(positions,inflated,points,offsets,neighbors,budget):
    # Leave unusual direct calls and tiny budgets on the original dispatcher.
    normal=(isinstance(positions,np.ndarray) and positions.dtype in (np.dtype('f4'),np.dtype('f8'))
            and inflated.dtype==np.dtype('f8') and points.shape==(960,3))
    workspace_bytes=get_num_threads()*len(points)*32
    if normal and workspace_bytes<=budget//64:
        return _workspace_counts(positions,inflated,points,offsets,neighbors)
    return counts(positions,inflated,points,offsets,neighbors,True,True)

@njit(cache=True,parallel=True,fastmath=False,nogil=True)
def counts_branchless(positions,inflated,points,offsets,neighbors,ordered):
    result=np.empty(len(positions),np.int64)
    for atom in prange(len(positions)):
        nearby=neighbors[offsets[atom]:offsets[atom+1]]
        if ordered:
            distance=np.empty(len(nearby))
            for j in range(len(nearby)):
                x=positions[nearby[j],0]-positions[atom,0];y=positions[nearby[j],1]-positions[atom,1];z=positions[nearby[j],2]-positions[atom,2]
                distance[j]=x*x+y*y+z*z
            order=np.argsort(distance)
        else:order=np.arange(len(nearby))
        sx=np.empty(len(points));sy=np.empty(len(points));sz=np.empty(len(points))
        for p in range(len(points)):
            sx[p]=positions[atom,0]+points[p,0]*inflated[atom]
            sy[p]=positions[atom,1]+points[p,1]*inflated[atom]
            sz[p]=positions[atom,2]+points[p,2]*inflated[atom]
        exposed=np.ones(len(points),np.bool_);n=len(points)
        for j in order:
            other=nearby[j]
            if other==atom:continue
            radius2=inflated[other]**2;n=0
            for p in range(len(points)):
                x=sx[p]-positions[other,0];y=sy[p]-positions[other,1];z=sz[p]-positions[other,2]
                blocked=x*x+y*y+z*z<radius2
                exposed[p]=exposed[p] & (not blocked)
                n+=int(exposed[p])
            if n==0:break
        result[atom]=n
    return result

def evaluate(positions,radii,budget,*,threads=4,ordered=True,compact=True,vectorized=False):
    inflated=np.asarray(radii)+1.4;points=sphere_points(960)
    tree=cKDTree(positions);radius=inflated+inflated.max()
    workers=min(threads,get_num_threads())
    lengths=tree.query_ball_point(positions,radius,return_length=True,workers=workers);total=int(lengths.sum())
    if total*64+len(positions)*256>budget//4:return original(positions,radii,budget)
    # Retain original exception/early-exit behavior for coincident inputs.
    if len(tree.query_pairs(0,output_type='ndarray')):return original(positions,radii,budget)
    lists=tree.query_ball_point(positions,radius,return_sorted=False,workers=workers)
    offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    neighbors=np.fromiter(itertools.chain.from_iterable(lists),dtype=np.int64,count=total)
    del lists
    with _lock:
        previous=get_num_threads()
        try:
            set_num_threads(min(threads,previous))
            accessible=(counts_branchless(positions,inflated,points,offsets,neighbors,ordered) if vectorized
                        else count_reusing(positions,inflated,points,offsets,neighbors,budget) if ordered and compact
                        else counts(positions,inflated,points,offsets,neighbors,ordered,compact))
        finally:set_num_threads(previous)
    unique,inverse=np.unique(inflated,return_inverse=True)
    # Preserve the original scalar radius arithmetic exactly. Only the finite
    # integer-count division (0..960) and final multiplication run as array ops.
    factors=np.asarray([4*np.pi*radius**2 for radius in unique])[inverse]
    return (accessible/960)*factors

def _budgeted_sasa(positions,radii,budget):
    from .knn_sasa import evaluate as knn_evaluate
    return knn_evaluate(positions,radii,budget)
