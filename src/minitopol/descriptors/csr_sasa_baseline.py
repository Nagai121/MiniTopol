"""Whole-frame native parallel fallback with bounded neighbor CSR."""
import threading
import numpy as np
from scipy.spatial import cKDTree
from numba import njit,prange,get_num_threads,set_num_threads
from .kernels import sphere_points,sasa_atoms

_native_lock=threading.RLock()


@njit(cache=True,parallel=True,fastmath=False,nogil=True)
def _counts(positions,inflated,points,offsets,neighbors):
    counts=np.empty(len(positions),np.int64)
    for atom in prange(len(positions)):
        center=positions[atom]
        surface=center+points*inflated[atom]
        exposed=np.ones(len(points),np.bool_);count=len(points)
        for offset in range(offsets[atom],offsets[atom+1]):
            other=neighbors[offset]
            if other==atom:continue
            if (center[0]==positions[other,0] and center[1]==positions[other,1] and center[2]==positions[other,2]):
                count=-1;break
            radius2=inflated[other]**2
            for point in range(len(points)):
                if exposed[point]:
                    x=surface[point,0]-positions[other,0]
                    y=surface[point,1]-positions[other,1]
                    z=surface[point,2]-positions[other,2]
                    if x*x+y*y+z*z<radius2:
                        exposed[point]=False;count-=1
            if count==0:break
        counts[atom]=count
    return counts


def _budgeted_sasa(positions,radii,budget):
    inflated=np.asarray(radii)+1.4;points=sphere_points(960)
    tree=cKDTree(positions);radius=inflated+inflated.max()
    lengths=tree.query_ball_point(positions,radius,return_length=True)
    total=int(lengths.sum())
    # Includes Python neighbor objects, CSR buffers and allocator headroom.
    if total*64+len(positions)*256>budget//4:
        return sasa_atoms(positions,radii)
    lists=tree.query_ball_point(positions,radius,return_sorted=False)
    offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    neighbors=np.empty(total,np.int64)
    for atom,items in enumerate(lists):neighbors[offsets[atom]:offsets[atom+1]]=items
    del lists
    # Safe even if the installed Numba selects a non-reentrant workqueue.
    with _native_lock:
        previous=get_num_threads()
        try:
            set_num_threads(min(4,previous))
            counts=_counts(positions,inflated,points,offsets,neighbors)
        finally:set_num_threads(previous)
    if np.any(counts<0):raise ValueError('Coincident heavy atoms make per-atom SASA assignment ambiguous')
    values=np.empty(len(positions))
    for atom,count in enumerate(counts):
        values[atom]=int(count)/960*(4*np.pi*inflated[atom]**2)
    return values
