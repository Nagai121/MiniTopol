"""Reuse one authoritative MDA matrix; reduce minima without new distances."""
import numpy as np
from numba import njit
from MDAnalysis.lib.distances import distance_array

@njit(cache=True,nogil=True,fastmath=False)
def _reduce(matrix,points,offsets,pairs):
    result=np.empty(len(pairs),dtype=np.float64)
    for p in range(len(pairs)):
        r,s=pairs[p,0],pairs[p,1]
        best=np.inf
        for i in range(offsets[r],offsets[r+1]):
            for j in range(offsets[s],offsets[s+1]):
                value=matrix[points[i],points[j]]
                if value<best:best=value
        result[p]=best
    return result

def prepare(groups,pairs,n):
    try:
        keys=sorted(groups);ordinal={r:i for i,r in enumerate(keys)}
        offsets=np.r_[0,np.cumsum([len(groups[r]) for r in keys])].astype(np.int64)
        points=np.asarray([i for r in keys for i in groups[r]],dtype=np.int64)
        if not len(points) or np.any(points<0) or np.any(points>=n):return None
        indices=np.asarray([(ordinal[r],ordinal[s]) for r,s in pairs],dtype=np.int64).reshape(-1,2)
        return points,offsets,indices
    except (KeyError,TypeError,ValueError,OverflowError):return None

def frame_minima(xyz,box,plan,budget):
    if plan is None or xyz.dtype!=np.dtype('f4') or xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():return None
    if len(xyz)**2*9+sum(a.nbytes for a in plan)>budget//4:return None
    matrix=distance_array(xyz,xyz,box=box)
    if not np.isfinite(matrix).all():return None
    return _reduce(matrix,*plan)
