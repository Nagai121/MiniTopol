"""External prototype: bounded native neighbor arrays, exact boundary fallback.

No approximate SASA predicates. Normal float32/float64 coordinates retain
their complete precision; boundary-neighbor queries use the legacy predicate.
"""
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree
from numba import get_num_threads,set_num_threads
from . import csr_sasa as legacy

def _neighbors(positions,inflated,budget):
 # Distances/indices/selection plus simultaneous subtraction/abs temporaries:
 # <=33*n*cap bytes; reserve40*n*cap plus256*n for small arrays/tree bookkeeping.
 if not isinstance(positions,np.ndarray) or positions.ndim!=2 or positions.shape[1]!=3:return None
 if np.ndim(inflated)!=1 or len(inflated)!=len(positions):return None
 n=len(positions);cap=min(128,(budget//4-n*256)//max(1,n*40))
 if cap<8 or positions.dtype not in [np.dtype('f4'),np.dtype('f8')]:return None
 if np.any(np.abs(positions)>1e6) or not np.all(np.isfinite(inflated)) or np.any(inflated<.1) or np.any(inflated>10):return None
 # Avoid underflow-sensitive exotic inputs without rounding normal NoJump f8
 # coordinates to f4. At the guarded scale the exact-boundary shell exceeds
 # double distance/sqrt error by several orders of magnitude.
 if positions.dtype==np.dtype('f8') and np.any((np.abs(positions)>0)&(np.abs(positions)<1e-20)):return None
 tree=cKDTree(positions)
 radius=inflated+inflated.max();margin=1e-8*max(1.,float(radius.max()))
 distances,indices=tree.query(positions,k=cap,distance_upper_bound=float(radius.max())+margin,workers=min(8,get_num_threads()))
 if n>1 and np.any(distances[:,1]==0):return None
 # Saturation cannot silently drop a neighbor. Retain the complete legacy path.
 if np.any(np.isfinite(distances[:,-1])):return None
 selected=distances<=radius[:,None]
 boundary=np.flatnonzero(np.any(np.abs(distances-radius[:,None])<=margin,axis=1))
 for atom in boundary:
  exact=tree.query_ball_point(positions[atom],radius[atom],return_sorted=False)
  if len(exact)>=cap:return None
  indices[atom,:len(exact)]=exact;selected[atom]=False;selected[atom,:len(exact)]=True
 lengths=selected.sum(axis=1,dtype=np.int64);total=int(lengths.sum())
 # Preserve the previous fallback boundary as well as its scientific path.
 if total*64+n*256>budget//4:return None
 neighbors=indices[selected];offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
 return offsets,neighbors,len(boundary)

@lru_cache(maxsize=2)
def _constants(dtype,shape,raw):
 # Private immutable-in-use arrays; public sphere_points still returns fresh
 # arrays. The caller's content, shape and dtype are part of the cache key.
 radii=np.frombuffer(raw,dtype=np.dtype(dtype)).reshape(shape)
 inflated=np.asarray(radii)+1.4
 points=legacy.sphere_points(960)
 unique,inverse=np.unique(inflated,return_inverse=True)
 factors=np.asarray([4*np.pi*r**2 for r in unique])[inverse]
 return inflated,points,factors

def evaluate(positions,radii,budget):
 # Keep the original guarded/fallback path before preparing cached constants.
 # Unsupported inputs must retain their original warnings and error order.
 inflated=np.asarray(radii)+1.4
 built=_neighbors(positions,inflated,budget)
 if built is None:return legacy.evaluate(positions,radii,budget,threads=8,ordered=True,compact=True)
 offsets,neighbors,_=built
 array=np.asarray(radii);prepared=None
 if array.ndim==1 and array.dtype.kind in 'fi' and array.nbytes*8+960*3*8<=min(512*1024,budget//128):
  prepared=_constants(array.dtype.str,array.shape,array.tobytes())
 points=legacy.sphere_points(960) if prepared is None else prepared[1]
 with legacy._lock:
  previous=get_num_threads()
  try:
   set_num_threads(min(8,previous));accessible=legacy.count_reusing(positions,inflated,points,offsets,neighbors,budget)
  finally:set_num_threads(previous)
 if prepared is None:
  unique,inverse=np.unique(inflated,return_inverse=True);factors=np.asarray([4*np.pi*r**2 for r in unique])[inverse]
 else:factors=prepared[2]
 return (accessible/960)*factors
