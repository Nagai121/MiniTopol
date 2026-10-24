"""External prototype: bounded native neighbor arrays, exact boundary fallback.

No approximate SASA predicates. Restrict fast path to original float32-valued
coordinates and normal radii; all other inputs retain the selected evaluator.
"""
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
 if positions.dtype==np.dtype('f8') and not np.array_equal(positions,positions.astype(np.float32).astype(np.float64)):return None
 tree=cKDTree(positions)
 if len(np.unique(positions,axis=0))!=n:return None
 radius=inflated+inflated.max();margin=1e-8*max(1.,float(radius.max()))
 distances,indices=tree.query(positions,k=cap,distance_upper_bound=float(radius.max())+margin,workers=min(8,get_num_threads()))
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

def evaluate(positions,radii,budget):
 inflated=np.asarray(radii)+1.4
 built=_neighbors(positions,inflated,budget)
 if built is None:return legacy.evaluate(positions,radii,budget,threads=8,ordered=True,compact=True)
 offsets,neighbors,_=built;points=legacy.sphere_points(960)
 with legacy._lock:
  previous=get_num_threads()
  try:
   set_num_threads(min(8,previous));accessible=legacy.counts(positions,inflated,points,offsets,neighbors,True,True)
  finally:set_num_threads(previous)
 unique,inverse=np.unique(inflated,return_inverse=True);factors=np.asarray([4*np.pi*r**2 for r in unique])[inverse]
 return (accessible/960)*factors
