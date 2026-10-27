"""Native candidate-distance-minimum fusion, reproducing installed MDA ortho C arithmetic."""
import math,numpy as np
from numba import njit
from scipy.spatial import cKDTree
from ..coordinates import box_dimensions
from .periodic_contacts import _frame_contacts_original as _frame_contacts
@njit(cache=True,nogil=True,fastmath=False)
def reduce_pairs(xyz,pairs,res,allowed,box):
 count=len(allowed);minimum=np.full((count,count),np.inf);inv=np.empty(3,np.float32)
 for k in range(3):inv[k]=np.float32(1.0/np.float64(box[k]))
 for pair in pairs:
  i,j=pair[0],pair[1];r=res[i];s=res[j]
  lo=min(r,s);hi=max(r,s)
  if not allowed[lo,hi]:continue
  x=np.float64(np.float32(xyz[i,0]-xyz[j,0]));fraction=np.float64(inv[0])*x
  rounded=math.floor(fraction+0.5) if fraction>=0 else math.ceil(fraction-0.5)
  dx=np.float64(box[0])*(fraction-rounded)
  x=np.float64(np.float32(xyz[i,1]-xyz[j,1]));fraction=np.float64(inv[1])*x
  rounded=math.floor(fraction+0.5) if fraction>=0 else math.ceil(fraction-0.5)
  dy=np.float64(box[1])*(fraction-rounded)
  x=np.float64(np.float32(xyz[i,2]-xyz[j,2]));fraction=np.float64(inv[2])*x
  rounded=math.floor(fraction+0.5) if fraction>=0 else math.ceil(fraction-0.5)
  dz=np.float64(box[2])*(fraction-rounded)
  distance=math.sqrt((dx*dx+dy*dy)+dz*dz)
  if distance<minimum[lo,hi]:minimum[lo,hi]=distance
 return minimum
def frame(xyz,box,res,allowed,budget):
 dims=box_dimensions(box);m=len(xyz);count=len(allowed)
 if not np.all(dims[3:]==90) or xyz.dtype!=np.dtype('f4') or not np.isfinite(xyz).all() or np.max(np.abs(xyz),initial=0)>1e6 or np.any(dims[:3]<1) or np.any(dims[:3]>1e4):return _frame_contacts(xyz,box,res,allowed,budget)
 if m*256+count*count*9>budget//4:return _frame_contacts(xyz,box,res,allowed,budget)
 lengths=dims[:3].astype(float);margin=32*np.finfo('f4').eps*(float(np.abs(xyz).max(initial=0))+float(lengths.max())+1);wrapped=np.mod(xyz.astype(float),lengths);wrapped=np.where(wrapped>=lengths,0.,wrapped);tree=cKDTree(wrapped,boxsize=lengths);radius=4.5+margin;fixed=m*256+count*count*9
 worst=m*(m-1)//2
 if worst*16*4+fixed<=budget*7//8:
  pairs=tree.query_pairs(radius,output_type='ndarray')
  if len(pairs)*128+fixed>budget//4:return _frame_contacts(xyz,box,res,allowed,budget)
 else:
  n=(int(tree.count_neighbors(tree,radius))-m)//2
  if n*128+fixed>budget//4:return _frame_contacts(xyz,box,res,allowed,budget)
  pairs=tree.query_pairs(radius,output_type='ndarray')
 minimum=reduce_pairs(xyz,pairs,res,allowed,dims[:3]);rr,ss=np.nonzero(minimum<=4.5)
 if len(rr)*256>budget//8:return _frame_contacts(xyz,box,res,allowed,budget)
 return rr,ss,minimum[rr,ss]
def bundle(positions,boxes,res,allowed,budget):return [frame(x,b,res,allowed,budget) for x,b in zip(positions,boxes)],dict(reused=False,frames=len(positions),kernel='fused_distance_minimum')
