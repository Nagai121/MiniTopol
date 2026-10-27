"""Conservative fractional-torus candidates; exact MDA triclinic distances."""
import numpy as np
from scipy.spatial import cKDTree
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.lib.util import check_box
from ..coordinates import box_dimensions
from .periodic_contacts import UseOriginal

def frame_contacts(xyz,box,atom_residue,allowed,budget):
    dimensions=box_dimensions(box)
    if xyz.dtype!=np.dtype('f4') or not np.isfinite(xyz).all() or np.any(np.abs(xyz)>1e6):raise UseOriginal()
    size=len(xyz);count=len(allowed);fixed=size*256+count*count*9
    if size>512 or fixed>budget//4 or np.any(dimensions[:3]<1) or np.any(dimensions[:3]>1e4):raise UseOriginal()
    kind,cell=check_box(dimensions)
    if kind!='tri_vecs':raise UseOriginal()
    cell=cell.astype('f8');singular=np.linalg.svd(cell,compute_uv=False)
    if singular[-1]<1e-3 or singular[0]/singular[-1]>100:raise UseOriginal()
    inverse=np.linalg.inv(cell)
    margin=32*np.finfo(np.float32).eps*(float(np.abs(xyz).max(initial=0))+float(np.abs(cell).max())+1.)
    # If any MDA lattice image is within cutoff, fractional norm is <=r/sigma_min.
    # Torus nearest norm is no greater. Inflate both radius and inverse bound.
    radius=(4.5+margin)/(singular[-1]*(1-1e-6))
    wrapped=np.mod(xyz.astype('f8')@inverse,1.);wrapped=np.where(wrapped>=1.,0.,wrapped)
    tree=cKDTree(wrapped,boxsize=1.)
    worst=size*(size-1)//2
    if worst*16*4+fixed<=budget*7//8:
        pairs=tree.query_pairs(radius,output_type='ndarray')
        if len(pairs)*128+fixed>budget//4:raise UseOriginal()
    else:
        pair_count=(int(tree.count_neighbors(tree,radius))-size)//2
        if pair_count*128+fixed>budget//4:raise UseOriginal()
        pairs=tree.query_pairs(radius,output_type='ndarray')
    r=atom_residue[pairs[:,0]];s=atom_residue[pairs[:,1]];low=np.minimum(r,s);high=np.maximum(r,s);keep=allowed[low,high]
    pairs=pairs[keep];low=low[keep];high=high[keep]
    distances=calc_bonds(xyz[pairs[:,0]],xyz[pairs[:,1]],box=dimensions)
    minima=np.full((count,count),np.inf);np.minimum.at(minima,(low,high),distances)
    finite=np.isfinite(minima);np.less_equal(minima,4.5,out=finite,where=finite);rr,ss=np.nonzero(finite)
    if len(rr)*256>budget//8:raise UseOriginal()
    return rr,ss,minima[rr,ss]
