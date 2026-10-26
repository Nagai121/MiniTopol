"""Count all960strict points within one atom block; no global point flag array."""
from numba import cuda,int32
from numba.cuda import libdevice

@cuda.jit(fastmath=False)
def fused_counts(xyz,radii,points,lengths,neighbors,counts):
    q=cuda.blockIdx.x;tid=cuda.threadIdx.x;n=xyz.shape[1]
    frame=q//n;atom=q%n
    partial=cuda.shared.array(128,dtype=int32)
    cx=xyz[frame,atom,0];cy=xyz[frame,atom,1];cz=xyz[frame,atom,2];radius=radii[atom]
    own=0
    for p in range(tid,960,128):
        sx=libdevice.dadd_rn(cx,libdevice.dmul_rn(points[p,0],radius))
        sy=libdevice.dadd_rn(cy,libdevice.dmul_rn(points[p,1],radius))
        sz=libdevice.dadd_rn(cz,libdevice.dmul_rn(points[p,2],radius))
        accessible=1
        for j in range(lengths[frame,atom]):
            other=neighbors[frame,atom,j]
            if other!=atom:
                x=libdevice.dadd_rn(sx,-xyz[frame,other,0]);y=libdevice.dadd_rn(sy,-xyz[frame,other,1]);z=libdevice.dadd_rn(sz,-xyz[frame,other,2])
                square=libdevice.dadd_rn(libdevice.dadd_rn(libdevice.dmul_rn(x,x),libdevice.dmul_rn(y,y)),libdevice.dmul_rn(z,z))
                if square<libdevice.dmul_rn(radii[other],radii[other]):accessible=0;break
        own+=accessible
    partial[tid]=own;cuda.syncthreads()
    stride=64
    while stride:
        if tid<stride:partial[tid]+=partial[tid+stride]
        cuda.syncthreads();stride//=2
    if tid==0:counts[frame,atom]=partial[0]
