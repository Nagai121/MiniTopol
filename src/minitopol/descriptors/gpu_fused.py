"""Count all960strict points within one atom block; no global point flag array."""
from numba import cuda,int32,float64,float32
from numba.cuda import libdevice

@cuda.jit(fastmath=False)
def fused_counts(xyz,radii,points,lengths,neighbors,counts):
    q=cuda.blockIdx.x;tid=cuda.threadIdx.x;n=xyz.shape[1]
    frame=q//n;atom=q%n
    partial=cuda.shared.array(512,dtype=int32)
    centers=cuda.shared.array((128,3),dtype=float64)
    radii_squared=cuda.shared.array(128,dtype=float64)
    ids=cuda.shared.array(128,dtype=int32)
    relative_centers=cuda.shared.array((128,3),dtype=float32)
    length=lengths[frame,atom]
    if tid<length:
        other=neighbors[frame,atom,tid]
        ids[tid]=other
        centers[tid,0]=xyz[frame,other,0]
        centers[tid,1]=xyz[frame,other,1]
        centers[tid,2]=xyz[frame,other,2]
        relative_centers[tid,0]=float32(libdevice.dadd_rn(xyz[frame,other,0],-xyz[frame,atom,0]))
        relative_centers[tid,1]=float32(libdevice.dadd_rn(xyz[frame,other,1],-xyz[frame,atom,1]))
        relative_centers[tid,2]=float32(libdevice.dadd_rn(xyz[frame,other,2],-xyz[frame,atom,2]))
        radii_squared[tid]=libdevice.dmul_rn(radii[other],radii[other])
    cuda.syncthreads()
    cx=xyz[frame,atom,0];cy=xyz[frame,atom,1];cz=xyz[frame,atom,2];radius=radii[atom]
    own=0
    for p in range(tid,960,512):
        sx=libdevice.dadd_rn(cx,libdevice.dmul_rn(points[p,0],radius))
        sy=libdevice.dadd_rn(cy,libdevice.dmul_rn(points[p,1],radius))
        sz=libdevice.dadd_rn(cz,libdevice.dmul_rn(points[p,2],radius))
        fp_radius=float32(radius)
        px=libdevice.fmul_rn(float32(points[p,0]),fp_radius)
        py=libdevice.fmul_rn(float32(points[p,1]),fp_radius)
        pz=libdevice.fmul_rn(float32(points[p,2]),fp_radius)
        accessible=1
        for j in range(length):
            other=ids[j]
            if other!=atom:
                fx=libdevice.fadd_rn(px,-relative_centers[j,0])
                fy=libdevice.fadd_rn(py,-relative_centers[j,1])
                fz=libdevice.fadd_rn(pz,-relative_centers[j,2])
                coarse=libdevice.fadd_rn(libdevice.fadd_rn(libdevice.fmul_rn(fx,fx),libdevice.fmul_rn(fy,fy)),libdevice.fmul_rn(fz,fz))
                if float64(coarse)>radii_squared[j]+0.01:continue
                x=libdevice.dadd_rn(sx,-centers[j,0])
                square_x=libdevice.dmul_rn(x,x)
                if square_x>=radii_squared[j]:continue
                y=libdevice.dadd_rn(sy,-centers[j,1])
                square_xy=libdevice.dadd_rn(square_x,libdevice.dmul_rn(y,y))
                if square_xy>=radii_squared[j]:continue
                z=libdevice.dadd_rn(sz,-centers[j,2])
                square=libdevice.dadd_rn(square_xy,libdevice.dmul_rn(z,z))
                if square<radii_squared[j]:accessible=0;break
        own+=accessible
    partial[tid]=own;cuda.syncthreads()
    stride=256
    while stride:
        if tid<stride:partial[tid]+=partial[tid+stride]
        cuda.syncthreads();stride//=2
    if tid==0:counts[frame,atom]=partial[0]
