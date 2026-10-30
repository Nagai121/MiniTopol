"""Experimental relative-float32 960point kernel; neighbor discovery remains original."""
from numba import cuda,int32,float32
from numba.cuda import libdevice
@cuda.jit(fastmath=False)
def fused_counts(xyz,radii,points,lengths,neighbors,counts):
    q=cuda.blockIdx.x;tid=cuda.threadIdx.x;n=xyz.shape[1];frame=q//n;atom=q%n
    partial=cuda.shared.array(512,dtype=int32);relative=cuda.shared.array((128,3),dtype=float32);r2=cuda.shared.array(128,dtype=float32);ids=cuda.shared.array(128,dtype=int32)
    length=lengths[frame,atom]
    if tid<length:
        other=neighbors[frame,atom,tid];ids[tid]=other
        relative[tid,0]=float32(libdevice.dadd_rn(xyz[frame,other,0],-xyz[frame,atom,0]))
        relative[tid,1]=float32(libdevice.dadd_rn(xyz[frame,other,1],-xyz[frame,atom,1]))
        relative[tid,2]=float32(libdevice.dadd_rn(xyz[frame,other,2],-xyz[frame,atom,2]))
        r2[tid]=float32(libdevice.dmul_rn(radii[other],radii[other]))
    cuda.syncthreads();radius=float32(radii[atom]);own=0
    for p in range(tid,960,512):
        px=libdevice.fmul_rn(float32(points[p,0]),radius);py=libdevice.fmul_rn(float32(points[p,1]),radius);pz=libdevice.fmul_rn(float32(points[p,2]),radius);accessible=1
        for j in range(length):
            if ids[j]!=atom:
                x=libdevice.fadd_rn(px,-relative[j,0]);y=libdevice.fadd_rn(py,-relative[j,1]);z=libdevice.fadd_rn(pz,-relative[j,2])
                square=libdevice.fadd_rn(libdevice.fadd_rn(libdevice.fmul_rn(x,x),libdevice.fmul_rn(y,y)),libdevice.fmul_rn(z,z))
                if square<r2[j]:accessible=0;break
        own+=accessible
    partial[tid]=own;cuda.syncthreads();stride=256
    while stride:
        if tid<stride:partial[tid]+=partial[tid+stride]
        cuda.syncthreads();stride//=2
    if tid==0:counts[frame,atom]=partial[0]
