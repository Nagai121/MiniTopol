"""Nearest-first GPU traversal, without CPU neighbor sorting."""
from numba import cuda,float64,int32
from numba.cuda import libdevice

@cuda.jit(fastmath=False)
def sort_nearest(xyz,lengths,neighbors):
    q=cuda.blockIdx.x;tid=cuda.threadIdx.x;n=xyz.shape[1]
    frame=q//n;atom=q%n
    distance=cuda.shared.array(128,dtype=float64);ids=cuda.shared.array(128,dtype=int32)
    count=lengths[frame,atom]
    if tid<count:
        other=neighbors[frame,atom,tid];ids[tid]=other
        x=libdevice.dadd_rn(xyz[frame,other,0],-xyz[frame,atom,0]);y=libdevice.dadd_rn(xyz[frame,other,1],-xyz[frame,atom,1]);z=libdevice.dadd_rn(xyz[frame,other,2],-xyz[frame,atom,2])
        distance[tid]=libdevice.dadd_rn(libdevice.dadd_rn(libdevice.dmul_rn(x,x),libdevice.dmul_rn(y,y)),libdevice.dmul_rn(z,z))
    else:distance[tid]=float('inf');ids[tid]=-1
    cuda.syncthreads()
    size=1
    while size<count:size*=2
    width=2
    while width<=size:
        stride=width//2
        while stride>0:
            partner=tid^stride
            if tid<size and partner>tid:
                reverse=(tid&width)!=0
                swap=(distance[tid]<distance[partner]) if reverse else (distance[tid]>distance[partner])
                if swap:
                    value=distance[tid];distance[tid]=distance[partner];distance[partner]=value
                    index=ids[tid];ids[tid]=ids[partner];ids[partner]=index
            cuda.syncthreads();stride//=2
        width*=2
    if tid<count:neighbors[frame,atom,tid]=ids[tid]
