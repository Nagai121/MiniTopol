"""Direct conservative GPU neighbor compaction, no quadratic global mask."""
from numba import cuda,int32
from numba.cuda import libdevice

@cuda.jit(fastmath=False)
def direct_neighbors(xyz,radii,max_radius,margin,lengths,neighbors,bad):
    q=cuda.blockIdx.x;n=xyz.shape[1];frame=q//n;i=q%n;tid=cuda.threadIdx.x
    count=cuda.shared.array(1,int32)
    if tid==0:count[0]=0
    cuda.syncthreads()
    radius=libdevice.dadd_rn(libdevice.dadd_rn(radii[i],max_radius),margin)
    threshold=libdevice.dmul_rn(radius,radius)
    lane=tid%32
    for base in range(0,n,128):
        j=base+tid;keep=False
        if j<n:
            x=libdevice.dadd_rn(xyz[frame,i,0],-xyz[frame,j,0]);y=libdevice.dadd_rn(xyz[frame,i,1],-xyz[frame,j,1]);z=libdevice.dadd_rn(xyz[frame,i,2],-xyz[frame,j,2])
            d=libdevice.dadd_rn(libdevice.dadd_rn(libdevice.dmul_rn(x,x),libdevice.dmul_rn(y,y)),libdevice.dmul_rn(z,z))
            if i!=j and d<=0.0001:cuda.atomic.max(bad,0,1)
            keep=d<=threshold
        # All32 lanes participate even in the ragged final atom tile.
        mask=cuda.ballot_sync(0xffffffff,keep);hits=cuda.popc(mask)
        offset=0
        if lane==0 and hits:offset=cuda.atomic.add(count,0,hits)
        offset=cuda.shfl_sync(0xffffffff,offset,0)
        if keep:
            index=offset+cuda.popc(mask&((1<<lane)-1))
            if index<neighbors.shape[2]:neighbors[frame,i,index]=j
    cuda.syncthreads()
    if tid==0:
        lengths[frame,i]=count[0]
        if count[0]>neighbors.shape[2]:cuda.atomic.max(bad,0,2)
