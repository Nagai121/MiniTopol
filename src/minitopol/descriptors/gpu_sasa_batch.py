"""Optional guarded exact GPU batches; all unsupported inputs use original CPU."""
import os
import threading
import numpy as np
from .kernels import sphere_points
_points=None
_point_context=None
_lock=threading.RLock()

def evaluate_batch(xyz,radii,budget,*,available=None):
    if os.environ.get('MINITOPOL_SASA_DEVICE','auto') == 'cpu':
        return None
    with _lock:
        return _evaluate(xyz,radii,budget,available=available)

def _evaluate(xyz,radii,budget,*,available=None):
    global _points,_point_context
    if (type(xyz) is not np.ndarray or xyz.dtype!=np.dtype('f8') or xyz.ndim!=3
            or not 1<=len(xyz)<=16 or xyz.shape[2]!=3 or not xyz.flags.c_contiguous
            or np.geterr()!=dict(divide='warn',over='warn',under='ignore',invalid='warn')
            or not np.isfinite(xyz).all() or np.max(np.abs(xyz),initial=0)>1e6
            or type(radii) is not np.ndarray or radii.dtype!=np.dtype('f8')
            or radii.shape!=(xyz.shape[1],) or not np.isfinite(radii).all()
            or np.any(radii<.1) or np.any(radii+1.4>10)):
        return None
    if available is None:available=budget//2
    b,m=xyz.shape[:2]
    if b*m<128:
        return None
    required=xyz.nbytes+radii.nbytes+960*3*8+b*m*128*4+b*m*(8+4)+4
    if not 1<=m<=2048 or 2*required>min(budget//2,available):return None
    if np.any((np.abs(xyz)>0)&(np.abs(xyz)<1e-20)):return None
    from ..acceleration import configure_cuda
    configure_cuda()
    try:from numba import cuda
    except ImportError:return None
    if not cuda.is_available():return None
    context=cuda.current_context()
    free,_=context.get_memory_info()
    if free<256*1024**2:return None
    device=cuda.get_current_device()
    resident_blocks=device.MULTIPROCESSOR_COUNT*(device.MAX_THREADS_PER_MULTI_PROCESSOR//128)
    shared_bound=min(b*m,resident_blocks)*128*12
    if 2*required+shared_bound>min(budget//2,available):return None
    from .gpu_fused import fused_counts
    from .gpu_sort import sort_nearest
    from .direct_neighbors import direct_neighbors
    inflated=radii+1.4;points=sphere_points(960)
    if _points is None or _point_context is not context:
        _points=cuda.to_device(points)
        _point_context=context
    dx=cuda.to_device(xyz);dr=cuda.to_device(inflated)
    lengths=cuda.device_array((b,m),dtype=np.int32)
    neighbors=cuda.device_array((b,m,128),dtype=np.int32);bad=cuda.to_device(np.zeros(1,dtype=np.int32))
    margin=1e-6*max(1.,float(np.max(np.abs(xyz),initial=0)))
    direct_neighbors[b*m,128](dx,dr,float(inflated.max()),margin,lengths,neighbors,bad)
    if bad.copy_to_host()[0]:return None
    sort_nearest[b*m,128](dx,lengths,neighbors)
    counts=cuda.device_array((b,m),dtype=np.int64)
    fused_counts[b*m,128](dx,dr,_points,lengths,neighbors,counts)
    accessible=counts.copy_to_host();cuda.synchronize()
    unique,inverse=np.unique(inflated,return_inverse=True);factors=np.asarray([4*np.pi*r**2 for r in unique])[inverse]
    return accessible/960*factors
