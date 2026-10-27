"""Optional guarded exact GPU batches; all unsupported inputs use original CPU."""
import os
import threading
import numpy as np
from .kernels import sphere_points
_points=None
_point_context=None
_buffers=None
_lock=threading.RLock()

def evaluate_batch(xyz,radii,budget,*,available=None):
    if os.environ.get('MINITOPOL_SASA_DEVICE','auto') == 'cpu':
        return None
    with _lock:
        return _evaluate(xyz,radii,budget,available=available)

def evaluate_counts(xyz,radii,budget,*,available=None):
    if os.environ.get('MINITOPOL_SASA_DEVICE','auto')=='cpu':return None
    with _lock:return _evaluate(xyz,radii,budget,available=available,counts_only=True)

def _evaluate(xyz,radii,budget,*,available=None,counts_only=False):
    global _points,_point_context,_buffers
    if (type(xyz) is not np.ndarray or xyz.dtype!=np.dtype('f8') or xyz.ndim!=3
            or not 1<=len(xyz)<=32 or xyz.shape[2]!=3 or not xyz.flags.c_contiguous
            or np.geterr()!=dict(divide='warn',over='warn',under='ignore',invalid='warn')
            or not np.isfinite(xyz).all() or np.max(np.abs(xyz),initial=0)>1e6
            or type(radii) is not np.ndarray or radii.dtype!=np.dtype('f8')
            or radii.shape!=(xyz.shape[1],) or not np.isfinite(radii).all()
            or np.any(radii<.1) or np.any(radii+1.4>10)):
        return None
    if available is None:available=budget//2
    b,m=xyz.shape[:2]
    if b*m<16:
        return None
    required=xyz.nbytes+radii.nbytes+960*3*8+b*m*128*4+b*m*(8+4)+4
    compatible=_buffers is not None and _buffers['shape'][1]==m and _buffers['shape'][0]>=b
    reserved_required=max(required,_buffers['bytes']+960*3*8) if compatible else required
    cached_extra=_buffers['bytes'] if _buffers is not None and not compatible else 0
    if not 1<=m<=2048 or 2*reserved_required+cached_extra>min(budget//2,available):return None
    if np.any((np.abs(xyz)>0)&(np.abs(xyz)<1e-20)):return None
    from ..acceleration import configure_cuda
    configure_cuda()
    try:from numba import cuda
    except ImportError:return None
    if not cuda.is_available():return None
    context=cuda.current_context()
    if _buffers is not None and _buffers['context'] is not context:_buffers=None
    free,_=context.get_memory_info()
    if free<256*1024**2:return None
    device=cuda.get_current_device()
    if device.MAX_THREADS_PER_BLOCK<512:return None
    resident_blocks=device.MULTIPROCESSOR_COUNT*(device.MAX_THREADS_PER_MULTI_PROCESSOR//128)
    shared_bound=min(b*m,resident_blocks)*128*72
    if 2*reserved_required+shared_bound+cached_extra>min(budget//2,available):return None
    from .gpu_fused import fused_counts
    from .gpu_sort import sort_nearest
    from .direct_neighbors import direct_neighbors
    inflated=radii+1.4;points=sphere_points(960)
    if _points is None or _point_context is not context:
        _points=cuda.to_device(points)
        _point_context=context
    key=inflated.tobytes()
    reuse=_buffers is not None and _buffers['context'] is context and _buffers['shape'][1]==m and _buffers['shape'][0]>=b
    if reuse:
        dx=_buffers['xyz'][:b];dr=_buffers['radii']
        lengths=_buffers['lengths'][:b];neighbors=_buffers['neighbors'][:b]
        bad=_buffers['bad'];counts=_buffers['counts'][:b]
        dx.copy_to_device(xyz)
        if _buffers['radii_bytes']!=key:
            dr.copy_to_device(inflated);_buffers['radii_bytes']=key
        bad.copy_to_device(np.zeros(1,dtype=np.int32))
    else:
        dx=cuda.to_device(xyz);dr=cuda.to_device(inflated)
        lengths=cuda.device_array((b,m),dtype=np.int32)
        neighbors=cuda.device_array((b,m,128),dtype=np.int32);bad=cuda.to_device(np.zeros(1,dtype=np.int32))
        counts=cuda.device_array((b,m),dtype=np.int64)
        if _buffers is None:
            _buffers=dict(context=context,shape=(b,m),xyz=dx,radii=dr,lengths=lengths,neighbors=neighbors,bad=bad,counts=counts,radii_bytes=key,bytes=sum(a.nbytes for a in (dx,dr,lengths,neighbors,bad,counts)))
    margin=1e-6*max(1.,float(np.max(np.abs(xyz),initial=0)))
    direct_neighbors[b*m,128](dx,dr,float(inflated.max()),margin,lengths,neighbors,bad)
    if bad.copy_to_host()[0]:return None
    sort_nearest[b*m,128](dx,lengths,neighbors)
    fused_counts[b*m,512](dx,dr,_points,lengths,neighbors,counts)
    accessible=counts.copy_to_host();cuda.synchronize()
    if counts_only:
        if accessible.min(initial=0)<0 or accessible.max(initial=0)>960:raise IOError('GPU accessibility count bound')
        return accessible.astype('<u2')
    unique,inverse=np.unique(inflated,return_inverse=True);factors=np.asarray([4*np.pi*r**2 for r in unique])[inverse]
    return accessible/960*factors
