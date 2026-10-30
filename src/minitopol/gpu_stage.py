"""Cross-process stage permits; OS locks release on hard exit."""
import os,time
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock,Timeout
@contextmanager
def lease(*,on_wait=None):
    directory=os.environ.get('MINITOPOL_GPU_STAGE_DIR','')
    count=int(os.environ.get('MINITOPOL_GPU_STAGE_SLOTS','0'))
    if not directory or count<1 or os.environ.get('MINITOPOL_SASA_DEVICE')=='cpu':
        yield
        return
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);held=None
    prepared=False
    while held is None:
        for slot in range(count):
            lock=FileLock(str(root/f'GPU-{slot}.lock'))
            try:lock.acquire(timeout=0)
            except Timeout:continue
            held=lock;break
        if held is None:
            if on_wait is not None and not prepared:
                prepared=True
                on_wait()
                continue
            time.sleep(.05)
    try:yield
    finally:
        # Free CUDA context before granting its stage permit to another worker.
        # CPU-only phases continue in the same pipeline without owning GPU RAM.
        try:
            from .descriptors import gpu_sasa_batch
            if gpu_sasa_batch._point_context is not None:
                from numba import cuda
                gpu_sasa_batch._buffers=None
                gpu_sasa_batch._points=None
                gpu_sasa_batch._point_context=None
                # Installed Numba caches CUfunctions by device ID, not context.
                # Drop only this stage's context-bound handles; keep compiled
                # binary/overload caches for safe loading into the next context.
                from .descriptors.gpu_fused import fused_counts
                from .descriptors.gpu_sort import sort_nearest
                from .descriptors.direct_neighbors import direct_neighbors
                device_id=cuda.current_context().device.id
                for dispatcher in (fused_counts,sort_nearest,direct_neighbors):
                    for kernel in list(dispatcher.overloads.values()):
                        kernel._codelibrary._cufunc_cache.pop(device_id,None)
                cuda.close()
        finally:held.release()
