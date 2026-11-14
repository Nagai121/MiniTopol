"""Provider-owned CUDA leases for already-published shared-memory buffers.

Reader sends a RAM lease token and receives only an opaque VRAM lease token.
All native host-to-device transfers, CUDA contexts, allocation ownership and
release live in Provider. No per-atom or per-frame Python copy loops are used.
"""
from __future__ import annotations

import secrets
from threading import RLock

import numpy as np


class VramLeases:
    def __init__(self, cache):
        self.cache = cache
        self._lock = RLock()
        self._leases = {}
        self._bytes = 0

    @staticmethod
    def _cuda():
        from ..acceleration import configure_cuda
        configure_cuda()
        try:
            from numba import cuda
        except ImportError as error:
            raise RuntimeError("CUDA support unavailable; install minitopol[gpu]") from error
        if not cuda.is_available():
            raise RuntimeError("No usable CUDA GPU is available to Provider")
        return cuda

    @staticmethod
    def _transfer(cuda, segment, meta):
        """One native CUDA copy per requested buffer, never per atom/frame."""
        view = memoryview(segment.buf)[:meta["bytes"]]
        try:
            if meta["format"] == "ndarray":
                host = np.ndarray(meta["shape"], dtype=np.dtype(meta["dtype"]), buffer=view)
            elif meta["format"] == "arrow-ipc":
                # Preserve the Arrow IPC bytes; no invented GPU table semantics.
                host = np.frombuffer(view, dtype=np.uint8)
            else:
                raise ValueError("Unsupported VRAM buffer format: " + meta["format"])
            try:
                return cuda.to_device(host)
            finally:
                del host
        finally:
            view.release()

    def upload(self, request):
        if not isinstance(request, dict) or set(request) - {"lease", "buffers", "device"}:
            raise ValueError("Invalid VRAM upload request")
        source = request.get("lease")
        if not isinstance(source, str) or not source:
            raise ValueError("A live RAM lease is required")
        device = request.get("device", 0)
        if type(device) is not int or device < 0:
            raise ValueError("CUDA device index must be a nonnegative integer")
        requested = request.get("buffers")
        if requested is not None and (not isinstance(requested, list) or not requested
                                      or any(not isinstance(n, str) or not n for n in requested)
                                      or len(set(requested)) != len(requested)):
            raise ValueError("buffers must be a nonempty list of unique names")

        # A temporary cache lease pins the source while GPU transfer is in
        # progress, even if the requesting Reader releases its own RAM lease.
        with self._lock:
            pin, entry = self.cache.pin(source)
            try:
                parts = dict(zip((m["name"] for m in entry["metadata"]),
                                 zip(entry["segments"], entry["metadata"])))
                names = list(parts) if requested is None else requested
                if set(names) - parts.keys():
                    raise ValueError("Unknown RAM buffers: " + ", ".join(sorted(set(names) - parts.keys())))
                selected = [parts[name] for name in names]
                size = sum(meta["bytes"] for _, meta in selected)
                cuda = self._cuda()
                if device >= len(cuda.gpus):
                    raise ValueError("CUDA device index is out of range")
                with cuda.gpus[device]:
                    free, _total = cuda.current_context().get_memory_info()
                    if size > free:
                        raise MemoryError("Insufficient available VRAM for selected buffers")
                    arrays = {
                        meta["name"]: self._transfer(cuda, segment, meta)
                        for segment, meta in selected
                    }
                    # Synchronize before releasing the RAM pin and returning.
                    cuda.synchronize()
                token = secrets.token_urlsafe(24)
                details = [
                    {k: v for k, v in meta.items()
                     if k in ("name", "bytes", "format", "dtype", "shape", "unit", "rows", "scope")}
                    for _, meta in selected
                ]
                self._leases[token] = {"device": device, "arrays": arrays,
                                       "buffers": details, "bytes": size}
                self._bytes += size
                return {"vram_lease": token, "device": device,
                        "allocated_bytes": size, "buffers": details}
            finally:
                self.cache.release(pin)

    def release(self, request):
        if not isinstance(request, dict) or set(request) != {"vram_lease"}:
            raise ValueError("Expected a VRAM lease token")
        token = request["vram_lease"]
        if not isinstance(token, str):
            raise ValueError("VRAM lease token must be a string")
        with self._lock:
            entry = self._leases.pop(token, None)
            if entry is None:
                raise KeyError("Unknown or already released VRAM lease")
            self._bytes -= entry["bytes"]
            # Drop the Provider-owned CUDA array references; Numba manages
            # final device deallocation under the originating CUDA context.
            del entry
            return {"released": True, **self.status()}

    def status(self):
        with self._lock:
            return {"active_leases": len(self._leases),
                    "allocated_bytes": self._bytes}

    def close(self):
        with self._lock:
            self._leases.clear()
            self._bytes = 0
