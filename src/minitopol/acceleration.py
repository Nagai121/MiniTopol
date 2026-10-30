"""Process-local discovery of an optional CUDA toolkit."""

import importlib.util
import os
from pathlib import Path
import threading

_lock = threading.Lock()
_configured = False
_dll_directories = []


def _package_directory(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError):
        return None
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    return None


def configure_cuda():
    """Locate installed libraries without downloading or modifying a toolkit.

    Prefer the user's CUDA_HOME/CUDA_PATH, then NVIDIA's optional pip wheels.
    Windows DLL directory handles must live as long as the CUDA context.
    """
    global _configured
    with _lock:
        if _configured:
            return
        home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        root = Path(home) if home else _package_directory("nvidia.cuda_nvcc")
        directories = []
        if root:
            os.environ.setdefault("CUDA_HOME", str(root))
            directories.extend((root / "bin", root / "nvvm" / "bin"))
        runtime = _package_directory("nvidia.cuda_runtime")
        if runtime:
            directories.append(runtime / "bin")
        if os.name == "nt":
            for directory in directories:
                if directory.is_dir():
                    _dll_directories.append(os.add_dll_directory(str(directory)))
        _configured = True
