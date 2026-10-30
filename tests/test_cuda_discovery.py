from pathlib import Path

from minitopol import acceleration


def test_explicit_toolkit_precedes_optional_packages(tmp_path, monkeypatch):
    root = tmp_path / "toolkit"
    for path in (root / "bin", root / "nvvm" / "bin"):
        path.mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(root))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(acceleration, "_configured", False)
    monkeypatch.setattr(acceleration, "_dll_directories", [])
    packages = []
    monkeypatch.setattr(acceleration, "_package_directory", lambda name: packages.append(name))
    acceleration.configure_cuda()
    assert packages == ["nvidia.cuda_runtime"]
    assert acceleration._configured


def test_nvidia_packages_are_discovered_without_copying_libraries(tmp_path, monkeypatch):
    nvcc, runtime = tmp_path / "nvcc", tmp_path / "runtime"
    for path in (nvcc / "bin", nvcc / "nvvm" / "bin", runtime / "bin"):
        path.mkdir(parents=True)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(acceleration, "_configured", False)
    monkeypatch.setattr(acceleration, "_dll_directories", [])
    monkeypatch.setattr(acceleration, "_package_directory", lambda name: nvcc if name.endswith("nvcc") else runtime)
    acceleration.configure_cuda()
    import os
    assert Path(os.environ["CUDA_HOME"]) == nvcc
    assert list(nvcc.rglob("*.dll")) == []


def test_missing_optional_packages_do_not_set_a_toolkit(monkeypatch):
    import os
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(acceleration, "_configured", False)
    monkeypatch.setattr(acceleration, "_package_directory", lambda name: None)
    acceleration.configure_cuda()
    assert "CUDA_HOME" not in os.environ
