"""Bounded subprocess execution of independent trajectory pipelines."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from filelock import FileLock


def _validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a nonempty list")
    normalized, destinations = [], set()
    allowed = {"workspace", "gro", "xtc", "solvent_resnames", "trajectory_id",
               "atom_elements", "annotation", "tpr", "source_to_tpr",
               "analysis", "estimated_work"}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) - allowed:
            raise ValueError("Unknown pipeline job fields")
        for field in ("workspace", "gro", "xtc", "solvent_resnames", "analysis"):
            if field not in job:
                raise ValueError(f"Missing pipeline field: {field}")
        if not isinstance(job["analysis"], dict) or "coordinate_mode" not in job["analysis"]:
            raise ValueError("Each job must explicitly specify analysis.coordinate_mode")
        if not isinstance(job["solvent_resnames"], list) or not all(isinstance(x, str) for x in job["solvent_resnames"]):
            raise ValueError("solvent_resnames must be an explicit list of names")
        if ("tpr" in job) != ("source_to_tpr" in job):
            raise ValueError("TPR bonds require an explicit source_to_tpr mapping")
        value = dict(job)
        for field in ("workspace", "gro", "xtc", "tpr"):
            if field in value:
                value[field] = str(Path(value[field]).resolve())
        destination = os.path.normcase(value["workspace"])
        if destination in destinations:
            raise ValueError("Parallel jobs require distinct Workspaces")
        destinations.add(destination)
        for field in ("gro", "xtc", "tpr"):
            if field in value and not Path(value[field]).is_file():
                raise FileNotFoundError(value[field])
        weight = value.get("estimated_work", 1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("estimated_work must be positive and finite")
        value["analysis"] = {"memory_mb": 128, **value["analysis"]}
        normalized.append((index, value, weight))
    ordered=sorted(normalized,key=lambda item:-item[2])
    # New combination: two small starters can complete independent CPU work
    # while GPU-bound jobs hold permits; keep remainder descending work.
    if len(ordered)>6:
        small=sorted(ordered,key=lambda item:(item[2],item[0]))[:2]
        chosen={item[0] for item in small}
        large=[item for item in ordered if item[0] not in chosen]
        return small+large[:4]+large[4:]
    return ordered


def _pipeline(job):
    from .converter import convert
    from .descriptors import analyze
    from .registry import Registry
    from .workspace import Workspace

    workspace = Workspace(job["workspace"])
    trajectory = convert(workspace, job["gro"], job["xtc"],
                         solvent_resnames=job["solvent_resnames"],
                         trajectory_id=job.get("trajectory_id"),
                         atom_elements=job.get("atom_elements"))
    if "annotation" in job:
        Registry(workspace).annotate(trajectory, **job["annotation"])
    edges = None
    if "tpr" in job:
        from .descriptors.bond_reconstruction import reconstruct_bonds
        edges = reconstruct_bonds(workspace, trajectory, job["tpr"], job["source_to_tpr"])
    report = analyze(workspace, trajectory, **job["analysis"])
    return {"status": report["status"], "workspace": str(workspace.path),
            "trajectory": trajectory, "bond_edges": edges, "analysis": report}


def _worker(path):
    from .storage import atomic_json, read_json

    request = read_json(path)
    try:
        report = _pipeline(request["job"])
    except Exception as error:
        atomic_json(path.parent / "result.json", {
            "status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    atomic_json(path.parent / "result.json", report)
    return 0 if report["status"] == "complete" else 1


def _stop(processes):
    """Stop only subprocess trees owned by this invocation."""
    import psutil

    owned = {}
    for child in processes:
        try:
            root = psutil.Process(child.pid)
            for process in [root, *root.children(recursive=True)]:
                owned[process.pid] = process
        except psutil.Error:
            pass
    for process in reversed(list(owned.values())):
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(owned.values()), timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    for child in processes:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run_pipelines(jobs, *, workers=6, native_threads=2, gpu_workers=2,
                  rss_limit_mb=4096):
    """Convert, annotate, restore optional TPR bonds and analyze each job.

    Jobs must use separate Workspaces. More than six jobs start the two smallest
    estimates first, followed by the remaining jobs in descending estimated_work.
    Otherwise jobs use descending estimated_work, preserving ties in input order.
    At most gpu_workers SASA stages hold GPU permits simultaneously. Each worker
    releases its CUDA context before returning its permit and continuing CPU work.
    The function is synchronous. Failures retain partial data for normal resume.
    """
    for name, value in (("workers", workers), ("native_threads", native_threads), ("gpu_workers", gpu_workers)):
        if type(value) is not int or value < (0 if name == "gpu_workers" else 1):
            raise ValueError(f"Invalid {name}")
    if gpu_workers > workers:
        raise ValueError("gpu_workers cannot exceed workers")
    if isinstance(rss_limit_mb, bool) or not isinstance(rss_limit_mb, (int, float)) or not math.isfinite(rss_limit_mb) or rss_limit_mb <= 0:
        raise ValueError("rss_limit_mb must be positive and finite")
    pending = _validate_jobs(jobs)
    try:
        import psutil
    except ImportError as error:
        raise ImportError('Parallel execution requires minitopol[parallel]') from error
    from .storage import atomic_json, read_json

    locks, active, streams = [], [], []
    reports = [None] * len(pending)
    gpu_jobs = {index for index, _, _ in pending[:gpu_workers]}
    peak = 0
    lease_directory = Path(tempfile.mkdtemp(prefix="gpu-stages-", dir=Path(pending[0][1]["workspace"]).parent)) if gpu_workers else None
    try:
        # Sorted acquisition is deterministic, and timeout=0 never waits while
        # holding another job's lock. No two invocations share a destination.
        for directory in sorted(job["workspace"] for _, job, _ in pending):
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(root / ".pipeline.lock"))
            lock.acquire(timeout=0)
            locks.append(lock)
        failed = False
        while pending or active:
            while pending and len(active) < workers and not failed:
                index, job, _ = pending.pop(0)
                control = Path(job["workspace"]) / ".pipeline"
                control.mkdir(exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix="run-", dir=control))
                path = directory / "job.json"
                atomic_json(path, {"job": job})
                log = (directory / "worker.log").open("w", encoding="utf-8")
                streams.append(log)
                env = dict(os.environ, NUMBA_NUM_THREADS=str(native_threads),
                           OMP_WAIT_POLICY="PASSIVE", PYTHONUNBUFFERED="1",
                           PYTHONPATH=str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                           MINITOPOL_GPU_STAGE_DIR=str(lease_directory) if lease_directory else "",
                           MINITOPOL_GPU_STAGE_SLOTS=str(gpu_workers),
                           MINITOPOL_SASA_DEVICE=("auto" if gpu_workers and os.environ.get("MINITOPOL_SASA_DEVICE") != "cpu" else "cpu"))
                child = subprocess.Popen([sys.executable, "-m", "minitopol.batch", "--worker", str(path)],
                                         env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                active.append((child, index, directory))
            rss = psutil.Process().memory_info().rss
            owned = {}
            for child, _, _ in active:
                try:
                    root = psutil.Process(child.pid)
                    for process in [root, *root.children(recursive=True)]:
                        owned[process.pid] = process
                except psutil.NoSuchProcess:
                    pass
            for process in owned.values():
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            if rss > rss_limit_mb * 1024**2:
                raise MemoryError("Pipeline RSS limit exceeded; partial Workspaces retained")
            for item in list(active):
                child, index, directory = item
                code = child.poll()
                if code is None:
                    continue
                result = directory / "result.json"
                reports[index] = read_json(result) if result.exists() else {"status": "failed", "error": f"Worker exited with code {code}"}
                reports[index]["log"] = str(directory / "worker.log")
                if code != 0 or reports[index]["status"] != "complete":
                    failed = True
                active.remove(item)
            if failed:
                for index, job, _ in pending:
                    reports[index] = {"status": "not_started", "workspace": job["workspace"]}
                pending.clear()
            if active:
                time.sleep(0.25)
        return {"status": "complete" if all(r["status"] == "complete" for r in reports) else "incomplete",
                "results": reports, "peak_rss_bytes": peak}
    finally:
        _stop([child for child, _, _ in active])
        for stream in streams:
            stream.close()
        for lock in reversed(locks):
            lock.release()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Use minitopol run for pipeline execution")
    raise SystemExit(_worker(Path(sys.argv[2])))
