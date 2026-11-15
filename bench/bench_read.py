"""Read-path benchmark + output-hash capture for MiniTopol Provider/Registry.

Usage:
  python bench_read.py hashes   -> write bench/hashes.json with output hashes
  python bench_read.py bench    -> print median ms per scenario
"""
import hashlib, json, statistics, sys, time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minitopol import Workspace
from minitopol.registry import Registry
from minitopol.provider.tiered import TieredProvider

import os
WS = Path(__file__).resolve().parents[1] / os.environ.get("MT_BENCH_WS", "bench-ws") / "storage"
BUDGET = 512 * 1024**2

# This VM reports only ~0.5 GiB free RAM, which makes the Provider's
# memory-pressure path (available < 6 GiB) clear its caches on almost every
# request. That is an artifact of the measurement environment, not of a
# normal workstation. Unless --raw is given, report a roomy machine so the
# read paths themselves are measured. Applied identically to baseline and
# optimized code.
if "--raw" not in sys.argv:
    import psutil
    _vm = psutil.virtual_memory()
    psutil.virtual_memory = lambda: _vm._replace(
        total=64 * 1024**3, available=32 * 1024**3, free=32 * 1024**3)

def tid_of():
    ws = Workspace(WS)
    return ws.list_trajectories()[0][0]

def dims(tid):
    """(frames, atoms) of the fixture trajectory, from its Zarr store."""
    import zarr
    from minitopol.registry import Registry
    registry = Registry(Workspace(WS))
    manifest = registry.manifest(tid)
    group = zarr.open_group(str(registry._read_trajectory_path(tid) / "A.zarr"), mode="r")
    positions = group["positions"]
    return int(positions.shape[0]), int(positions.shape[1])

def shm_bytes(part):
    seg = shared_memory.SharedMemory(name=part["shm"], create=False)
    try:
        return bytes(seg.buf[:part["bytes"]])
    finally:
        seg.close()

def run_get(provider, request):
    reply = provider.get(request)
    try:
        return {p["name"]: shm_bytes(p) for p in reply["buffers"]}
    finally:
        provider.cache.release(reply["lease"])

def scenarios(tid):
    frames_total, atoms_total = dims(tid)
    rng = np.random.default_rng(123)
    n_range = max(1, frames_total // 4)
    n_small = max(1, frames_total // 8)
    rand_range = sorted(rng.choice(frames_total, size=n_range, replace=False).tolist())
    rand_small = sorted(rng.choice(frames_total, size=n_small, replace=False).tolist())
    return {
        "coords_range128": {"trajectory": tid, "coordinates": True, "descriptors": [], "frames": list(range(0, n_range))},
        "coords_random128": {"trajectory": tid, "coordinates": True, "descriptors": [], "frames": rand_range},
        "coords_subset64": {"trajectory": tid, "coordinates": True, "descriptors": [], "frames": list(range(0, n_small)), "atoms": list(range(0, atoms_total, 6))},
        "desc_range128": {"trajectory": tid, "coordinates": False, "descriptors": ["pairdist"], "frames": list(range(0, n_range))},
        "desc_scattered64": {"trajectory": tid, "coordinates": False, "descriptors": ["pairdist"], "frames": rand_small},
        "desc_global": {"trajectory": tid, "coordinates": False, "descriptors": ["rmsf"], "frames": list(range(0, n_small))},
        "desc_small_range": {"trajectory": tid, "coordinates": False, "descriptors": ["sample"], "frames": list(range(0, n_range))},
        "mixed64": {"trajectory": tid, "coordinates": True, "descriptors": ["sample", "pairdist"], "frames": list(range(0, n_small))},
    }

def rotated(request, k):
    """Shift frame window by k*7 so each rep misses the exact-response cache."""
    req = dict(request)
    frames = req["frames"]
    total = rotated.total
    span = max(frames) - min(frames)
    if len(frames) == span + 1:  # contiguous
        start = (min(frames) + k * 7) % (total - span)
        req["frames"] = list(range(start, start + span + 1))
    else:
        shift = (k * 7) % total
        req["frames"] = sorted({(f + shift) % total for f in frames})
    return req

def bench_one(fn, reps=9):
    times = []
    for k in range(reps):
        t0 = time.perf_counter()
        fn(k)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 3)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    tid = tid_of()
    provider = TieredProvider(WS, BUDGET)
    try:
        sc = scenarios(tid)
        rotated.total = dims(tid)[0]
        if mode == "hashes":
            out = {}
            for name, req in sc.items():
                bufs = run_get(provider, req)
                out[name] = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(bufs.items())}
            # Registry blocks hash
            reg = Registry(Workspace(WS))
            h = hashlib.sha256()
            for block in reg.blocks(tid):
                h.update(block.positions.tobytes()); h.update(block.time.tobytes()); h.update(block.box.tobytes())
            out["registry_blocks"] = h.hexdigest()
            ws = Workspace(WS)
            span = max(1, dims(tid)[0] // 512)
            t = ws.query_results(descriptor="pairdist", trajectory=tid, start=10 * span, stop=90 * span)
            out["query_results_rows"] = t.num_rows
            Path(__file__).with_name("hashes.json").write_text(json.dumps(out, indent=1))
            print("hashes written")
            return
        results = {}
        # warm-up
        for req in sc.values():
            run_get(provider, req)
        for name, req in sc.items():
            results[name] = bench_one(lambda k, r=req: run_get(provider, rotated(r, k)))
        # exact-repeat cache hit
        req = sc["mixed64"]
        run_get(provider, req)
        results["repeat_hit_mixed64"] = bench_one(lambda k: run_get(provider, req), reps=15)
        # Registry full scan
        reg = Registry(Workspace(WS))
        total_frames = dims(tid)[0]
        def scan(k):
            n = 0
            for block in reg.blocks(tid):
                n += block.positions.shape[0]
            assert n == total_frames
        results["registry_scan512"] = bench_one(scan, reps=5)
        # query_results direct
        ws = Workspace(WS)
        span = max(1, total_frames // 512)
        results["query_results_range80"] = bench_one(
            lambda k: ws.query_results(descriptor="pairdist", trajectory=tid, start=(k % 100) * span, stop=(k % 100) * span + 80 * span), reps=7)
        print(json.dumps(results, indent=1))
    finally:
        provider.cache.close()

if __name__ == "__main__":
    main()
