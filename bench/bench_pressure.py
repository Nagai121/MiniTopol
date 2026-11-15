import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "bench")
sys.argv.append("--raw")
import bench_read as B
import psutil
from collections import namedtuple
_fake = namedtuple("vm", "total available percent used free")
_real = psutil.virtual_memory
psutil.virtual_memory = lambda: _fake(_real().total, 128 * 1024**2, 97.0, 0, 128 * 1024**2)
from minitopol.provider.tiered import TieredProvider
tid = B.tid_of()
provider = TieredProvider(B.WS, B.BUDGET)
sc = B.scenarios(tid)
for rnd in range(3):
    time.sleep(1.2)
    for name, req in sc.items():
        if name == "repeat_hit_mixed64": continue
        before = (provider.part_bytes//2**20, provider.decoded_used//2**20, provider.cache.used//2**20, provider.pressure_events)
        t0 = time.perf_counter(); B.run_get(provider, req); dt = (time.perf_counter()-t0)*1000
        print(f"r{rnd} {name:22s} {dt:7.1f}ms  before parts/dec/resp MiB={before}")
time.sleep(1.2)
print("events:", provider.pressure_events)
provider.close()
