# Provider optimization: three isolated trials (2026-11-04)

Parent baseline: `46a2c82c`, decoded-chunk LRU. Reader unchanged; main untouched.

## Experiment

Each experiment has its own Provider process. Environment variable
`MINITOPOL_PROVIDER_OPTIMIZATION` selects one of:
- `lru_base`: reference implementation from the parent.
- `single_gather`: reduce intermediate NumPy array allocation by collapsing the two-step frame/atom selection into one broadcast gather.
- `wide_cache`: decoded-chunk LRU allowance changed from `min(64 MiB, budget / 5)` to `min(256 MiB, budget / 3)`. Shared-memory cache retains the remainder of the same startup 30%-of-available-RAM ceiling.
- `parallel_miss`: build distinct shared-cache misses outside the global cache lock, using a separate lock for decoded-chunk LRU mutation. Duplicate concurrent misses may reconstruct the same key, but only one result is published.

Real MiniTopol AGX3 serialization and separate `MiniTopol-reader` client.
Ubuntu GitHub Actions runner; Python 3.12.15; 4 logical CPUs; ~15.63 GB available RAM.
Synthetic random-walk input, **2,304 frames × 4,800 atoms (11,059,200 AF)**.
Actual physical position chunk 64 frames. Decoded positions are ~126.6 MiB,
larger than the reference 64 MiB cap but smaller than the experimental 256 MiB cap.
Every mode is tested **twice**, in forward/reverse order.
Results report AF **requested**, not all atoms reconstructed from source chunks.

## Results

All numbers are **AF/min**, derived from the median of elapsed times in two runs.

| Mode | Small random 16×24 | 4 concurrent random 64-frame calls | Random 256 | Sequential 256 | Random 256 + 3 synthetic descriptors |
|---|---:|---:|---:|---:|---:|
| lru_base | 45,459,142 | 52,535,571 | 210,034,527 | 468,557,556 | 132,401,190 |
| single_gather | 45,981,121 | 48,319,709 | 214,530,758 | 536,589,981 | 139,328,153 |
| **wide_cache** | **334,130,955** | **398,467,267** | 214,262,050 | 526,809,192 | **751,046,686** |
| parallel_miss | 47,143,117 | 150,354,841 | 212,172,402 | 520,700,030 | 168,985,031 |

Relative to `lru_base`:
- `single_gather`: 1.01× small random; 0.92× concurrent.
- `wide_cache`: 7.35× small random; 7.58× concurrent; 5.67× synthetic descriptor mixed.
- `parallel_miss`: 1.04× small random; 2.86× concurrent; 1.28× synthetic descriptor mixed.

Position, time and PBC box SHA-256 hashes matched across all four
strategies for a non-monotonic frame/atom selection. The benchmark also
consumed the returned shared-memory coordinates and released leases.
Provider regression tests: 5 passed before the benchmark.

## Interpretation / limits

**Wide cache wins this specific enlarged-working-set benchmark.** Its cap
can contain the full decoded coordinate working set, whereas the 64 MiB
reference cache cannot. This is not evidence of the same 7.35× improvement
for a trajectory much larger than 256 MiB. It does not imply 50 billion
AF/min is achieved.

Parallel unlocked misses improve 4-worker concurrent throughput, but
they remain subject to the 64 MiB decoded cache limit. Single gather
has modest gains on contiguous reads but no meaningful improvement in
the key small-random workload.

Provider startup and data conversion are excluded from timing.
OS disk page cache was not flushed. Synthetic descriptor tables represent
RMSD, gyration and RMSF, **not all scientific descriptors**.
Tests are specific to this AGX3 synthetic dataset and CI runner.
These modes are **individual experiments**, not a combined candidate.

- [Successful benchmark and raw logs](https://github.com/Nagai121/MiniTopol/actions/runs/41838921845)
- [Experiment branch](https://github.com/Nagai121/MiniTopol/tree/experiment/provider-three-optimizations)
