# Read-path optimization loop log (local branch read-opt-exp, NOT for main)

Baseline = main cc682c6, TieredProvider (production), synthetic workspace:
1536 atoms, 512 frames, chunk_frames=64, descriptors sample (512 rows),
pairdist (131072 rows), rmsf (1536 rows). Median ms, this VM (1.5 vCPU).

| scenario | baseline |
|---|---|
| coords_range128 | 18.98 |
| coords_random128 | 18.96 |
| coords_subset64 | 8.84 |
| desc_range128 | 91.92 |
| desc_scattered64 | 170.55 |
| desc_global | 90.95 |
| desc_small_range | 53.21 |
| mixed64 | 118.59 |
| repeat_hit_mixed64 | 28.31 |
| registry_scan512 | 108.26 |
| query_results_range80 | 42.21 |

Profile findings (MINITOPOL_PROVIDER_PROFILE + cProfile):
- descriptor_catalog_lookup ~21 ms/request; _duckdb.connect ~9-15 ms per call,
  4+ connects per descriptor request (catalog lookup, _paths xN, query_results,
  global revision checks).
- descriptor_query: sample (512 rows) 24 ms, pairdist range 48 ms,
  pairdist scattered 122 ms.
- rmsf: other_provider_work 38 ms (revision/_paths/encoding-key catalog opens).
- coords: zarr_open_metadata ~5 ms, coordinate_extract_copy 6.4 ms
  (double fancy indexing, full-range atom list), fetch+decode 6.8 ms.
- Cache-hit request still 28 ms => fixed per-request overhead dominates.

## Iterations (each: change -> bench -> output-hash check; hashes matched every time)

1. Wrap Provider.get in Workspace.query_session (eager): descriptor requests
   improved (global 91->30 ms) but coordinate-only requests regressed
   (19->44 ms) because the session opened a DuckDB connection even when the
   request never queries the catalog.
2. Lazy query_session: connection+lock materialize on first catalog() call
   inside the session. Coordinate requests back to baseline; descriptor
   gains kept.
3. Generic descriptor table cache (QueryProvider._generic_descriptor):
   full published table cached in the existing parts cache, keyed by result
   IDs + file versions of parquet and sidecar; per-request selection by
   slice / per-frame ranges (frame column non-null + monotonic) or the same
   mask filter as before. desc_range 92->39, desc_scattered 171->41.
4. Catalog-row snapshot in DataProvider (stat-validated by mtime/size/inode
   of integrated.duckdb + WAL): runtime catalog lookup and _paths no longer
   open DuckDB connections. desc_global ->2.8, desc_small ->3.0,
   repeat_hit 28->3.8.
5. Zarr source cache: open group/arrays once per (path, trajectory identity)
   instead of per request. coords_range 19.5->13.9, subset 9.6->4.5.
6. Coordinate extract fast path: full in-order atom selection = plain frame
   take; other selections = np.take per axis (micro-bench: fancy indexing
   all atoms 2.2-3.9 ms vs 0.11 ms slice assignment per chunk).
   coords_range ->8.6, random ->10.1.
7. Registry.blocks: read time/box once for the whole range instead of one
   Zarr read per chunk. scan 110->100.
8. Registry.manifest stat-validated cache (returns a shallow copy).
   coords_range ->7.2, scan ->96.
9. Persistent decode ThreadPool in _parallel_coordinates (was created per
   request). Neutral on warm path, helps cold decode; kept.
10. Registry.blocks: windowed position reads (up to 4 blocks per Zarr call,
    per-block copy preserved). scan ->78-81.
11. Workspace.query_results bounded LRU (256 MB, stat-validated):
    identical repeated query 46->0.2 ms. Rotating-range pattern unchanged.

## Examined and rejected
- AGX3 zigzag vectorization with NumPy: slower than the njit loop
  (0.83 vs 0.20 ms/chunk); not applied.
- Blosc decode nthreads 1/2/4: no difference (0.78 ms); not applied.
- DuckDB connect read_only / extension-autoload off: no difference
  (~18.5 ms); not applied.
- Removing AGX3's two SHA-256 integrity checks (~1.5 ms of ~6.2 ms decode):
  rejected - they are the format's verification contract; precision and
  integrity are not negotiable.
- Sparse IN-predicate route for generic descriptors: subsumed by (3).

## Confound found (pre-existing, not caused by these changes)
AdvancedProvider._get clears parts/decoded/response caches whenever
psutil reports < 6 GiB available RAM, with target = available//5. On this
VM (~0.5 GiB free) it fired on most requests and made mixed workloads look
3-10x slower for BOTH baseline and optimized code. The harness therefore
reports a roomy machine (psutil patch in bench_read.py) for A/B numbers;
--raw reproduces the thrashing. On a workstation with >= 6 GiB free the
path does not fire. Flagging as a design smell worth a separate look.

## Final A/B (same harness, median of medians, ms)
| scenario | main cc682c6 | optimized | x |
|---|---|---|---|
| coords_range128 | 18.5 | 10.7 | 1.7 |
| coords_random128 | 18.2 | 7.9 | 2.3 |
| coords_subset64 | 7.8 | 3.1 | 2.5 |
| desc_range128 | 94 | 12.4 | 7.6 |
| desc_scattered64 | 188 | 10.5 | 18 |
| desc_global | 85 | 2.0 | 43 |
| desc_small_range | 50 | 2.4 | 21 |
| mixed64 | 119 | 11.9 | 10 |
| repeat_hit_mixed64 | 25.2 | 3.3 | 7.6 |
| registry_scan512 | 103 | 64 | 1.6 |
| query_results_range80 (rotating) | 45.8 | 48.3 | 1.0 |
Output SHA-256 of every scenario buffer + full Registry scan: identical
between baseline and optimized.

## Follow-up work (2026-11-15, branch muse-work)

### 12. Memory-pressure path: proportional eviction + cooldown
AdvancedProvider._get used to clear the part and decoded-chunk caches
outright on EVERY request while available RAM < 6 GiB (target =
available // 5), so under sustained pressure the caches could never warm
up. Now: the check runs at most once per second, and eviction is LRU and
proportional -- oldest part entries, then decoded chunks, then response
cache, only until the combined footprint fits the same target.
_parallel_coordinates had the same pattern (clear all decoded chunks
before a load when available < budget // 4); it now trims oldest chunks
to a bounded share of available RAM.
Measurement (bench/bench_pressure.py; psutil patched to 128 MiB
available, rotating distinct requests, 1.2 s between rounds):
original = 3 pressure events with repeated full wipes (parts 17->0 MiB,
decoded 9->0 MiB; mixed request in round 1 costs 55 ms re-decoding);
fixed = 1 event, decoded and response caches stay warm, later rounds
all requests <= 8 ms. Output hashes unchanged; provider test files pass.

### 13. Golden-hash stability test (tests/test_read_output_stability.py)
The hash check used in this loop is now a repository test: golden
SHA-256 values captured from pristine main cc682c6 for 7 Provider
requests + Registry scan + query_results over a small synthetic
workspace. It passes on pristine main and on this branch, so any future
read-path change that alters one output byte fails the suite.

### 14. Large-scale A/B (12288 atoms x 2048 frames, 32x atom-frames)
Fixture tooling is parameterized (MT_BENCH_ATOMS / MT_BENCH_FRAMES /
MT_BENCH_WS). Median ms, baseline = pristine cc682c6:
| scenario (scaled) | main | optimized | x |
|---|---|---|---|
| coords range 512f | 641 | 498 | 1.29 |
| coords random 512f | 1796 | 1748 | 1.03 |
| coords subset 256f | 40 | 36 | 1.11 |
| desc range 512f | 216 | 42 | 5.1 |
| desc scattered 256f | 611 | 37 | 16.7 |
| desc global | 89 | 2.8 | 32 |
| desc small range | 51 | 3.0 | 17 |
| mixed 256f | 384 | 177 | 2.2 |
| repeat hit mixed | 52 | 66 | 0.79 |
| registry scan 2048f | 1539 | 1473 | 1.04 |
| query_results range | 127 | 109 | 1.17 |
Output hashes identical at this scale too. Reading: descriptor wins
hold at scale; coordinate reads become AGX-decode-bound (one 64-frame
chunk = 9.4 MB at 12288 atoms), so connection/copy overhead no longer
matters there -- the remaining lever is the codec path itself, which is
precision-locked. The repeat-hit regression (0.79x) is noted as
unexplained; the response is ~40 MB of shared memory at this scale and
the difference may be lease/copy variance on this shared VM.
