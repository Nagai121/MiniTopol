# MiniTopol Provider: five-strategy optimization, 2026-11-04

Measured with the real MiniTopol Reader over local Provider IPC and shared memory.  The test source was 1,152 frames × 4,800 atoms, 5,529,600 atom-frames, AGX3 compressed A Zarr, atomwise random-walk coordinates.  Ubuntu GitHub Actions (4 logical CPUs, Python 3.12, ~15.6 GB available RAM).  Each strategy ran in its own Provider process twice, with order reversed on the second pass. Results below are **median aggregate AF/min**. Source data and code are in the experimental branch:

https://github.com/Nagai121/MiniTopol/tree/experiment/provider-five-strategies
https://github.com/Nagai121/MiniTopol/actions/runs/41829924372

| Strategy | 256 random frames (AF/min) | 16×24 random-frame requests (AF/min) | 256 random frames + 3 synthetic descriptor tables (AF/min) |
|---|---:|---:|---:|
| Existing baseline | 136,752,898 | 20,088,519 | 133,396,056 |
| Reuse one Zarr group open per request | 154,620,495 | 21,218,018 | 138,283,646 |
| Vectorized copy of per-chunk selections | 166,560,112 | 21,018,150 | 143,345,984 |
| Zarr orthogonal indexing | 313,955,271 | 49,479,178 | 277,705,375 |
| Parallel per-chunk loading (4 workers) | 272,316,284 | 40,014,067 | 240,464,177 |
| **Bounded decoded-chunk LRU** | **271,431,043** | **308,617,067** | **659,835,149** |

The selected decoded-chunk LRU achieved **15.36×** small-random-request throughput and **1.98×** single large random throughput versus baseline; mixed synthetic descriptor request throughput improved **4.95×**. Its 256-contiguous-frame result was 367,074,428 AF/min versus 324,217,359 baseline. Zarr orthogonal indexing won *individual* cold requests (313,955,271 AF/min), but lost repeatedly changing small requests by a factor of ~6.2 against decoded-chunk LRU.

All six strategies produced byte-for-byte matching requested coordinates, timestamps and periodic boxes (SHA-256) under non-monotonic frame and atom selection. The benchmark includes Reader connection, shared-memory view, consuming the coordinates and explicitly releasing the lease, but excludes Provider process startup and source trajectory conversion.

**Interpretation bounds:** OS disk page cache was not flushed; the input is a synthetic but actual AGX3 conversion, not a biological production trajectory. Synthetic descriptor tables stand in for RMSD, gyration, RMSF; not all real descriptors are present. Results are specific to working-set size, cache cap (min(64 MiB, 20% of the overall Provider budget)), access order and GitHub runner; they are not a universal performance guarantee or proof of the 5-billion-AF/min target.

The candidate implementation retains the original Reader wire protocol and public Python methods. For decoded-chunk retention it reserves up to 64 MiB of the same startup budget, reducing shared-memory capacity by that amount. Both counted caches combined remain within the original 30% of startup available RAM. Active Reader leases protect shared-memory segments; the decoded-chunk LRU can evict any source chunk after it has been copied into the response buffer. A memory-pressure check drops retained decoded chunks.
