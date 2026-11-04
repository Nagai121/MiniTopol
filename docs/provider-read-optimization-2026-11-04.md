# MiniTopol Provider: five-strategy optimization, 2026-11-04

Measured with the real MiniTopol Reader over local Provider IPC and shared memory. The test source was 1,152 frames × 4,800 atoms, 5,529,600 atom-frames, AGX3 compressed A Zarr, atomwise random-walk coordinates. Ubuntu GitHub Actions (4 logical CPUs, Python 3.12, ~15.3 GB available RAM). Each strategy ran in its own Provider process twice, with order reversed on the second pass. Values below are **median aggregate AF/min** from the successful final run.

- Successful run and full raw records: https://github.com/Nagai121/MiniTopol/actions/runs/41830401329
- Six-variant implementation and reproducible benchmark: https://github.com/Nagai121/MiniTopol/tree/experiment/provider-five-strategies

| Strategy | 256 random frames (AF/min) | 16×24 random-frame requests (AF/min) | 256 random frames + 3 synthetic descriptor tables (AF/min) |
|---|---:|---:|---:|
| Existing baseline | 153,529,313 | 22,416,917 | 150,415,884 |
| Reuse one Zarr group open per request | 172,859,420 | 23,748,818 | 160,290,805 |
| Vectorized copy of per-chunk selections | 184,489,269 | 23,665,899 | 158,663,344 |
| Zarr orthogonal indexing | 374,763,638 | 56,493,752 | 318,768,645 |
| Parallel per-chunk loading (4 workers) | 321,004,532 | 46,469,297 | 293,502,016 |
| **Bounded decoded-chunk LRU** | **323,087,845** | **384,947,666** | **856,754,401** |

The selected decoded-chunk LRU achieved **17.17×** small-random-request throughput and **2.10×** single large random throughput versus baseline. The mixed synthetic descriptor request improved **5.70×**. Its 256-contiguous-frame result was 439,437,829 AF/min vs 371,905,329 baseline. Zarr orthogonal indexing won *individual* previously uncached random requests (374,763,638 AF/min), but was much slower on repeatedly changing small requests.

All six strategies produced byte-for-byte matching requested coordinates, timestamps and periodic boxes (SHA-256) under non-monotonic frame and atom selection. The benchmark includes Reader connection, shared-memory view, consuming the coordinates and explicitly releasing the lease, but excludes Provider process startup and source trajectory conversion.

The previous complete measurement on another GitHub runner was consistent in ranking: LRU 308,617,067 AF/min versus baseline 20,088,519 AF/min for the 16×24 case (15.36×). That execution finished measurements but reported a temporary-workspace cleanup race. The successful final run fixes that cleanup race and exits successfully.

**Interpretation bounds:** OS disk page cache was not flushed; the input is a synthetic but actual AGX3 conversion, not a biological production trajectory. Synthetic descriptor tables stand in for RMSD, gyration, RMSF; not all real descriptors are present. Results are specific to working-set size, cache cap (min(64 MiB, 20% of the overall Provider budget)), access order and GitHub runner; they are not a universal performance guarantee or proof of the 5-billion-AF/min target.

The candidate implementation retains the original Reader wire protocol and public Python methods. For decoded-chunk retention it reserves up to 64 MiB of the same startup budget, reducing shared-memory capacity by that amount. Both counted caches combined remain within the original 30% of startup available RAM. Active Reader leases protect shared-memory segments; the decoded-chunk LRU can evict any source chunk after it has been copied into the response buffer. A memory-pressure check drops retained decoded chunks.
