# Three Provider optimization variants (experimental)

Parent: 46a2c82c (decoded-chunk LRU). No Reader changes, no main changes.

Modes (set MINITOPOL_PROVIDER_OPTIMIZATION before starting Provider):
- lru_base (default): unchanged selected decoded-chunk LRU code path.
- single_gather: one NumPy broadcast gather, reducing intermediate allocations.
- wide_cache: decoded cache up to 256 MiB or 1/3 of startup budget instead of 64 MiB or 1/5.
- parallel_miss: build distinct cache misses outside global shared-cache lock, protect decoded LRU with separate lock, with duplicate concurrent misses permitted.

Every mode shares the existing startup RAM budget for decoded and shared-memory cache. 
Correctness check hashes coordinate, time and periodic-box bytes over reordered frames/atoms against baseline.

Benchmark: AGX3 2304 frames x 4800 atoms (larger than baseline 64MiB decoded cap);
sequential 256, random 256, 16 x 24 small random, four simultaneous 64-frame Reader requests,
and 256-frame synthetic-three-descriptor request. Two rounds in reversed strategy order.
Includes Reader IPC, SHM consumption/release; excludes conversion and Provider startup.
OS page cache is not cleared, so this is not a cold-storage benchmark.
