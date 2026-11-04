# MiniTopol Provider / Reader

Provider lives in src/minitopol/provider/ and is the only owner of MiniTopol storage interpretation, reconstruction and shared-cache lifecycle. Its separate Reader repository needs Python 3.11+ only; it neither imports MiniTopol nor interprets persisted codecs.

Register a Workspace from any directory in which minitopol is installed:

~~~bash
python -m minitopol.provider register --workspace /path/to/BaseTraj_Workspace
python -m minitopol.provider serve
~~~

The Reader connects using the user-local registry at ~/.minitopol/provider.json (override: MINITOPOL_PROVIDER_REGISTRY). If no Provider process is alive, the Reader starts the previously registered Python installation. The Provider locks its registration to prevent duplicate servers, listens only on 127.0.0.1 and authenticates each request with a random token stored in a user-private registry file.

API v1:
- GET /v1/status: current cache budget, use, hits/misses, leases
- GET /v1/list: completed trajectories and completed descriptor names
- POST /v1/get: trajectory, coordinates (bool), descriptors (list or "all"), frames (list) OR start/stop (exclusive), optional atoms (coordinates only)
- POST /v1/release: lease ID
- POST /v1/stop: refuses while active leases exist

Each get returns a lease, full requested frame order and distinct named shared-memory buffers. Coordinates, time and periodic box are independent little-endian ndarray buffers with dtype/shape/unit. Descriptors, including sparse or null-valued tables, are separate Arrow IPC streams. Descriptor rows with a frame field are frame-filtered; global descriptors remain trajectory-wide; contacts also retains aggregate frequency rows.

The Provider uses MiniTopol's Registry.blocks for Zarr coordinates and Workspace.query_results for completed logical descriptors (including SASA and Ramachandran). Requests are grouped by source chunk when reading positions. Repeat identical requests share immutable cached buffers. Distinct requests also reuse a bounded LRU of decoded source chunks, indexed by trajectory identity and source frame interval. Buffers are never evicted while leased; release makes them eligible for LRU reclamation. Both caches share the original startup RAM allowance. The startup cap is 30% of available RAM (not total installed RAM), allocated on demand; large requests must be subdivided. Memory pressure triggers reclaim of unleased entries.

Known implementation limits: shared-memory response reuse still keys complete requests, while independently requested frames reuse previously decoded source chunks; intermediate decode/Arrow allocations can exceed the shared-cache cap; the service currently serializes cache misses; record layout remains stable but capacity and peak throughput are unmeasured. These are intended targets for later performance profiling, not claimed benchmarks. No training code is required for these integration tests.

## Optional Provider phase profiling (append-only TXT)

Set `MINITOPOL_PROVIDER_PROFILE=1` in the **Provider process environment**
before starting it. Profiling is disabled by default, and Reader requests
do not need to change.

For each completed Provider `get()` request, the Provider appends exactly one
UTF-8 JSON line to `<workspace>/provider-profile.txt`. The file is created
on the first request, is **never truncated**, and retains all previous lines
even after Provider restarts. There is no in-memory record history, record
limit, profile retrieval API or `profile` CLI command.

To choose a different location, set `MINITOPOL_PROVIDER_PROFILE_PATH`
in the Provider environment before launch. For example, in PowerShell:

~~~powershell
$env:MINITOPOL_PROVIDER_PROFILE = "1"
$env:MINITOPOL_PROVIDER_PROFILE_PATH = "D:\\MiniTopol\\provider-profile.txt"
python -m minitopol.provider serve
~~~

The TXT can be inspected while Provider is running. Each line stores a
timestamp, abbreviated query information, status/error, total Provider elapsed
milliseconds, per-stage milliseconds, and cache counters. Concurrent requests
serialize only their append writes, so lines cannot interleave. Failed
`get()` requests are recorded too. If the TXT cannot be written, Provider
reports an error to stderr without changing a successful Reader response.

The measured stages are:
- `manifest_lookup`, `descriptor_catalog_lookup`: metadata/catalog;
- `zarr_open_metadata`, `chunk_grouping`: open Zarr and map frames;
- `zarr_fetch_and_decode`: storage fetch **and** codec decompression together;
- `coordinate_extract_copy`, `coordinate_serialization`: select/pack;
- `descriptor_query`, `descriptor_arrow_encode`: descriptor storage/query and Arrow;
- `shared_memory_allocate`, `shared_memory_copy`: shared-memory publication;
- `shared_cache_lock_wait`: waiting for the shared-cache lock;
- `other_provider_work`: remaining request overhead.

Cache hit/miss counters refer separately to decoded chunks and exact shared
response buffers. Source decode bytes, output shared-memory bytes, descriptor
row count, and requested atom-frame count are also recorded.

**Measurement boundary:** `total_ms` is sampled **before** JSON encoding and
TXT file I/O, so file recording is excluded from the reported Provider time.
Writing still adds real latency to the Reader-visible request. File access
plus AGX decompression are not separately timed. Reader transport/consumption
are not included. Use non-profile runs when benchmarking absolute end-to-end
throughput.
