"""Golden-hash stability for the Provider/Registry read paths.

The project's absolute constraint is that read-path optimizations never
change output values: not within a tolerance, bit for bit. This test pins
the SHA-256 of every buffer returned for a fixed set of requests against a
small synthetic workspace, plus a full Registry scan and a direct
query_results call. The golden hashes were captured from main at
cc682c6 (the pre-optimization code) on CPython 3.12 / Linux; any future
change to the read paths that alters a single output byte fails here.

Regenerate (only with an intentional, reviewed value change) by running
with MINITOPOL_PRINT_GOLDEN=1 and pasting the printed JSON into GOLDEN.
"""
import hashlib
import json
import os
from multiprocessing import shared_memory

import pyarrow as pa
import pytest

from minitopol import Workspace, convert
from minitopol.descriptors import DescriptorDefinition, ResultStore
from minitopol.provider.tiered import TieredProvider
from minitopol.registry import Registry

GOLDEN = {
 "coords_range": {
  "box": "58a31ce0e36dbfd58cbd547f74b8719a9a0dc7673093c3be072eaed7bfffdc41",
  "coordinates": "a38ca9a182de2ab187c9775f8fc5c2439c56aee396fa79bead1bc558094bc177",
  "time": "328e5709b2f63fa4d1f71fcba6244712a65019c8d7605a98f044dd46916a8d52"
 },
 "coords_scattered": {
  "box": "0fb7d69146f87e81ad29822f6532013aed5ae0e0e89a4bfb3ec30cb289d9e992",
  "coordinates": "bdf1e3eb245c562f3be237068be006b2582c692933a98d8f6205514434f556d0",
  "time": "d702d0a412ae7ad9a0df5c8680aa7223868841ef425791f7b4fe32119d78d653"
 },
 "coords_subset": {
  "box": "1d51a256caa0509fbad9de826b585dc5eec27a09839d34303fbc204456eb5196",
  "coordinates": "71653eace9743784d25eb8cfa1622bec001d46f106eb747e43236104ff0dd0d1",
  "time": "6ee00f9f757f45d83732f70bb4c45ddba228a10383c9cf0979748348b6b854ab"
 },
 "desc_global": {
  "descriptor:rmsf": "ce255b6cd04bba96a7d8b6cc6a625c7c1d2504a2842d7e6522d435fd3cedc959"
 },
 "desc_sample_range": {
  "descriptor:sample": "7610f863425377b22a2e7d8dfeb147d6bd6c01afab6a953ef76f18953ee4d68a"
 },
 "desc_sample_scattered": {
  "descriptor:sample": "8b8aa736096eb52e6278f6f234280df7e9a58cff3ffae73f94e18e4583682e0f"
 },
 "mixed": {
  "box": "1d51a256caa0509fbad9de826b585dc5eec27a09839d34303fbc204456eb5196",
  "coordinates": "c788e5d5fe0831dea06d45f8fb57917a197bd7f7d973ece4d9fbfcb7512bad1d",
  "descriptor:contacts": "2545a970206ce3c42bbf24f0f38c18fde7155fd7f627387d43215323d88daed7",
  "descriptor:sample": "f87800848f9d1c2ff092f5e0c0646c71929f615ed53bd74ec703953c6e86d41e",
  "time": "8ad7cebeafce3e73cd4f274d9ad09dc4d8be192b1dbbc6ce7a3551b573e03fb2"
 },
 "query_results_sample": "7610f863425377b22a2e7d8dfeb147d6bd6c01afab6a953ef76f18953ee4d68a",
 "registry_blocks": "8aa3f11c083e9fe195d97215abe614a7eb9cca621c8ff4534e85bf5a1a17db7e"
}


@pytest.fixture
def stability_workspace(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "storage")
    tid = convert(workspace, trajectory[0], trajectory[1],
                  solvent_resnames=["SOL"], chunk_frames=3)
    store = ResultStore(workspace)
    basic = DescriptorDefinition("sample", "1", "A", "test", "unit")
    store.publish(tid, basic, 0, 9,
                  pa.table({"frame": pa.array([0, 2, 4, 8], type=pa.int64()),
                            "value": pa.array([10., 20., 30., 40.])}))
    global_desc = DescriptorDefinition("rmsf", "test", "A", "global", "angstrom")
    store.publish(tid, global_desc, 0, 9,
                  pa.table({"point": [0, 1], "value": [1.5, 2.5]}))
    mixed = DescriptorDefinition("contacts", "test", "A", "mixed", "unit")
    mixed.register(workspace)
    with store.dataset(tid, mixed, layout={"output_partitioning": "test"}) as writer:
        writer.append("frames-0-9",
                      pa.table({"level": ["state", "state", "frequency"],
                                "frame": pa.array([0, 8, None], type=pa.int64()),
                                "value": [1., 1., .25]}))
    return workspace, tid


def _shm_data(part):
    segment = shared_memory.SharedMemory(name=part["shm"], create=False)
    try:
        return bytes(segment.buf[:part["bytes"]])
    finally:
        segment.close()


def _response_hashes(provider, request):
    reply = provider.get(request)
    try:
        return {part["name"]: hashlib.sha256(_shm_data(part)).hexdigest()
                for part in reply["buffers"]}
    finally:
        provider.cache.release(reply["lease"])


def _capture(workspace, tid):
    provider = TieredProvider(workspace.path, budget=64 * 1024**2)
    try:
        requests = {
            "coords_range": {"trajectory": tid, "coordinates": True,
                             "descriptors": [], "frames": list(range(9))},
            "coords_scattered": {"trajectory": tid, "coordinates": True,
                                     "descriptors": [], "frames": [8, 0, 4]},
            "coords_subset": {"trajectory": tid, "coordinates": True,
                              "descriptors": [], "frames": [1, 3],
                              "atoms": [0, 1]},
            "desc_sample_range": {"trajectory": tid, "coordinates": False,
                                  "descriptors": ["sample"],
                                  "frames": list(range(9))},
            "desc_sample_scattered": {"trajectory": tid, "coordinates": False,
                                      "descriptors": ["sample"],
                                      "frames": [8, 2]},
            "desc_global": {"trajectory": tid, "coordinates": False,
                            "descriptors": ["rmsf"], "frames": [0, 1]},
            "mixed": {"trajectory": tid, "coordinates": True,
                      "descriptors": ["sample", "contacts"], "frames": [0, 8]},
        }
        captured = {name: _response_hashes(provider, request)
                    for name, request in requests.items()}
    finally:
        provider.close()
    registry_hash = hashlib.sha256()
    for block in Registry(workspace).blocks(tid):
        registry_hash.update(block.positions.tobytes())
        registry_hash.update(block.time.tobytes())
        registry_hash.update(block.box.tobytes())
    captured["registry_blocks"] = registry_hash.hexdigest()
    table = workspace.query_results(descriptor="sample", trajectory=tid,
                                    start=0, stop=9)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table.combine_chunks())
    captured["query_results_sample"] = hashlib.sha256(
        sink.getvalue().to_pybytes()).hexdigest()
    return captured


def test_read_outputs_match_golden_hashes(stability_workspace):
    workspace, tid = stability_workspace
    captured = _capture(workspace, tid)
    if os.environ.get("MINITOPOL_PRINT_GOLDEN"):
        print(json.dumps(captured, indent=1, sort_keys=True))
        return
    assert GOLDEN, "golden hashes not recorded yet"
    assert captured == GOLDEN
