"""AGX3 chunk decode breakdown (measurement only; no library changes).

Replicates position_moddelta.decode stage by stage on a real chunk
from the large bench workspace and times each stage:
header parse / payload SHA-256 / Blosc (zstd+bitshuffle) decode /
unpredict (zigzag inverse + cumulative sum) / raw SHA-256 verify,
plus the library decode end-to-end and the zarr codec wrapper
(includes the final tobytes copy).

Usage: python bench/agx_decode_breakdown.py [chunk-file ...]
Defaults to the first two chunks of the large bench trajectory.
"""
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from minitopol.position_moddelta import decode, _zigzag_inplace  # noqa: E402
from minitopol.agx_zarr import AGX3Numcodec  # noqa: E402
from numcodecs import Blosc  # noqa: E402
from numcodecs.blosc import _cbuffer_sizes  # noqa: E402


def median_ms(samples):
    return float(np.median(samples) * 1000)


def timeit(fn, repeats=30):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def stages(raw):
    size = struct.unpack("<Q", raw[8:16])[0]
    header = json.loads(raw[16:16 + size])
    payload = raw[16 + size:]
    n, m, _ = header["shape"]
    expected = n * m * 3 * 4

    def header_parse():
        s = struct.unpack("<Q", raw[8:16])[0]
        json.loads(raw[16:16 + s])

    def payload_sha():
        hashlib.sha256(payload).hexdigest()

    def blosc_decode():
        return bytes(Blosc().decode(payload))

    decoded = blosc_decode()
    _cbuffer_sizes(payload)

    def unpredict():
        packed = np.frombuffer(decoded, dtype="<u4").reshape(m, 3, n).copy()
        _zigzag_inplace(packed, True)
        return np.ascontiguousarray(
            np.add.accumulate(packed.transpose(2, 0, 1), axis=0, dtype="u4")).view("f4")

    result = unpredict()

    def raw_sha():
        hashlib.sha256(result.tobytes()).hexdigest()

    return header, expected, [
        ("header parse + contract", timeit(header_parse)),
        ("payload SHA-256 verify", timeit(payload_sha)),
        ("Blosc decode (zstd/bitshuffle)", timeit(blosc_decode)),
        ("unpredict (zigzag + cumsum)", timeit(unpredict)),
        ("raw SHA-256 verify", timeit(raw_sha)),
    ]


def main():
    args = sys.argv[1:]
    if not args:
        base = (Path(__file__).resolve().parent.parent
                / "bench-ws-large/storage/trajectories"
                / "2ee2be5d21bf9dde754030225dcc68e4/A.zarr/positions/c")
        args = [str(base / "0/0/0"), str(base / "1/0/0")]
    for name in args:
        raw = Path(name).read_bytes()
        decode(raw)  # warm-up: njit compile + blosc init
        header, expected, rows = stages(raw)
        full = timeit(lambda: decode(raw))
        codec = AGX3Numcodec(header["shape"][1])
        wrapped = timeit(lambda: codec.decode(raw))
        print(f"\nchunk {name}")
        print(f"  compressed {len(raw):,} B -> raw {expected:,} B, "
              f"shape {header['shape']}")
        total = 0.0
        for label, samples in rows:
            med = median_ms(samples)
            total += med
            print(f"  {label:34s} {med:8.2f} ms")
        print(f"  {'sum of stages':34s} {total:8.2f} ms")
        print(f"  {'library decode() end-to-end':34s} {median_ms(full):8.2f} ms")
        print(f"  {'AGX3Numcodec.decode (with copy)':34s} {median_ms(wrapped):8.2f} ms")


if __name__ == "__main__":
    main()
