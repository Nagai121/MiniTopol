"""Convert the synthetic source and publish read-benchmark descriptors."""
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minitopol import Workspace, convert
from minitopol.descriptors import DescriptorDefinition, ResultStore

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "bench-ws")
FRAMES = 512
PAIRS = 256

def main():
    ws = Workspace(ROOT / "storage")
    tid = convert(ws, ROOT / "synth.gro", ROOT / "synth.xtc",
                  solvent_resnames=["SOL"], chunk_frames=64)
    print("tid", tid)
    store = ResultStore(ws)
    rng = np.random.default_rng(7)
    # Per-frame scalar descriptor (small)
    store.publish(tid, DescriptorDefinition("sample", "1", "A", "test", "unit"), 0, FRAMES,
                  pa.table({"frame": pa.array(range(FRAMES), type=pa.int64()),
                            "value": pa.array(rng.random(FRAMES))}))
    # Large per-frame pair table (generic descriptor, single parquet part)
    frames = np.repeat(np.arange(FRAMES), PAIRS)
    store.publish(tid, DescriptorDefinition("pairdist", "1", "A", "test", "angstrom"), 0, FRAMES,
                  pa.table({"frame": pa.array(frames, type=pa.int64()),
                            "i": pa.array(np.tile(np.arange(PAIRS), FRAMES), type=pa.int64()),
                            "value": pa.array(rng.random(FRAMES * PAIRS).astype(np.float32))}))
    # Global descriptor
    store.publish(tid, DescriptorDefinition("rmsf", "test", "A", "global", "angstrom"), 0, FRAMES,
                  pa.table({"point": pa.array(range(1536), type=pa.int64()),
                            "value": pa.array(rng.random(1536))}))
    print("published")

if __name__ == "__main__":
    main()
