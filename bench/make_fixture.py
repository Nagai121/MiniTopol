"""Build a synthetic MiniTopol workspace for read-path benchmarking.

Deterministic positions; solvent-free heavy atoms only so converter keeps all.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import MDAnalysis as mda

import os
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "bench-ws")
ATOMS = int(os.environ.get("MT_BENCH_ATOMS", 1536))
RESIDUES = ATOMS // 4
FRAMES = int(os.environ.get("MT_BENCH_FRAMES", 512))

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    base = rng.uniform(0, 60, size=(ATOMS, 3)).astype(np.float32)
    resindex = np.repeat(np.arange(RESIDUES), ATOMS // RESIDUES)
    u = mda.Universe.empty(ATOMS, n_residues=RESIDUES, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("names", [["N", "CA", "C", "O"][i % 4] for i in range(ATOMS)])
    u.add_TopologyAttr("resnames", ["ALA"] * RESIDUES)
    u.add_TopologyAttr("resids", list(range(1, RESIDUES + 1)))
    u.dimensions = [80, 80, 80, 90, 90, 90]
    gro, xtc = ROOT / "synth.gro", ROOT / "synth.xtc"
    u.atoms.positions = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u.atoms.write(str(gro))
    drift = np.array([0.01, 0.005, 0.0], dtype=np.float32)
    with mda.Writer(str(xtc), n_atoms=ATOMS, precision=4) as w:
        for f in range(FRAMES):
            u.atoms.positions = base + drift * f
            u.trajectory.ts.time = f * 100.0
            w.write(u.atoms)
    print("fixture source written", gro, xtc)

if __name__ == "__main__":
    main()
