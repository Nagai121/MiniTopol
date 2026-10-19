"""Command-line entry point; workspace creation is idempotent."""

import argparse
import json

from .converter import convert
from .registry import Registry
from .workspace import Workspace
from .descriptors import ResultStore


def main(argv=None):
    parser = argparse.ArgumentParser(prog="minitopol")
    parser.add_argument("--workspace", help="Override MINITOPOL_WORKSPACE or the stable project workspace")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("list")
    sub.add_parser("recover")
    importing = sub.add_parser("convert")
    importing.add_argument("gro")
    importing.add_argument("xtc")
    importing.add_argument("--solvent", nargs="*", required=True, help="Solvent residue names; zero entries means none")
    importing.add_argument("--id")
    importing.add_argument("--chunk-frames", type=int, help="Default: adapt to atom count (~32 MiB)")
    importing.add_argument("--elements", help="JSON array with one explicit element per source atom")
    register = sub.add_parser("register")
    register.add_argument("name")
    register.add_argument("definition", help="JSON with version, groups, descriptor_definitions")
    annotation = sub.add_parser("annotate")
    annotation.add_argument("trajectory")
    annotation.add_argument("metadata", help="JSON with chains and optional alpha_carbons, masses, radii; indices refer to A")
    calculate = sub.add_parser("calculate")
    calculate.add_argument("trajectory")
    calculate.add_argument("--coordinates", required=True, choices=["nonperiodic", "continuous", "nojump"])
    calculate.add_argument("--initial-whole", action="store_true")
    calculate.add_argument("--half-box-steps", action="store_true")
    calculate.add_argument("--descriptors", nargs="+")
    calculate.add_argument("--memory-mb", type=float, default=128)
    calculate.add_argument("--chunk-frames", type=int)
    calculate.add_argument("--pair-block", type=int, default=32)
    calculate.add_argument("--fft-block", type=int, help="Default: adapt FFT time tile to the memory budget")
    args = parser.parse_args(argv)
    workspace = Workspace(args.workspace)
    if args.command == "init":
        print(workspace.path)
    elif args.command == "list":
        print(json.dumps({"trajectories": workspace.list_trajectories(),
                          "representations": Registry(workspace).representations()}))
    elif args.command == "convert":
        from pathlib import Path
        elements = json.loads(Path(args.elements).read_text()) if args.elements else None
        print(convert(workspace, args.gro, args.xtc, solvent_resnames=args.solvent,
                      trajectory_id=args.id, atom_elements=elements, chunk_frames=args.chunk_frames))
    elif args.command == "register":
        from pathlib import Path
        definition = json.loads(Path(args.definition).read_text())
        print(Registry(workspace).register(args.name, **definition))
    elif args.command == "recover":
        print(ResultStore(workspace).recover())
    elif args.command == "annotate":
        from pathlib import Path
        print(Registry(workspace).annotate(args.trajectory, **json.loads(Path(args.metadata).read_text())))
    elif args.command == "calculate":
        from .descriptors import analyze
        report = analyze(workspace, args.trajectory, descriptors=args.descriptors,
                         coordinate_mode=args.coordinates, initial_whole=args.initial_whole,
                         half_box_steps=args.half_box_steps, memory_mb=args.memory_mb,
                         chunk_frames=args.chunk_frames, pair_block=args.pair_block, fft_block=args.fft_block)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
