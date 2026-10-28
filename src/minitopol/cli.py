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
    sub.add_parser("migrate-atlas-sources")
    run = sub.add_parser("run")
    run.add_argument("jobs", help="JSON array of independent trajectory pipeline jobs")
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--native-threads", type=int, default=2)
    run.add_argument("--gpu-workers", type=int, default=2)
    run.add_argument("--rss-limit-mb", type=float, default=4096)
    atlas_check = sub.add_parser("atlas-check")
    atlas_check.add_argument("dataset")
    atlas_check.add_argument("--kind", choices=["analysis", "protein", "total"], default="analysis")
    atlas_import = sub.add_parser("atlas-import")
    atlas_import.add_argument("dataset")
    atlas_import.add_argument("--kind", choices=["analysis", "protein", "total"], default="analysis")
    atlas_import.add_argument("--solvent", nargs="*", default=["SOL", "WAT", "HOH"])
    atlas_import.add_argument("--hydrogen-bonds", action="store_true",
                              help="Extract H-bonds from full-atom XTC and TPR during A import")
    atlas_import.add_argument("--nonperiodic-hbonds", action="store_true")
    importing = sub.add_parser("convert")
    importing.add_argument("gro")
    importing.add_argument("xtc")
    importing.add_argument("--solvent", nargs="*", required=True, help="Solvent residue names; zero entries means none")
    importing.add_argument("--id")
    importing.add_argument("--chunk-frames", type=int, help="Default: adapt to atom count (~32 MiB)")
    importing.add_argument("--elements", help="JSON array with one explicit element per source atom")
    importing.add_argument("--mdp", help="Optional GROMACS MDP settings for system_conditions.json")
    importing.add_argument("--hbond-tpr", help="Optional TPR for same-pass hydrogen-bond extraction")
    importing.add_argument("--hbond-mapping", help="JSON array with validated source-to-TPR atom indices")
    importing.add_argument("--nonperiodic-hbonds", action="store_true")
    backfill = sub.add_parser("hydrogen-bonds-backfill")
    backfill.add_argument("trajectory")
    backfill.add_argument("gro")
    backfill.add_argument("xtc")
    backfill.add_argument("tpr")
    backfill.add_argument("mapping", help="JSON atom mapping")
    backfill.add_argument("--solvent", nargs="*", required=True)
    backfill.add_argument("--nonperiodic", action="store_true")
    system = sub.add_parser("system-metadata")
    system.add_argument("trajectory")
    system.add_argument("--tpr", help="Optional GROMACS TPR for static molecule data")
    system.add_argument("--mapping", help="JSON array: validated source-to-TPR atom indices")
    system.add_argument("--mdp", help="Optional GROMACS MDP file")
    system.add_argument("--conditions", help="JSON object of explicitly known condition values")
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
    if args.command == "run":
        if args.workspace:
            parser.error("run uses the explicit workspace in each job")
        from pathlib import Path
        from .batch import run_pipelines
        path = Path(args.jobs).resolve()
        jobs = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(jobs, list):
            for job in jobs:
                if isinstance(job, dict):
                    for field in ("workspace", "gro", "xtc", "tpr"):
                        if field in job:
                            job[field] = str(path.parent / job[field])
        report = run_pipelines(jobs, workers=args.workers, native_threads=args.native_threads,
                               gpu_workers=args.gpu_workers, rss_limit_mb=args.rss_limit_mb)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "complete":
            raise SystemExit(1)
        return
    workspace = Workspace(args.workspace)
    if args.command == "init":
        print(workspace.path)
    elif args.command == "migrate-atlas-sources":
        from .atlas.ledger import migrate_atlas_sources
        print(json.dumps({"source_directories_moved": migrate_atlas_sources(workspace)}))
    elif args.command == "system-metadata":
        from pathlib import Path
        from .system_metadata import publish_conditions, publish_molecules
        if bool(args.tpr) != bool(args.mapping):
            parser.error("--tpr and --mapping must be supplied together")
        values = json.loads(Path(args.conditions).read_text(encoding="utf-8")) if args.conditions else None
        conditions = publish_conditions(workspace, args.trajectory, tpr=args.tpr,
                                        mdp=args.mdp, supplied=values)
        molecules = (publish_molecules(
            workspace, args.trajectory, args.tpr,
            json.loads(Path(args.mapping).read_text(encoding="utf-8")))
            if args.tpr else None)
        print(json.dumps({"conditions": conditions, "molecules": molecules}, ensure_ascii=False))
    elif args.command == "list":
        print(json.dumps({"trajectories": workspace.list_trajectories(),
                          "representations": Registry(workspace).representations()}))
    elif args.command == "atlas-check":
        from .atlas import check_atlas
        plans = check_atlas(workspace, args.dataset, args.kind)
        print(json.dumps([{k: v for k, v in plan.items() if k != "source_to_tpr"}
                          for plan in plans], indent=2))
    elif args.command == "atlas-import":
        from .atlas import import_atlas
        result = import_atlas(workspace, args.dataset, args.kind,
                              solvent_resnames=args.solvent,
                              hydrogen_bonds=args.hydrogen_bonds,
                              periodic=not args.nonperiodic_hbonds)
        print(json.dumps(result, indent=2))
        if result["status"] != "complete":
            raise SystemExit(1)
    elif args.command == "convert":
        from pathlib import Path
        elements = json.loads(Path(args.elements).read_text()) if args.elements else None
        if bool(args.hbond_tpr) != bool(args.hbond_mapping):
            parser.error("--hbond-tpr and --hbond-mapping must be supplied together")
        hb = {}
        options = ({"tpr": args.hbond_tpr,
                    "source_to_tpr": json.loads(Path(args.hbond_mapping).read_text()),
                    "periodic": not args.nonperiodic_hbonds} if args.hbond_tpr else None)
        tid = convert(workspace, args.gro, args.xtc, solvent_resnames=args.solvent,
                      trajectory_id=args.id, atom_elements=elements, chunk_frames=args.chunk_frames,
                      hydrogen_bonds=options, hbond_report=hb)
        if args.mdp or (options and hb["status"] == "complete"):
            from .system_metadata import publish_conditions, publish_molecules
            publish_conditions(workspace, tid, mdp=args.mdp,
                               tpr=args.hbond_tpr if options else None)
            if options and hb["status"] == "complete":
                publish_molecules(workspace, tid, args.hbond_tpr,
                                  options["source_to_tpr"])
        print(json.dumps({"trajectory": tid, "hydrogen_bonds": hb} if options else tid))
        if options and hb["status"] != "complete":
            raise SystemExit(1)
    elif args.command == "hydrogen-bonds-backfill":
        from pathlib import Path
        from .hydrogen_bonds import reconstruct_hydrogen_bonds
        report = reconstruct_hydrogen_bonds(
            workspace, args.trajectory, args.gro, args.xtc, args.tpr,
            json.loads(Path(args.mapping).read_text()), args.solvent,
            periodic=not args.nonperiodic)
        print(json.dumps(report))
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
