"""Command-line entry point; workspace creation is idempotent."""

import argparse
import json

from .converter import convert
from .registry import Registry
from .workspace import Workspace
from .descriptors import ResultStore


def main(argv=None):
    parser = argparse.ArgumentParser(prog="minitopol")
    parser.add_argument("--workspace", help="Default: MINITOPOL_WORKSPACE or ./BaseTraj_Workspace")
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


if __name__ == "__main__":
    main()
