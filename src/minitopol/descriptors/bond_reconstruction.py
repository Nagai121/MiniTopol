"""Restore static chemical connectivity from GROMACS TPR interaction types.

MDAnalysis Universe.bonds also includes nonchemical constraints and harmonic
restraints. Read the typed TPR interaction lists without modifying MDAnalysis
global state, and retain only GROMACS chemical-connection function types.
"""
from __future__ import annotations

import numpy as np

from ..bond_store import BondStore
from ..registry import Registry
from ..storage import digest, fingerprint

# GROMACS molecular graph connections: [bonds] types 1..5,7,8
# and [constraints] type 1. Type 6 HARMONIC, type 9 TABBONDSNC,
# type 10 RESTRAINTPOT, and type 2 CONSTRNC are not chemical edges.
_CHEMICAL = frozenset({
    "BONDS", "G96BONDS", "MORSE", "CUBICBONDS", "CONNBONDS",
    "FENEBONDS", "TABBONDS", "CONSTR",
})


def _edges_in_ilists(ilists):
    """Return molecule-local chemical atom pairs from typed TPR interactions."""
    from MDAnalysis.topology.tpr.obj import InteractionKind

    edges = set()
    for ilist in ilists:
        name = ilist.ik[0]
        if not ilist.nr or name not in _CHEMICAL:
            continue
        interaction = InteractionKind(*ilist.ik)
        if interaction.natoms != 2:
            raise ValueError(f"Unrecognized connectivity arity: {name}")
        for pair in interaction.process(ilist.iatoms):
            a, b = map(int, pair)
            if a == b:
                raise ValueError("TPR contains a self-connection")
            edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def _covalent_edges_from_tpr(tpr):
    """Decode typed TPR interactions without treating restraints as bonds.

    GROMACS SETTLE (water O-H) is omitted: A excludes hydrogens and solvent,
    so it cannot produce an edge in the resulting heavy-atom graph.
    """
    from MDAnalysis.topology.tpr import utils

    with open(tpr, "rb") as stream:
        data = utils.TPXUnpacker(stream.read())
    th = utils.read_tpxheader(data)
    if th.fver >= 119 and th.fgen >= 27:
        data = utils.TPXUnpacker2020.from_unpacker(data)
    if th.bBox:
        utils.extract_box_info(data, th.fver)
    if th.ngtc > 0:
        if th.fver < 69:
            utils.ndo_real(data, th.ngtc)
        utils.ndo_real(data, th.ngtc)
    if not th.bTop:
        raise ValueError("TPR contains no topology")
    symtab = utils.do_symtab(data)
    utils.do_symstr(data, symtab)
    utils.do_ffparams(data, th.fver)
    moltypes = []
    for _ in range(data.unpack_int()):
        utils.do_symstr(data, symtab)
        atoms = utils.do_atoms(data, symtab, th.fver)
        ilists = utils.do_ilists(data, th.fver)
        edges = _edges_in_ilists(ilists)
        if any(a < 0 or b >= atoms.nr for a, b in edges):
            raise ValueError("TPR contains out-of-range bond indices")
        utils.do_block(data)
        utils.do_blocka(data)
        moltypes.append((atoms.nr, edges))
    count = 0
    all_edges = []
    for _ in range(data.unpack_int()):
        block = utils.do_molblock(data)
        atoms_per_molecule, edges = moltypes[block.molb_type]
        if block.molb_natoms_mol != atoms_per_molecule:
            raise ValueError("TPR molecule block atom count mismatch")
        for _ in range(block.molb_nmol):
            all_edges.extend((count + a, count + b) for a, b in edges)
            count += atoms_per_molecule
    return count, all_edges


def reconstruct_bonds(workspace, trajectory: str, tpr: str,
                      source_to_tpr: list[int]) -> int:
    """Use checker-provided atom mapping, never independently infer one."""
    if source_to_tpr is None:
        raise ValueError("A validated source-to-TPR atom mapping is required")
    atoms = Registry(workspace).atoms(trajectory).to_pydict()
    source_atoms = atoms["source_atom"]
    if (not source_atoms or max(source_atoms) >= len(source_to_tpr)
        or any(type(index) is not int or index < 0 for index in source_to_tpr)
        or len(set(source_to_tpr)) != len(source_to_tpr)):
        raise ValueError("Invalid source-to-TPR atom mapping")
    total, bonds = _covalent_edges_from_tpr(tpr)
    if not source_to_tpr or max(source_to_tpr) >= total:
        raise ValueError("TPR mapping points outside its topology")
    to_a = {source_to_tpr[source_atom]: point
            for point, source_atom in enumerate(source_atoms)}
    edges = set()
    for i, j in bonds:
        if i in to_a and j in to_a:
            a, b = sorted((to_a[i], to_a[j]))
            if a == b:
                raise ValueError("TPR contains a self-connection")
            edges.add((a, b))
    return BondStore(workspace).publish(
        trajectory, edges, tpr_sha256=fingerprint(tpr)["sha256"],
        mapping_sha256=digest(source_to_tpr))
