"""Validate complete, unambiguous source-to-TPR identities before publication.

XTC does not carry atom identity; validation uses the source GRO/PDB/TPR
topology recorded in the trajectory manifest. Never trust numeric index
bounds alone for physical-property or connectivity mappings.
"""
from pathlib import Path

import MDAnalysis as mda

from .storage import fingerprint


def _identities(path):
    universe = mda.Universe(str(path))
    try:
        atoms = universe.atoms
        names = list(map(str, atoms.names))
        residues = list(map(str, atoms.resnames))
        indices = list(map(int, atoms.resindices))
        if not (len(names) == len(residues) == len(indices)):
            raise ValueError("Source/TPR atom identity arrays have inconsistent lengths")
        seen, ordinal, labels = {}, {}, []
        for name, resname, index in zip(names, residues, indices):
            resname = resname.strip().upper()
            if index not in ordinal:
                seen[resname] = seen.get(resname, 0) + 1
                ordinal[index] = seen[resname]
            labels.append((resname, name.strip().upper(), ordinal[index]))
        try:
            elements_attr = atoms.elements
        except AttributeError:
            # MDAnalysis NoDataError is also a ValueError: absent native
            # elements in GRO are allowed, corrupt element axes are not.
            elements = None
        else:
            elements = [str(x).strip().upper() for x in elements_attr]
            if len(elements) != len(labels):
                raise ValueError("Element metadata does not match atom count")
        return labels, elements
    finally:
        if getattr(universe, "trajectory", None) is not None:
            universe.trajectory.close()


def validate_source_mapping(source, tpr, mapping, *, expected_source=None):
    """Reject mismatched or ambiguous source -> TPR mappings.

    Both topology files must be readable. The source hash is checked when
    this is a previously published A trajectory.
    """
    if expected_source is not None and fingerprint(source) != expected_source:
        raise ValueError("Source topology differs from the converted trajectory")
    original, original_elements = _identities(source)
    target, target_elements = _identities(tpr)
    if (not isinstance(mapping, (list, tuple)) or len(mapping) != len(original)
            or any(type(index) is not int or index < 0 or index >= len(target)
                   for index in mapping)
            or any(i >= j for i, j in zip(mapping, mapping[1:]))):
        raise ValueError("Source-to-TPR mapping must cover every source atom with unique ordered in-range indices")
    if not original:
        raise ValueError("Empty source topology cannot be matched")
    for source_index, tpr_index in enumerate(mapping):
        if original[source_index] != target[tpr_index]:
            raise ValueError(
                f"Source-to-TPR chemical identity mismatch at source atom {source_index}: "
                f"{original[source_index]} != {target[tpr_index]}")
        if original_elements is not None and target_elements is not None:
            a, b = original_elements[source_index], target_elements[tpr_index]
            if a and b and a != b:
                raise ValueError(f"Source-to-TPR element mismatch at source atom {source_index}: {a} != {b}")
    # A matching sequence is insufficient if several physically distinct
    # source-to-TPR assignments are possible. Reuse the ATLAS uniqueness rule.
    from .atlas.checker import unique_subsequence
    if list(mapping) != unique_subsequence(tuple(original), tuple(target)):
        raise ValueError("Ambiguous source-to-TPR mapping: topology identities cannot establish unique correspondence")
    return list(mapping)


def validate_workspace_mapping(workspace, trajectory, tpr, mapping):
    """Validate against the exact immutable source registered with A."""
    from .registry import Registry
    manifest = Registry(workspace).manifest(trajectory)
    sources = manifest["options"]["source"]
    topology_names = [key for key in sources if key != "xtc"]
    if len(topology_names) != 1 or topology_names[0] not in ("gro", "pdb", "tpr"):
        raise ValueError("Unknown converted source topology type")
    kind = topology_names[0]
    source = manifest.get(kind + "_path")
    if not source or not Path(source).is_file():
        raise FileNotFoundError("Original source topology is unavailable for TPR identity verification")
    if kind == "tpr" and fingerprint(tpr) != sources[kind]:
        raise ValueError("Direct TPR metadata must use the original topology")
    return validate_source_mapping(source, tpr, mapping, expected_source=sources[kind])
