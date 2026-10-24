"""Import extracted ATLAS replicas through validated plans and existing A converter."""
from __future__ import annotations

from .checker import check_atlas
from ..converter import convert
from ..descriptors.bond_reconstruction import reconstruct_bonds


def import_atlas(workspace, dataset: str, kind: str = "analysis", *,
                 solvent_resnames=("SOL", "WAT", "HOH"), chunk_frames=None) -> dict:
    plans = check_atlas(workspace, dataset, kind)
    results = []
    for plan in plans:
        identifier = f'atlas-{plan["dataset"]}-{plan["kind"]}-{plan["replica"]}'
        tid = convert(workspace, plan["topology"], plan["xtc"],
                      solvent_resnames=solvent_resnames, trajectory_id=identifier,
                      chunk_frames=chunk_frames)
        result = {"replica": plan["replica"], "trajectory_id": tid, "A": "complete"}
        if plan["tpr"] is None:
            result["bonds"] = "unavailable"
        elif plan["bond_error"]:
            result.update(bonds="failed", reason=plan["bond_error"])
        else:
            try:
                reconstruct_bonds(workspace, tid, plan["tpr"], plan["source_to_tpr"])
            except Exception as error:
                result.update(bonds="failed", reason=f"{type(error).__name__}: {error}")
            else:
                result["bonds"] = "complete"
        results.append(result)
    return {"dataset": plans[0]["dataset"], "kind": plans[0]["kind"],
            "status": "complete" if all(r["bonds"] in ("complete", "unavailable")
                                        for r in results) else "partial",
            "replicas": results}
