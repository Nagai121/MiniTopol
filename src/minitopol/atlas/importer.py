"""Import extracted ATLAS replicas through validated plans and existing A converter."""
from __future__ import annotations

from .checker import _check_atlas_unlocked
from ..converter import convert
from ..descriptors.bond_reconstruction import reconstruct_bonds
from ..system_metadata import publish_conditions, publish_molecules


def import_atlas(workspace, dataset: str, kind: str = "analysis", *,
                 solvent_resnames=("SOL", "WAT", "HOH"), chunk_frames=None,
                 hydrogen_bonds=False, periodic=True) -> dict:
    # Keep ATLAS source paths stable from validation through final A/HB import.
    with workspace.lock("atlas-sources"):
        return _import_atlas_unlocked(
            workspace, dataset, kind, solvent_resnames=solvent_resnames,
            chunk_frames=chunk_frames, hydrogen_bonds=hydrogen_bonds,
            periodic=periodic)


def _import_atlas_unlocked(workspace, dataset: str, kind: str = "analysis", *,
                           solvent_resnames=("SOL", "WAT", "HOH"), chunk_frames=None,
                           hydrogen_bonds=False, periodic=True) -> dict:
    plans = _check_atlas_unlocked(workspace, dataset, kind)
    results = []
    for plan in plans:
        if "error" in plan:
            results.append({"replica": plan["replica"], "A": "failed",
                            "bonds": "unavailable", "reason": plan["error"]})
            continue
        identifier = f'atlas-{plan["dataset"]}-{plan["kind"]}-{plan["replica"]}'
        hb = {}
        hb_options = ({"tpr": plan["tpr"], "source_to_tpr": plan["source_to_tpr"],
                       "periodic": periodic}
                      if hydrogen_bonds and plan["tpr"] and not plan["bond_error"] else None)
        try:
            tid = convert(workspace, plan["topology"], plan["xtc"],
                          solvent_resnames=solvent_resnames, trajectory_id=identifier,
                          chunk_frames=chunk_frames,
                          hydrogen_bonds=hb_options, hbond_report=hb)
        except Exception as error:
            results.append({"replica": plan["replica"], "A": "failed",
                            "bonds": "unavailable",
                            "reason": f"{type(error).__name__}: {error}"})
            continue
        result = {"replica": plan["replica"], "trajectory_id": tid, "A": "complete"}
        # System extraction is independent of A and the bond graph.
        try:
            valid_tpr = plan["tpr"] if not plan["bond_error"] else None
            publish_conditions(workspace, tid, tpr=valid_tpr)
            if valid_tpr and plan["source_to_tpr"] is not None:
                result["system"] = publish_molecules(
                    workspace, tid, valid_tpr, plan["source_to_tpr"])
            else:
                result["system"] = "conditions_only"
        except Exception as error:
            result["system"] = "failed"
            result["system_error"] = f"{type(error).__name__}: {error}"
        if hydrogen_bonds:
            if hb_options is None:
                result["hydrogen_bonds"] = "unavailable"
            else:
                result["hydrogen_bonds"] = hb["status"]
                if hb["status"] != "complete":
                    result["hydrogen_bond_error"] = hb.get("error")
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
            "status": "complete" if all(r["A"] == "complete"
                                        and r["bonds"] in ("complete", "unavailable")
                                        and r.get("system") != "failed"
                                        and (not hydrogen_bonds or r.get("hydrogen_bonds") in ("complete", "unavailable"))
                                        for r in results) else "partial",
            "replicas": results}
