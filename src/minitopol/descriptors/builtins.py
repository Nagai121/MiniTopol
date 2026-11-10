"""A-1 scientific calculators plus optional TPR-based electrostatics."""

from importlib import import_module
import shutil
import pyarrow.parquet as pq

from ..storage import atomic_json, digest, fingerprint
from .context import AnalysisContext
from .definitions import DescriptorDefinition
from .results import ResultStore


CALCULATORS = {
    "rmsd": ("rmsd", "angstrom", "sqrt(mean_atoms(|aligned x(t)-aligned x(0)|^2))"),
    "rmsf": ("rmsf", "angstrom", "sqrt(mean_time(|aligned x-mean_time(aligned x)|^2)); denominator N"),
    "msd": ("msd", "angstrom^2", "mean_atoms,all_origins(|aligned x(t+k)-aligned x(t)|^2); all k=0..N-1"),
    "sasa": ("sasa", "angstrom^2", "Shrake-Rupley; 960 golden-angle sphere points; probe 1.4 angstrom; heavy atoms only"),
    "ramachandran": ("ramachandran", "degree", "signed backbone phi/psi from explicitly confirmed continuous segments; [-180,180)"),
    "gyration": ("gyration", "angstrom", "sqrt(sum(m_i*|x_i-center_of_mass|^2)/sum(m_i))"),
    "secondary_structure": ("secondary", "classification", "local phi/psi rectangular regions; no run-length smoothing; null for undefined angles"),
    "contacts": ("contacts", "dimensionless", "minimum heavy-atom residue distance <=4.5 angstrom; sparse frame states and full-trajectory contact fraction"),
    "time_correlation": ("time_correlation", "dimensionless", "C-alpha displacement dot autocorrelation, N-k origins, normalized by full-trajectory variance; all lags"),
    "dynamic_correlation": ("dynamic_correlation", "dimensionless", "aligned C-alpha displacement cross correlation normalized by both variances; residue pairs i<j"),
    "electrostatic_score": ("electrostatics", "e^2/angstrom", "TPR all-atom residue net-charge product / PBC-aware heavy-atom geometric-center distance; cutoff 15 angstrom"),
}


def definition_for(ctx, name):
    _, unit, method = CALCULATORS[name]
    parameters = dict(ctx.parameters)
    if name in ("msd", "time_correlation"):
        parameters.update(lags="0..N-1", origins="all N-k", time_unit="ps",
                          timestamp_tolerance="max(1e-6 ps, 4*eps_float32*max(abs(time)))", resampling=False)
    if name == "sasa":
        parameters.update(probe_angstrom=1.4, sphere_points=960, sphere_method="golden angle; z=1-2*(i+0.5)/n",
                          radii=ctx.properties["radii"], radii_source=ctx.properties["radii_source"],
                          periodic_image_occluders=False, unknown_radius="fail; no silent default")
        parameters["precision"]='CPU-equivalent float64 predicate via bounded float32 filter; float64 boundary fallback'
    if name == "gyration":
        path = ctx.workspace.trajectory_path(ctx.trajectory) / "system" / "atom_properties.parquet"
        has_mass = path.is_file() and "mass" in pq.read_schema(path).names
        parameters.update(masses=ctx.properties["masses"],
                          mass_source="TPR" if has_mass else ctx.properties["mass_source"],
                          tpr_mass_attributes_sha256=fingerprint(path)["sha256"] if has_mass else None,
                          unknown_mass="fail")
    if name == "electrostatic_score":
        path = ctx.workspace.trajectory_path(ctx.trajectory) / "system" / "atom_properties.parquet"
        if not path.is_file():
            raise ValueError("Electrostatic score requires TPR atom properties")
        parameters.update(charge_source="TPR, all atoms including hydrogen",
                          tpr_attributes_sha256=fingerprint(path)["sha256"],
                          charge_epsilon_e=1e-5, cutoff_angstrom=15.0,
                          distance="geometric centers of A heavy atoms",
                          boundary="minimum image for periodic mode; unwrap each residue before centering",
                          eligible="all distinct residue pairs, including neighbors",
                          omitted="implicit zero for neutral or out-of-cutoff pairs",
                          missing_charge="error", average="not stored")
    if name == "secondary_structure":
        parameters.update(helix_phi=[-100, -30], helix_psi=[-80, -5], extended_phi=[-180, -90],
                          extended_psi=[[90, 180], [-180, -150]], angles="[-180,180); other bounds inclusive",
                          minimum_run_length=None, missing="null with reason")
    if name == "contacts":
        parameters.update(cutoff_angstrom=4.5, excluded_sequence_distance=2,
                          sequence_source="explicit continuous segments only", zero_frequency="implicit zero only for eligible pairs",
                          state_value=1, frequency_denominator="all N frames")
    if name in ("time_correlation", "dynamic_correlation"):
        parameters.update(zero_variance="null", variance_denominator="N")
    return DescriptorDefinition(name, "ES-1" if name == "electrostatic_score" else "A-1", "A", method, unit, parameters)


def analyze(workspace, trajectory, *, descriptors=None, coordinate_mode,
            initial_whole=False, half_box_steps=False, memory_mb=128,
            chunk_frames=None, pair_block=32, fft_block=None):
    """Calculate requested A descriptors, reporting each unsupported condition.

    All definitions are registered before computation. A failed descriptor is
    never published as complete. KeyboardInterrupt leaves resumable work intact.
    Normal results remain in Parquet, with only paths/definitions in DuckDB.
    """
    if descriptors is None:
        names = [name for name in CALCULATORS if name != "electrostatic_score"]
        path = workspace.trajectory_path(trajectory) / "system" / "atom_properties.parquet"
        if path.is_file() and {"a_point", "tpr_residue_index", "partial_charge"} <= set(pq.read_schema(path).names):
            names.append("electrostatic_score")
    else:
        names = list(descriptors)
    if not names or len(set(names)) != len(names) or any(name not in CALCULATORS for name in names):
        raise ValueError("Choose nonduplicate descriptor names from: " + ", ".join(CALCULATORS))
    ctx = AnalysisContext(workspace, trajectory, coordinate_mode=coordinate_mode,
                          initial_whole=initial_whole, half_box_steps=half_box_steps,
                          memory_mb=memory_mb, chunk_frames=chunk_frames, pair_block=pair_block, fft_block=fft_block)
    definitions = {name: definition_for(ctx, name) for name in names}
    for definition in definitions.values():
        definition.register(workspace)
    report = {"trajectory": trajectory, "source_identity": ctx.manifest["identity"],
              "definition_set": "A-1 + ES-1" if "electrostatic_score" in names else "A-1", "results": {}, "layout": ctx.layout}
    report_path = workspace.path / "analyses" / (digest([trajectory, ctx.parameters, names, ctx.layout]) + ".json")
    execution_names=(["sasa"] if "sasa" in definitions else [])+[name for name in definitions if name!="sasa"]
    ctx._sasa_alignment_requested=bool({'rmsd','rmsf','msd','time_correlation','dynamic_correlation'} & definitions.keys())
    ctx._share_aligned_ca={'rmsd','dynamic_correlation'}.issubset(definitions)
    if ctx._share_aligned_ca:
        execution_names.remove('dynamic_correlation')
        execution_names.insert(execution_names.index('rmsd')+1,'dynamic_correlation')
    report["execution_order"]=execution_names
    ctx._share_raw_backbone={'ramachandran','secondary_structure'}.issubset(definitions)
    with ctx.cache_lock:
        def execute(name):
            definition=definitions[name]
            try:
                output_layout = ({**ctx.layout, "output_partitioning": "compact-v1"}
                                 if name in {"sasa", "ramachandran", "secondary_structure", "contacts", "time_correlation"}
                                 else ctx.layout)
                if name == "ramachandran": output_layout = {**output_layout, "angle_intermediates": "float64-v1"}
                if name == "contacts" and ctx.mode != "nonperiodic":
                    output_layout = {**output_layout, "output_partitioning": "periodic-spatial-zstd-v1",
                                     "spatial_memory_bytes": ctx.budget,
                                     "spatial_part_rows": max(4096,min(16384,ctx.budget//16384))}
                    from .periodic_contacts import plan_pair_block
                    spatial_width = plan_pair_block(len(ctx.residues), ctx.pair_block, ctx.budget)
                    if spatial_width != ctx.pair_block:
                        output_layout["spatial_pair_block"] = spatial_width
                if name in {'sasa','ramachandran','secondary_structure'}:
                    from .compact_output import _frames
                    output_layout={**output_layout,'output_partitioning':'bounded-columnar-v2',
                                   'output_batch_frames':_frames(ctx,name),'output_budget_bytes':ctx.budget}
                if name=='time_correlation':
                    from .compact_output import _timecorr_group_size
                    output_layout={**output_layout,'output_partitioning':'bounded-point-group-v1','point_group_size':_timecorr_group_size(ctx)}
                with ResultStore(workspace).dataset(trajectory, definition, layout=output_layout) as writer:
                    if not writer.complete:
                        module = import_module("." + CALCULATORS[name][0], __package__)

                        from .compact_output import calculate
                        calculate(name, module, ctx, writer)
                report["results"][name] = {"status": "complete", "definition_id": definition.identity,
                                          "result_id": writer.result_id, "rows": writer.state["rows"],
                                          "undefined_values": sum(p.get("undefined_values", 0) for p in writer.state["parts"])}
            except Exception as error:
                report["results"][name] = {"status": "failed", "definition_id": definition.identity,
                                          "reason": f"{type(error).__name__}: {error}"}
            if name=='dynamic_correlation':ctx._shared_aligned_ca=None
            if {'ramachandran','secondary_structure'}.issubset(report['results']):
                ctx._shared_raw_backbone=None
            atomic_json(report_path, report)
        def prepare_independent_descriptor():
            for name in ('ramachandran','secondary_structure'):
                if name in definitions and name not in report['results']:
                    execute(name)
        ctx._gpu_wait_work=prepare_independent_descriptor
        report['execution_policy']='sasa-first; GPUwait may complete independentRamachandran andSecondary beforeSASA'
        for name in execution_names:
            if name not in report['results']:
                execute(name)
        report["sasa_alignment_audit"]=getattr(ctx,"_sasa_alignment_audit",None)
        report["aligned_handoff_audit"] = getattr(ctx,"_aligned_handoff_audit",None)
        report["status"] = "complete" if all(r["status"] == "complete" for r in report["results"].values()) else "incomplete"
        report["report_path"] = str(report_path)
        atomic_json(report_path, report)
        if report["status"] == "complete" and ctx.cache_path.exists():
            # Only this run's owned intermediate coordinate/FFT cache is removed.
            # User trajectories, annotations, partial outputs and results remain.
            ctx.cache = None
            owned = {"state", "aligned.zarr", "aligned-checksums", "msd", "time-correlation-scratch.zarr"}
            if {p.name for p in ctx.cache_path.iterdir()}.issubset(owned):
                shutil.rmtree(ctx.cache_path)
            else:
                report["cache_retained"] = "Unrecognized files are present; no cache directory was deleted"
                atomic_json(report_path, report)
    return report
