# Hydrogen-bond event layer (opt-in)

MiniTopol keeps the full-atom source GRO/PDB + XTC and validated TPR atom
mapping only during import. This optional operation extracts geometrically
qualified hydrogen-bond events **on the same source-frame pass** as heavy-atom
A conversion. A, the original static heavy-atom bond graph, and all ten A-1
descriptors are unchanged. Missing TPR, hydrogen coordinates, or unverified
atom correspondence never produces a fabricated "zero bonds" result.

## Inputs and definition

* Source GRO/PDB and XTC, with explicit hydrogen coordinates for relevant
  donor–hydrogen pairs.
* TPR provides the molecule membership, charges, and typed covalent
  connectivity. Unlike the legacy A bond graph, the HB reader also accepts
  GROMACS SETTLE O-H connectivity. Harmonic restraints and nonconnecting
  constraints are not interpreted as covalent edges.
* A **complete, unique, order-preserving** source-to-TPR atom mapping, validated
  by the ATLAS checker or supplied explicitly. No geometric atom matching.
* Explicit solvent residue names, the same set used by the A converter.
* Default geometry: donor–acceptor distance <= 3.0 Å and D–H–A angle >= 150°.
  Periodic analyses require a valid per-frame box; nonperiodic mode is
  explicitly selectable. The distance uses the minimum-image convention.
* Default chemistry is **explicitly a versioned heuristic**: N/O/S atoms bonded
  to explicit H/D/T via typed TPR connections are donors; N/O/S atoms with TPR
  charge <= -0.5 e are acceptors. This is not a universal force-field-
  independent hydrogen-bond definition. Other chemistry requires a new
  definition/version rather than silent fallback.

An event is a unique D-H-A triplet at one source frame. Donor and acceptor
molecular membership produces mutually exclusive categories:

* intramolecular: same non-solvent molecule;
* intermolecular: distinct non-solvent molecules;
* solvent_molecular: one solvent and one non-solvent molecule.

Solvent-solvent interactions are omitted. Solvent absent from the source
produces "unavailable", not zero contacts. Geometry definitions and input
hashes are stored separately from the ten scientific A-1 descriptors.

## Storage and recovery

    trajectories/<id>/hydrogen_bonds/
        atoms.parquet
        intramolecular/part-00000000.parquet ...
        intermolecular/part-00000000.parquet ...
        solvent_molecular/part-00000000.parquet ...
        progress.json  (incomplete only)
        manifest.json  (completed only)

Each part is a Parquet+Zstd table with frame, donor, hydrogen, acceptor,
distance_angstrom, angle_degree. Donor/H/acceptor IDs reference the
source_atom in atoms.parquet, which includes the TPR mapping, A-point link,
molecule ID and solvent flag. Only present events are stored. Time for a
frame is retrieved from A.zarr. The full event definition and SHA-256 checks
appear in the completion sidecar. DuckDB's hydrogen_bond_sets table contains
only dataset metadata, not events.

The H-bond progress marker is independent of A's conversion manifest and
the static bonds.json. A failed H-bond extraction does not change the
already-committed A chunks, and retry on the same source/definition resumes
the remaining event parts. Existing completed event parts are verified,
not silently overwritten. Only manifest-complete results are discoverable.

## Running

Batch pipelines may include:

    {
      "workspace": "...", "gro": "...", "xtc": "...",
      "solvent_resnames": ["SOL"], "tpr": "...",
      "source_to_tpr": [0, 1, 2, ...],
      "hydrogen_bonds": true,
      "analysis": {"coordinate_mode": "nojump",
                   "initial_whole": true, "half_box_steps": true}
    }

The same-source conversion calls the HB processor when explicitly requested.
For ATLAS import, use:

    minitopol atlas-import 1k5n_A --kind total --hydrogen-bonds

ATLAS protein/analysis distributions do not provide solvent coordinates; the
solvent category is unavailable. The full-atom total distribution is needed
for solvent–solute hydrogen bonds. Import results report A, covalent bonds,
and hydrogen bonds independently per replica.

For a standalone conversion:

    minitopol convert input.gro input.xtc --solvent SOL \
      --hbond-tpr input.tpr --hbond-mapping validated_mapping.json

To recover HB data independently from the unchanged original inputs:

    minitopol hydrogen-bonds-backfill <id> input.gro input.xtc input.tpr \
      validated_mapping.json --solvent SOL

Python interfaces: HydrogenBondStore (streaming import), 
reconstruct_hydrogen_bonds (source backfill), hydrogen_bond_info,
hydrogen_bond_events and recover_hydrogen_bonds. Registry.hydrogen_bond_info
exposes the verified result. MCP provides get_hydrogen_bond_info and
query_hydrogen_bond_events (maximum 1000 rows). Completed-data registration
can be restored from verified manifest.json without recalculating geometry.

## Limitations

The default acceptor charge filter can omit chemically valid acceptors and
does not by itself establish lone-pair availability. H-bond geometry is an
explicit operational label, not an energetic proof of binding. This first
version does not perform water-bridge analysis, lifetime smoothing, alternative
hydrogen-placement modeling, or full-source trajectories lacking hydrogen
coordinates. If only active event rows are retained, a subsequently relaxed
distance/angle cutoff cannot be reconstructed without the source trajectory.
Large event sets should be streamed or selected by time window rather than
materialized as one table. Continuous source-frame iteration avoids materializing
solvent coordinates for all frames in a conversion chunk.
