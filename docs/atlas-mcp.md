# ATLAS acquisition and local MCP server

ATLAS is the [Atlas of proTein moLecular dynAmicS](https://www.dsimb.inserm.fr/ATLAS/).
Its publicly documented REST API supports chain metadata and ZIPs in
"analysis" (1000 frames/replicate, no solvent), "protein" (10000 frames,
no solvent) and "total" (10000 frames, full system) variants.
The downloaded ATLAS dataset is provided under **CC BY-NC 4.0**; consult ATLAS
for permitted downstream reuse and attribution.

## Python acquisition and tiny TSV ledger

```python
from minitopol import Workspace
from minitopol.atlas import AtlasClient, DownloadLedger

workspace = Workspace()
client = AtlasClient()
print(client.metadata("1k5n_A"))
result = client.download(workspace, "1k5n_A", "analysis")
print(result["archive"], result["files"])
print(DownloadLedger(workspace).list())
```

Archive is stored under
`BaseTraj_Workspace/trajectories/atlas-1k5n_A-analysis/source/1k5n_A_analysis.zip`.
By default, only *.xtc, *.pdb, *.tpr, *.gro members are extracted beside
the ZIP, without trusting ZIP member paths. Use `extract=False` to retain only
the archive. An archive with a matching ledger entry is rechecked before reuse;
untracked or checksum-mismatched existing files are never overwritten.
A crash after publication of a valid ZIP but before the completion record can
be recovered from the existing archive without another network transfer.
Abandoned unpublished `.atlas-download-*` and `.atlas-extract-*` temporary
files are cleaned up on the next locked acquisition. The ledger is an atomic TSV
at `BaseTraj_Workspace/atlas/downloads.tsv` and is separate from DuckDB,
which tracks only **converted** trajectories. The ledger reports `complete` only when the requested operation succeeds,
including extraction when requested. If download, archive validation, or
extraction fails, the ledger reports `failed` and includes the error;
verified ZIP archives are retained for safe extraction retry.

ATLAS usually supplies PDB/TPR + XTC rather than a matching GRO + XTC.
After download and extraction, use the separate checker/import step:
`python -m minitopol atlas-check 1k5n_A --kind analysis` and
`python -m minitopol atlas-import 1k5n_A --kind analysis`.
The checker identifies XTC replicas, selects unambiguous GRO/PDB input topologies,
checks XTC atom counts, identifies repeated residue occurrences in the
topology, and validates a unique ordered TPR atom mapping. When an ATLAS
download ledger entry exists, the checker also verifies its ZIP hash and all
extracted input CRCs. Manually staged inputs are marked `manual` and do not
carry a provider-verified provenance claim.
Each replica becomes a separate A trajectory; a usable TPR also produces a
separate static heavy-atom covalent graph. TPR errors are reported separately
and never destroy an already completed A. Missing TPR permits A-only import.
No dummy GRO, PDB-derived bonds or distance-based bond inference is used.
The TPR bond reader excludes nonchemical harmonic restraints, nonconnecting
constraints and tabulated nonconnections; it retains only GROMACS
chemical-connection interaction types. Zero-edge graphs are valid.
XTC itself has no atom identifiers: matching counts plus unambiguous ATLAS
file naming do not independently prove the actual atom order inside XTC.
Unverifiable pairings and ambiguous TPR mappings must not be silently guessed.
If one replica fails validation or A conversion, other independently validated
replicas can still complete; failures are reported by replica.
The original GRO + XTC converter and the ten scientific definitions remain unchanged.
Full-sized live ATLAS download has not been verified in CI (offline mocks only).
The ZIP is checked for a valid central directory, the presence of an XTC,
and CRC integrity of all members, in addition to HTTP Content-Length when
provided. Its SHA-256 is recorded; this is not a provider-supplied checksum.
Verification requires reading/decompressing the ZIP contents but operates with
bounded buffer memory. Existing archives are not automatically overwritten
if corruption is discovered.

API references:
- https://www.dsimb.inserm.fr/ATLAS/api/redoc
- https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/metadata/1k5n_A
- https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/1k5n_A

## MCP adapter

Install the optional dependency and start the **local stdio** server:

```bash
python -m pip install -e ".[mcp]"
minitopol-mcp
```

As for MiniTopol itself, `MINITOPOL_WORKSPACE` chooses the workspace.
Configure an MCP client to launch the installed `minitopol-mcp` command.
The server communicates via stdout JSON-RPC; do not print ordinary
diagnostics on stdout. No remote HTTP MCP listener is started.

Available tools include ATLAS metadata/download/ledger, atlas_check, atlas_import, get_bond_graph, converted trajectory
and representation listing, GRO+XTC conversion, chain annotation, A
descriptor calculation, C representation registration/definition reading, completed descriptor result
metadata listing, and limited result row retrieval (at most 1000 rows per call).
MCP calls use existing Python APIs; scientific calculations are not
reimplemented. Descriptor calculations are **synchronous** and can exceed
an MCP client's timeout; no job scheduler is claimed or added. Large frames
or result tables are deliberately not dumped through MCP tools.

No API keys are needed for the public ATLAS endpoints. Local MCP clients
still receive filesystem access permitted to their OS user.

## Independent bond graph

`BondStore` and `Registry.bonds(trajectory)` expose a verified Parquet edge table
with columns `point_i`, `point_j` (0-based A heavy-atom point indices).
The original A arrays, A manifest, existing scientific descriptors and the ATLAS
TSV ledger are not altered. A bond publication is independently crash-resumable,
checksum verified and idempotent for the same A/TPR/mapping.
The logical bond DB is the dedicated `bond_graphs` table in the existing DuckDB;
the edge payload lives in `trajectories/<id>/bonds.parquet`, with `bonds.json`
as its independently published completion marker.

`atlas_import` reports separate A/bond statuses per replica. `atlas_check`
only examines already extracted files. Use `atlas_download` first for retrieval;
it does not automatically start conversion. For a pre-existing A, Python's
`reconstruct_bonds(workspace, trajectory, tpr, validated_source_to_tpr)`
adds only the bond graph without recomputing A.
