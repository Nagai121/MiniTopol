"""Streaming execution contract for explicitly defined per-frame calculators.

Scientific built-ins are added only after their agreed definitions are available.
Windowed/lagged/global statistics must use their own streaming state and publish
their declared intervals through ResultStore rather than resetting each chunk.
"""

from ..registry import Registry
from ..storage import digest
from .results import ResultStore


def calculate_frames(workspace, trajectory, definition, calculator, *, chunk_frames=None):
    """Run a per-frame callable returning an Arrow table for each FrameBlock.

    Return the number of completed partitions. A completed partition is skipped
    on restart. Method/parameters/code version must uniquely identify the callable
    in the definition; MiniTopol does not infer scientific semantics from code.
    """
    registry, store = Registry(workspace), ResultStore(workspace)
    definition.register(workspace)
    manifest = registry.manifest(trajectory)
    partitions = 0
    for block in registry.blocks(trajectory, definition.representation, chunk_frames=chunk_frames):
        identity = digest({"trajectory": trajectory, "source_identity": manifest["identity"],
                           "definition": definition.as_dict(), "start": block.start, "stop": block.stop})
        if not store.is_complete(identity):
            table = calculator(block)
            if "frame" not in table.column_names:
                raise ValueError("Per-frame calculators must return an explicit frame column")
            store.publish(trajectory, definition, block.start, block.stop, table)
        partitions += 1
    return partitions
