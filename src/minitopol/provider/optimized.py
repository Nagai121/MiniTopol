"""Production entry points for the indexed local Provider."""
from .persistent import PersistentProvider

MIB = 1024**2
FEATURES = frozenset((
    'parts', 'adaptive', 'current', 'coord-prefetch', 'global', 'next',
    'catalog', 'encoded', 'native', 'pipeline', 'rama-index', 'contacts-index',
    'direct-ipc', 'rama-compact', 'coordinate-views',
))


class OptimizedProvider(PersistentProvider):
    features = FEATURES
    variant = 'sparse_query'
    workers = 3

    def __init__(self, workspace, budget):
        self.coord_mib = min(512, budget // (12 * MIB))
        self.part_mib = min(3072, budget // (2 * MIB))
        super().__init__(workspace, budget)

    def prepare_ramachandran(self, trajectory):
        paths = self._paths(trajectory, 'ramachandran')
        if not paths:
            raise LookupError(f'No completed Ramachandran result for {trajectory}')
        self.build_derived = True
        self.builds = []
        try:
            for result_id, relative in paths:
                self._rama_tables(result_id, relative)
            return self.builds
        finally:
            self.build_derived = False
