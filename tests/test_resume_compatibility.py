"""Supported unfinished formats resume without external source trees."""
import hashlib
import numpy as np
import pyarrow as pa
import pytest
import zarr
from minitopol import Workspace, convert
import minitopol.converter as converter
from minitopol.descriptors import DescriptorDefinition, ResultStore
from minitopol.storage import atomic_json, read_json


def test_auto_codec_partial_resumes_without_metadata_migration(tmp_path, trajectory, monkeypatch):
    pytest.importorskip('pcodec')
    from zarr.codecs.numcodecs import PCodec
    ws = Workspace(tmp_path / 'resume')
    original = converter.atomic_json
    def interrupt(path, data):
        if data.get('committed_frames') == 8:
            raise RuntimeError('uncommitted second chunk')
        original(path, data)
    def auto_codec(*args, **kwargs):
        kwargs.pop('delta_spec', None)
        kwargs.pop('delta_encoding_order', None)
        return PCodec(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr('zarr.codecs.numcodecs.PCodec', auto_codec)
        patch.setattr(converter, 'atomic_json', interrupt)
        with pytest.raises(RuntimeError):
            convert(ws, *trajectory[:2], solvent_resnames=['SOL'], trajectory_id='same', chunk_frames=4)
    path = ws.trajectory_path('same') / 'A.zarr'
    before = (path / 'positions/zarr.json').read_bytes()
    convert(ws, *trajectory[:2], solvent_resnames=['SOL'], trajectory_id='same', chunk_frames=4)
    assert (path / 'positions/zarr.json').read_bytes() == before
    fresh = Workspace(tmp_path / 'fresh')
    convert(fresh, *trajectory[:2], solvent_resnames=['SOL'], trajectory_id='same', chunk_frames=4)
    actual = zarr.open_group(str(path), mode='r')
    expected = zarr.open_group(str(fresh.trajectory_path('same') / 'A.zarr'), mode='r')
    for key in ['positions', 'time', 'box']:
        assert np.asarray(actual[key][:]).tobytes() == np.asarray(expected[key][:]).tobytes()


def test_v3_partial_stays_v3_and_preserves_existing_parts(tmp_path, trajectory):
    ws = Workspace(tmp_path / 'ws')
    tid = convert(ws, *trajectory[:2], solvent_resnames=['SOL'], chunk_frames=4)
    definition = DescriptorDefinition('resume_compat', '1', 'A', 'test-only', 'dimensionless')
    definition.register(ws)
    store = ResultStore(ws)
    layout = {'test': 1}
    table = pa.table({'frame': [0], 'value': [1.]})
    with pytest.raises(RuntimeError):
        with store.dataset(tid, definition, layout=layout) as writer:
            writer.journal = False
            atomic_json(writer.progress, writer.state)
            writer.append('0', table)
            saved = next(writer.directory.glob('*.parquet'))
            before = (hashlib.sha256(saved.read_bytes()).hexdigest(), saved.stat().st_mtime_ns)
            raise RuntimeError('unfinished v3 fixture')
    with store.dataset(tid, definition, layout=layout) as writer:
        assert writer.journal is False and read_json(writer.progress)['schema'] == 3
        writer.append('0', table)
        writer.append('1', pa.table({'frame': [1], 'value': [2.]}))
    assert (hashlib.sha256(saved.read_bytes()).hexdigest(), saved.stat().st_mtime_ns) == before
    with store.dataset(tid, definition, layout=layout) as writer:
        assert writer.complete and writer.state['rows'] == 2
