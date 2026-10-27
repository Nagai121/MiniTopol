"""Persisted codec contracts, independent of benchmark fixtures or directories."""
import numpy as np
import pytest

from minitopol import position_grid
from minitopol.agx_zarr import AGXNumcodec


@pytest.mark.parametrize("frames,atoms", [(1, 1), (17, 257), (257, 3)])
def test_agx_preserves_all_float32_bits_and_rejects_corruption(frames, atoms):
    bits = np.random.default_rng(atoms).integers(0, 2**32, (frames, atoms, 3), dtype="u4")
    values = bits.view("f4")  # Includes non-finite values and arbitrary mantissas.
    codec = AGXNumcodec(atoms)
    encoded = codec.encode(values)
    assert codec.decode(encoded) == values.tobytes()
    broken = bytearray(encoded)
    broken[-1] ^= 1
    with pytest.raises((IOError, ValueError, RuntimeError)):
        codec.decode(broken)


def test_agx_grid_residual_keeps_signed_zero_and_off_grid_values():
    values = np.array([0., -0., .1, np.nextafter(np.float32(.1), np.float32(1.)), 7., -3.], dtype="f4").reshape(1, 2, 3)
    encoded, _ = position_grid.encode(values)
    np.testing.assert_array_equal(position_grid.decode(encoded).view("u4"), values.view("u4"))


@pytest.mark.parametrize("frames,atoms", [(1, 1), (17, 13), (257, 3)])
def test_native_sasa_count_axes_and_all_values_roundtrip(frames, atoms):
    from minitopol.direct_counts import encode_native, extract
    values = np.random.default_rng(frames).integers(0, 961, (frames, atoms), dtype="u2")
    indexes = np.arange(11, 11 + frames)
    times = np.arange(frames, dtype="f8") * .125
    encoded = encode_native(values, indexes, times)
    actual, actual_indexes, actual_times = extract(encoded)
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(actual_indexes, indexes)
    np.testing.assert_array_equal(actual_times.view("u8"), times.view("u8"))
    broken = bytearray(encoded)
    broken[-1] ^= 1
    with pytest.raises((AssertionError, IOError, ValueError)):
        extract(broken)


def test_native_sasa_encoder_keeps_cpu_only_install_supported(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    monkeypatch.setitem(sys.modules, "numba", None)
    spec = importlib.util.spec_from_file_location("cpu_only_counts", Path(__file__).parents[1] / "src/minitopol/direct_counts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.array([[0, 960], [1, 959]], dtype="u2")
    actual, _, _ = module.extract(module.encode_native(values, np.arange(2), np.arange(2, dtype="f8")))
    np.testing.assert_array_equal(actual, values)


def test_sasa_published_full_range_prefix_and_axes_integrity(tmp_path,trajectory,monkeypatch):
    from minitopol import Workspace,convert
    from minitopol.descriptors import analyze
    from minitopol.descriptors import count_store
    from minitopol.storage import read_json
    monkeypatch.setenv('MINITOPOL_SASA_DEVICE','cpu')
    ws=Workspace(tmp_path/'ws');gro,xtc,_=trajectory
    tid=convert(ws,gro,xtc,solvent_resnames=['SOL'])
    report=analyze(ws,tid,descriptors=['sasa'],coordinate_mode='nonperiodic')
    assert report['status']=='complete'
    definition=report['results']['sasa']['definition_id']
    with ws.catalog() as con:relative=con.execute("SELECT path FROM results WHERE descriptor='sasa'").fetchone()[0]
    marker=read_json((ws.path/relative).with_suffix('.json'));part=marker['parts'][0]
    full=ws.query_results(definition_id=definition)
    assert count_store.read_rows(ws,relative,2)==full.slice(0,2).to_pylist()
    assert count_store.read_rows(ws,relative,0)==[]
    empty=ws.query_results(descriptor='sasa',start=100,stop=101)
    assert len(empty)==0 and empty.schema.equals(full.schema,check_metadata=True)
    with pytest.raises(LookupError):ws.query_results(descriptor='sasa',representation='C')
    first_part=(ws.path/relative).parent/part['file']
    for path in (first_part,first_part.parent/'axes'):
        original=path.read_bytes();broken=bytearray(original);broken[-1]^=1
        try:
            path.write_bytes(broken)
            for read in (lambda:list(count_store.logical_tables(ws,relative)),
                         lambda:ws.query_results(descriptor='sasa',start=0,stop=1),
                         lambda:count_store.read_rows(ws,relative,2)):
                with pytest.raises((IOError,ValueError,AssertionError)):read()
        finally:path.write_bytes(original)


def test_virtual_angles_missing_annotations_and_runtime_pin(tmp_path,trajectory,monkeypatch):
    from minitopol import Workspace,convert
    from minitopol.descriptors import analyze
    from minitopol.descriptors import virtual_rama_store
    ws=Workspace(tmp_path/'ws');gro,xtc,_=trajectory
    tid=convert(ws,gro,xtc,solvent_resnames=['SOL'])
    report=analyze(ws,tid,descriptors=['ramachandran'],coordinate_mode='nonperiodic')
    assert report['status']=='complete'
    full=ws.query_results(descriptor='ramachandran')
    assert len(full)>0 and full['value'].null_count==len(full)
    monkeypatch.setattr(virtual_rama_store,'environment',lambda:dict(changed=True))
    with pytest.raises(IOError,match='contract changed'):
        ws.query_results(descriptor='ramachandran')
