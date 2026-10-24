"""Definition, fallback and publication compatibility of optional acceleration."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from minitopol.descriptors import sasa
from minitopol.descriptors.correlation import atom_correlation
from minitopol.descriptors.kernels import _sasa_atoms_python, sasa_atoms
from minitopol.descriptors.results import _payload_hash


@pytest.mark.parametrize('count', [1, 2, 17, 64])
def test_optional_sasa_is_exact_with_original(count):
    positions = np.random.default_rng(count).normal(size=(count, 3)) * 4
    radii = np.full(count, 1.7)
    np.testing.assert_array_equal(sasa_atoms(positions, radii),
                                  _sasa_atoms_python(positions, radii))


def test_coincident_sasa_error_is_preserved():
    with pytest.raises(ValueError, match='Coincident'):
        sasa_atoms(np.zeros((2, 3)), np.full(2, 1.7))


@pytest.mark.parametrize('count', [True, 2.5])
def test_unusual_sample_count_keeps_original_error(count):
    arguments = dict(sphere_count=count)
    positions, radii = np.zeros((1, 3)), np.array([1.7])
    with pytest.raises(TypeError) as original:
        _sasa_atoms_python(positions, radii, **arguments)
    with pytest.raises(TypeError) as accelerated:
        sasa_atoms(positions, radii, **arguments)
    assert str(accelerated.value) == str(original.value)


def test_zero_sample_count_keeps_numpy_result():
    with np.errstate(all='ignore'):
        np.testing.assert_array_equal(
            sasa_atoms(np.zeros((1, 3)), np.array([1.7]), sphere_count=0),
            _sasa_atoms_python(np.zeros((1, 3)), np.array([1.7]), sphere_count=0))


def test_float32_radii_and_list_positions_keep_original_arithmetic():
    positions = np.random.default_rng(9).normal(size=(17, 3))
    for xyz, radii in [(positions, np.full(17, 1.7, dtype=np.float32)),
                       (positions.tolist(), np.full(17, 1.7))]:
        np.testing.assert_array_equal(sasa_atoms(xyz, radii), _sasa_atoms_python(xyz, radii))


def test_all_lags_and_time_origins_against_direct_sums(tmp_path):
    positions = np.random.default_rng(5).normal(size=(17, 5, 3))
    mean = positions.mean(axis=0)
    context = SimpleNamespace(
        n=17, m=5, fft_block=17, budget=128 * 1024**2,
        read_aligned=lambda start, stop, points: positions[start:stop][:, points, :])
    for point in [0, 1, 4]:
        dots, squares = atom_correlation(context, point, mean[point], tmp_path/str(point))
        centered = positions[:, point] - mean[point]
        expected = np.array([sum(float(np.dot(centered[t], centered[t+lag]))
                                 for t in range(17-lag)) for lag in range(17)])
        np.testing.assert_allclose(dots[:], expected, atol=5e-14, rtol=0)
        direct = np.r_[0., np.cumsum([float(np.dot(row, row)) for row in centered])]
        np.testing.assert_allclose(squares[:], direct, atol=5e-14, rtol=0)


def test_column_tables_keep_legacy_part_keys_and_payload_hashes():
    # Cross multiple original 4096-row boundaries for both atom and residue rows.
    count = 17000
    values = np.arange(count, dtype=float) * .125
    residue_index = list(range(count))
    res_atoms = {i: [i] for i in range(count)}
    expected = []
    for part, start in enumerate(range(0, count, 4096)):
        rows = [dict(frame=0, time_ps=2.5, level='atom', point=i,
                     residue=i, value=values[i]) for i in range(start, min(start+4096, count))]
        expected.append((f'atom-0-{part}', pa.Table.from_pylist(rows, schema=sasa.SCHEMA)))
    for start in range(0, count, 4096):
        rows = [dict(frame=0, time_ps=2.5, level='residue', point=None,
                     residue=i, value=values[[i]].sum()) for i in range(start, min(start+4096, count))]
        expected.append((f'residue-0-{start}', pa.Table.from_pylist(rows, schema=sasa.SCHEMA)))
    expected.append(('total-0', pa.Table.from_pylist([
        dict(frame=0, time_ps=2.5, level='total', point=None,
             residue=None, value=values.sum())], schema=sasa.SCHEMA)))
    actual = list(sasa.output_tables(0, 2.5, values, residue_index, res_atoms))
    for (oldkey, old), (newkey, new) in zip(expected, actual, strict=True):
        assert oldkey == newkey
        assert _payload_hash(old) == _payload_hash(new)


def test_optional_numba_absence_keeps_python_and_mcp_imports():
    source = Path(__file__).parents[1]/'src'
    script = '''
import importlib.abc,json,sys
class WithoutNumba(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='numba' or fullname.startswith('numba.'):
            raise ModuleNotFoundError('Optional dependency deliberately absent',name='numba')
sys.meta_path.insert(0,WithoutNumba())
import numpy as np
from minitopol.descriptors.kernels import sasa_atoms,_sasa_atoms_python
from minitopol.descriptors import contacts
from minitopol.mcp import server
xyz=np.array([[0.,0.,0.],[20.,0.,0.]])
radii=np.array([1.7,1.7])
np.testing.assert_array_equal(sasa_atoms(xyz,radii),_sasa_atoms_python(xyz,radii))
print(json.dumps(dict(fallback=True,mcp_imported=True)))
'''
    result = subprocess.run([sys.executable, '-c', script], check=True,
                            capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(source)))
    assert json.loads(result.stdout) == dict(fallback=True, mcp_imported=True)


def test_contact_scratch_corruption_fails_before_publication():
    import sqlite3
    pytest.importorskip('numba')
    from minitopol.descriptors.spatial_contacts import _frame_hash, _verify_scratch
    with sqlite3.connect(':memory:') as con:
        con.execute('CREATE TABLE states(bi,bj,frame,r,s,d)')
        con.execute('INSERT INTO states VALUES (0,0,0,1,2,3.0)')
        con.commit()
        _verify_scratch(con, [_frame_hash([(1, 2, 3.)])])
        con.execute('UPDATE states SET d=4.0')
        con.commit()
        with pytest.raises(IOError, match='checksum'):
            _verify_scratch(con, [_frame_hash([(1, 2, 3.)])])
