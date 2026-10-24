"""Bounded scratch resources and exact memory-guard counting, including edges."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial import cKDTree
from minitopol.descriptors import spatial_contacts as spool


@pytest.mark.parametrize('kind',['coincident','random','cutoff'])
def test_dual_count_preserves_memory_guard(kind):
    if kind=='coincident':xyz=np.zeros((31,3))
    elif kind=='random':xyz=np.random.default_rng(932).normal(size=(100,3))*5
    else:
        radius=4.50001
        xyz=np.array([[0.,0.,0.],[radius,0,0],[np.nextafter(radius,0),0,0],
                      [np.nextafter(radius,np.inf),0,0],[-radius,0,0]])
    tree=cKDTree(xyz)
    assert int(tree.count_neighbors(tree,4.50001))==int(tree.query_ball_point(xyz,4.50001,return_length=True).sum())


def test_handle_limit_uses_previous_compatible_path(monkeypatch):
    ctx=SimpleNamespace(res_atoms={},atoms={'residue_index':[]},annotation={'alpha_carbons':[]},
                        residues=list(range(1025)),pair_block=32)
    sentinel=object();calls=[]
    def original(context,writer):calls.append((context,writer));return sentinel
    monkeypatch.setattr(spool,'original',original)
    writer=object()
    assert spool.calculate(ctx,writer) is sentinel and calls==[(ctx,writer)]
