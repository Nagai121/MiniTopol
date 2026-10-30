import numpy as np
import pytest
from scipy.spatial import cKDTree
pytest.importorskip("numba")
from minitopol.descriptors import csr_sasa,knn_sasa
from minitopol.descriptors.kernels import sphere_points

@pytest.mark.parametrize('dtype',['f4','f8'])
@pytest.mark.parametrize('seed',[1,2,3])
def test_reused_buffers_preserve_every_accessible_count(dtype,seed):
    xyz=(np.random.default_rng(seed).normal(size=(65,3))*4).astype(dtype);inflated=np.resize(np.array([1.7,1.55,1.52,1.8])+1.4,len(xyz));points=sphere_points(960);tree=cKDTree(xyz);lists=tree.query_ball_point(xyz,inflated+inflated.max(),return_sorted=False);offsets=np.r_[0,np.cumsum([len(a) for a in lists])];neighbors=np.concatenate(lists).astype('i8')
    expected=csr_sasa.counts(xyz,inflated,points,offsets,neighbors,True,True)
    for _ in range(3):assert np.array_equal(csr_sasa.count_reusing(xyz,inflated,points,offsets,neighbors,128*1024**2),expected)

def test_float64_boundary_neighbors_retain_exact_query():
    inflated=np.array([3.1]*7);edge=6.2
    xyz=np.zeros((7,3));xyz[:,0]=[0.,edge,np.nextafter(edge,-np.inf),np.nextafter(edge,np.inf),20.,30.,40.]
    built=knn_sasa._neighbors(xyz,inflated,128*1024**2);assert built is not None
    offsets,neighbors,_=built;tree=cKDTree(xyz)
    for atom in range(len(xyz)):
        expected=tree.query_ball_point(xyz[atom],inflated[atom]+inflated.max())
        assert set(neighbors[offsets[atom]:offsets[atom+1]])==set(expected)

def test_duplicates_and_exotic_tiny_coordinates_keep_fallback():
    radius=np.array([3.1,3.1]);assert knn_sasa._neighbors(np.zeros((2,3)),radius,128*1024**2) is None
    xyz=np.array([[1e-100,0,0],[10.,0,0]]);assert knn_sasa._neighbors(xyz,radius,128*1024**2) is None
