import numpy as np
import pytest
from minitopol.descriptors.precision64 import backbone_angles64
from minitopol.descriptors.kernels import backbone_angles

QUADS={'phi':[[0,1,2,3]],'psi':[None]}
REASONS={'phi':[None],'psi':['segment_terminal']}
def reference(p):
    b0=p[0]-p[1];b1=p[2]-p[1];b2=p[3]-p[2]
    unit=b1/np.linalg.norm(b1);v=b0-np.dot(b0,unit)*unit;w=b2-np.dot(b2,unit)*unit
    return (np.degrees(np.arctan2(np.dot(np.cross(unit,v),w),np.dot(v,w)))+180)%360-180

@pytest.mark.parametrize('sign',[-1,1])
def test_signed_dihedral(sign):
    p=np.array([[1,0,0],[0,0,0],[0,1,0],[0,1,sign]],dtype=np.float32)
    value,why=backbone_angles64(p,np.eye(3),QUADS,REASONS,periodic=False)
    assert value['phi'][0]==-sign*90
    assert np.isnan(value['psi'][0]) and why['psi']==['segment_terminal']

def test_middle_rounding_improves():
    p=np.array([[.00001,0,0],[1,.00002,0],[1.1,1,0],[.01,1.3,.9]],dtype=np.float32)
    expected=reference(p.astype(float))
    old,_=backbone_angles(p,np.eye(3),QUADS,REASONS,periodic=False)
    new,_=backbone_angles64(p,np.eye(3),QUADS,REASONS,periodic=False)
    assert abs(new['phi'][0]-expected)<1e-11
    assert abs(new['phi'][0]-expected)<abs(old['phi'][0]-expected)

def test_periodic_vector_meaning_unchanged():
    p=np.array([[9,4,2],[9.75,4.125,2],[10.25,5.25,2.25],[10.75,5.75,3.25]],dtype=np.float32)
    wrapped=np.mod(p,10);box=np.eye(3,dtype=np.float32)*10
    value,_=backbone_angles64(wrapped,box,QUADS,REASONS,periodic=True)
    assert abs(value['phi'][0]-reference(p.astype(float)))<1e-11

def test_degenerate_reason():
    p=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype=np.float32)
    value,why=backbone_angles64(p,np.eye(3),QUADS,REASONS,periodic=False)
    assert np.isnan(value['phi'][0]) and why['phi']==['degenerate_dihedral']
