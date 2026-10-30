"""Bounded batched Kabsch; strict comparison against scalar fit is required."""
import numpy as np
def fit_positions_batch(positions,reference,ca):
    moving=np.ascontiguousarray(positions,dtype=np.float64)
    if moving.ndim!=3 or len(moving)>16:raise ValueError('Fit batch requires at most16frames')
    if len(ca)<3:raise ValueError('At least three non-collinear, unambiguous C-alpha atoms are required')
    p=moving[:,ca];q=np.asarray(reference,dtype=np.float64)[ca]
    pc=p.mean(axis=1);qc=q.mean(axis=0);p=p-pc[:,None,:];q=q-qc
    if np.any(np.linalg.matrix_rank(p)<2) or np.linalg.matrix_rank(q)<2:raise ValueError('C-alpha alignment is undefined for a collinear or degenerate configuration')
    u,_,vt=np.linalg.svd(np.swapaxes(p,1,2)@q)
    sign=np.tile(np.eye(3),(len(moving),1,1));sign[:,-1,-1]=np.sign(np.linalg.det(u@vt))
    return (moving-pc[:,None,:])@(u@sign@vt)+qc
