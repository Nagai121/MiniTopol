"""Strict-math CPU kernels preserving sampling and scientific definitions."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def atom_sasa(center, positions, neighbors, points, inflated, atom):
    exposed = np.ones(len(points), np.bool_)
    surface = center + points * inflated[atom]
    count = len(points)
    for other in neighbors:
        if other == atom:
            continue
        if (center[0] == positions[other, 0] and center[1] == positions[other, 1]
                and center[2] == positions[other, 2]):
            raise ValueError("Coincident heavy atoms make per-atom SASA assignment ambiguous")
        radius2 = inflated[other] ** 2
        for p in range(len(points)):
            if exposed[p]:
                x = surface[p, 0] - positions[other, 0]
                y = surface[p, 1] - positions[other, 1]
                z = surface[p, 2] - positions[other, 2]
                if x*x + y*y + z*z < radius2:
                    exposed[p] = False
                    count -= 1
        if count == 0:
            break
    return count


@njit(cache=True, fastmath=False)
def residue_minima(xyz, pairs, atom_residue, allowed, cutoff=4.5):
    """Match MDAnalysis float32 displacement, float64 distance arithmetic."""
    n = len(allowed)
    minima = np.full((n,n), np.inf)
    for pair in pairs:
        i,j = pair[0],pair[1]
        r,s = atom_residue[i],atom_residue[j]
        if r > s:
            r,s = s,r
        if not allowed[r,s]:
            continue
        x = np.float64(np.float32(xyz[i,0]-xyz[j,0]))
        y = np.float64(np.float32(xyz[i,1]-xyz[j,1]))
        z = np.float64(np.float32(xyz[i,2]-xyz[j,2]))
        distance = np.sqrt(x*x+y*y+z*z)
        if distance <= cutoff and distance < minima[r,s]:
            minima[r,s] = distance
    return minima
