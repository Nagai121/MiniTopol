"""Native whole-molecule/NoJump coordinate reconstruction.

Only verified chemical bonds are used. Disconnected heavy-atom groups are
tracked separately rather than inventing chemical connections.
"""
import numpy as np
from numba import njit


def bond_forest(atom_count, bonds):
    """Static parent-before-child traversal of the A heavy-atom bond graph."""
    if type(atom_count) is not int or atom_count < 1:
        raise ValueError("An A trajectory needs at least one atom")
    adjacency = [[] for _ in range(atom_count)]
    for i, j in np.asarray(bonds, dtype=np.int64).reshape(-1, 2):
        i, j = int(i), int(j)
        if i == j or i < 0 or j < 0 or i >= atom_count or j >= atom_count:
            raise ValueError("Invalid chemical-bond graph")
        adjacency[i].append(j)
        adjacency[j].append(i)
    order = []
    parents = []
    offsets = [0]
    seen = np.zeros(atom_count, dtype=np.bool_)
    for root in range(atom_count):
        if seen[root]:
            continue
        seen[root] = True
        queue = [root]
        q = 0
        order.append(root)
        parents.append(-1)
        while q < len(queue):
            current = queue[q]
            q += 1
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
                    order.append(neighbor)
                    parents.append(current)
        offsets.append(len(order))
    return (np.asarray(order, dtype=np.int64),
            np.asarray(parents, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64))


@njit(cache=True, nogil=True)
def reconstruct_frame(xyz, box, inverse, order, parents, offsets, previous, first):
    """Make bonded components whole and follow their geometric centers.

    xyz is a writable float64 frame. previous holds a fractional-space center
    for every bonded component; both are updated in place. A triclinic box
    uses a local 27-image search for each chemical bond.
    """
    n = len(xyz)
    reduced = np.empty((n, 3), dtype=np.float64)
    whole = np.empty((n, 3), dtype=np.float64)
    for a in range(n):
        for axis in range(3):
            reduced[a, axis] = (xyz[a, 0] * inverse[0, axis]
                                + xyz[a, 1] * inverse[1, axis]
                                + xyz[a, 2] * inverse[2, axis])
    triclinic = box[1, 0] != 0.0 or box[2, 0] != 0.0 or box[2, 1] != 0.0
    for component in range(len(offsets) - 1):
        start, stop = offsets[component], offsets[component + 1]
        root = order[start]
        for axis in range(3):
            whole[root, axis] = reduced[root, axis]
        for t in range(start + 1, stop):
            a, parent = order[t], parents[t]
            d0 = reduced[a, 0] - reduced[parent, 0]
            d1 = reduced[a, 1] - reduced[parent, 1]
            d2 = reduced[a, 2] - reduced[parent, 2]
            c0, c1, c2 = d0 - np.rint(d0), d1 - np.rint(d1), d2 - np.rint(d2)
            if triclinic:
                best = np.inf
                win0, win1, win2 = c0, c1, c2
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            x, y, z = c0-i, c1-j, c2-k
                            ux = x*box[0, 0] + y*box[1, 0] + z*box[2, 0]
                            uy = y*box[1, 1] + z*box[2, 1]
                            uz = z*box[2, 2]
                            d2_cart = ux*ux + uy*uy + uz*uz
                            if d2_cart < best:
                                best = d2_cart
                                win0, win1, win2 = x, y, z
                c0, c1, c2 = win0, win1, win2
            whole[a, 0] = whole[parent, 0] + c0
            whole[a, 1] = whole[parent, 1] + c1
            whole[a, 2] = whole[parent, 2] + c2
        mean0, mean1, mean2 = 0.0, 0.0, 0.0
        for t in range(start, stop):
            a = order[t]
            mean0 += whole[a, 0]
            mean1 += whole[a, 1]
            mean2 += whole[a, 2]
        count = stop - start
        mean0 /= count
        mean1 /= count
        mean2 /= count
        shift0, shift1, shift2 = 0.0, 0.0, 0.0
        if not first:
            shift0 = np.rint(mean0 - previous[component, 0])
            shift1 = np.rint(mean1 - previous[component, 1])
            shift2 = np.rint(mean2 - previous[component, 2])
        previous[component, 0] = mean0 - shift0
        previous[component, 1] = mean1 - shift1
        previous[component, 2] = mean2 - shift2
        for t in range(start, stop):
            a = order[t]
            x = whole[a, 0] - shift0
            y = whole[a, 1] - shift1
            z = whole[a, 2] - shift2
            for axis in range(3):
                xyz[a, axis] = x*box[0, axis] + y*box[1, axis] + z*box[2, axis]
