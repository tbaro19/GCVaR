"""Optimized planar two-ray skyband screening (index-preserving)."""
from __future__ import annotations
import numpy as np


class Fenwick:
    def __init__(self, n: int):
        self.counts = [0] * (n + 1)
        self.n = n

    def add(self, pos: int):
        i = pos + 1
        while i <= self.n:
            self.counts[i] += 1
            i += i & -i

    def prefix(self, pos: int):
        val = 0
        i = pos + 1
        while i > 0:
            val += self.counts[i]
            i -= i & -i
        return val


def skyband_2d(points, k: int) -> np.ndarray:
    """O(N log N) expected sorting/comparisons, O(N) working space.

    Points can be tuples of Fraction/int/float; no epsilon comparisons.
    Equal coordinates are distinct records, tie priority by ID.
    """
    pts = list(map(tuple, points))
    n = len(pts)
    if not 1 <= k <= n:
        raise ValueError('invalid k')
    if any(len(point) != 2 for point in pts):
        raise ValueError('requires transformed 2d factors')
    ids = sorted(range(n), key=lambda i: (-pts[i][0], -pts[i][1], i))
    coords = sorted(set(p[1] for p in pts), reverse=True)
    loc = {val: j for j, val in enumerate(coords)}
    bit = Fenwick(len(coords))
    kept = []
    for i in ids:
        j = loc[pts[i][1]]
        dominated = bit.prefix(j)
        if dominated < k:
            kept.append(i)
        bit.add(j)
    return np.asarray(sorted(kept), dtype=np.int64)


def cone_transform(factors, rays):
    data = np.asarray(factors, dtype=float)
    rr = np.asarray(rays, dtype=float)
    if data.ndim != 2 or rr.shape != (2, data.shape[1]):
        raise ValueError('rays must be two vectors in factor space')
    return data @ rr.T

