"""Exact planar onion prototype, designed for certification not speed claims.

Uses exact integer/Fraction predicates. Hull preprocessing O(N^2 log N)
conservatively; normals binary search in O(log N) exact comparisons.
"""
from __future__ import annotations
from functools import cmp_to_key
import heapq

try:
    from exact import rational
except ImportError:
    from .exact import rational


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def half(a):
    return 0 if a[1] > 0 or (a[1] == 0 and a[0] >= 0) else 1


def angle_cmp(a, b):
    h = half(a) - half(b)
    if h:
        return -1 if h < 0 else 1
    z = cross(a, b)
    return -1 if z > 0 else (1 if z < 0 else 0)


def hull_coordinates(coords):
    """Strict extreme vertex coordinates in CCW order, boundary collinear excluded."""
    v = sorted(set(coords))
    if len(v) <= 2:
        return v

    def chain(seq):
        out = []
        for point in seq:
            while len(out) >= 2:
                x, y = out[-2], out[-1]
                if cross((y[0] - x[0], y[1] - x[1]), (point[0] - y[0], point[1] - y[1])) > 0:
                    break
                out.pop()
            out.append(point)
        return out

    hull = chain(v)[:-1] + chain(reversed(v))[:-1]
    return hull


class OnionIndex:
    def __init__(self, records):
        self.points = [tuple(map(rational, pt)) for pt in records]
        if not self.points or any(len(pt) != 2 for pt in self.points):
            raise ValueError('nonempty planar records required')
        remaining = set(range(len(self.points)))
        self.layers = []
        self.layer_of = {}
        while remaining:
            positions = {}
            for i in remaining:
                positions.setdefault(self.points[i], []).append(i)
            coords = hull_coordinates(positions.keys())
            chosen = [min(positions[pt]) for pt in coords]
            for i in chosen:
                self.layer_of[i] = len(self.layers)
            remaining.difference_update(chosen)
            normals = []
            m = len(chosen)
            if m >= 3:
                for j in range(m):
                    x = self.points[chosen[j]]
                    y = self.points[chosen[(j + 1) % m]]
                    e = (y[0] - x[0], y[1] - x[1])
                    normals.append((e[1], -e[0]))
                normal_idx = sorted(range(m), key=cmp_to_key(lambda i, j: angle_cmp(normals[i], normals[j])))
            else:
                normal_idx = []
            self.layers.append({
                'ids': chosen,
                'normals': normals,
                'normal_idx': normal_idx,
                'positions': {i: j for j, i in enumerate(chosen)}
            })

    def extreme(self, layer, direction):
        a = tuple(map(rational, direction))
        if a == (0, 0):
            raise ValueError('zero direction handled in separate iterator')
        layer = self.layers[layer]
        ids = layer['ids']
        m = len(ids)
        if m <= 2:
            return max(ids, key=lambda i: (self.points[i][0] * a[0] + self.points[i][1] * a[1], -i))
        normals = layer['normals']
        idx = layer['normal_idx']
        l = 0
        r = m
        while l < r:
            mid = (l + r) // 2
            if angle_cmp(normals[idx[mid]], a) < 0:
                l = mid + 1
            else:
                r = mid
        return ids[idx[l % m]]


class OnionIterator:
    def __init__(self, index, direction):
        self.index = index
        self.a = tuple(map(rational, direction))
        if self.a == (0, 0):
            raise ValueError('zero direction')
        self.Q = []
        self.seen = set()
        self.deep = -1
        self.activations = 0
        self.outputs = 0
        self.activate(0)

    def score(self, i):
        x = self.index.points[i]
        return self.a[0] * x[0] + self.a[1] * x[1]

    def push(self, i, t):
        if i not in self.seen:
            self.seen.add(i)
            heapq.heappush(self.Q, (-self.score(i), i, t))

    def activate(self, t):
        self.push(self.index.extreme(t, self.a), t)
        self.deep = t
        self.activations += 1

    def __iter__(self):
        return self

    def __next__(self):
        if not self.Q:
            raise StopIteration
        neg, i, t = heapq.heappop(self.Q)
        if t == self.deep and self.deep + 1 < len(self.index.layers):
            self.activate(self.deep + 1)
        layer = self.index.layers[t]['ids']
        m = len(layer)
        if m > 1:
            pos = self.index.layers[t]['positions'][i]
            for v in {layer[(pos - 1) % m], layer[(pos + 1) % m]}:
                self.push(v, t)
        self.outputs += 1
        return -neg, i


def component_iterator(index, direction):
    """Certified constant stream for zero-direction pieces; onion otherwise."""
    d = tuple(map(rational, direction))
    if d == (0, 0):
        return ((rational(0), i) for i in range(len(index.points)))
    return OnionIterator(index, d)

