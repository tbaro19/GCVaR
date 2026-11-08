"""Exact rational scalar kinetic spectral profile for certification.

Event grouping handles simultaneous times, tied ordinates and identical lines.
No claims of fast end-to-end labeling performance are implied.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
from bisect import bisect_right

try:
    from exact import rational
except ImportError:
    from .exact import rational


def build_profile(intercepts, slopes, weights, T):
    z = list(map(rational, intercepts))
    v = list(map(rational, slopes))
    w = list(map(rational, weights))
    T = rational(T)
    n = len(z)
    if len(v) != n or len(w) != n or T < 0:
        raise ValueError('invalid profile dimension or T')
    if any(w[i] < w[i + 1] for i in range(n - 1)) or min(w) < 0:
        raise ValueError('spectral weights required')
    order = sorted(range(n), key=lambda i: (-z[i], -v[i], i))
    pos = {i: r for r, i in enumerate(order)}
    g = sum((w[r] * z[i] for r, i in enumerate(order)), F(0))
    s = sum((w[r] * v[i] for r, i in enumerate(order)), F(0))
    events = defaultdict(lambda: defaultdict(set))
    for i in range(n):
        for j in range(i + 1, n):
            if v[i] != v[j]:
                t = (z[j] - z[i]) / (v[i] - v[j])
                if 0 < t < T:
                    y = z[i] + v[i] * t
                    events[t][y].update((i, j))
    knots = [(F(0), g, s)]
    last = F(0)
    for t in sorted(events):
        g += s * (t - last)
        # Distinct ordinates at a given time define disjoint contiguous blocks.
        for y, members in events[t].items():
            left = min(pos[i] for i in members)
            right = max(pos[i] for i in members)
            # Grow across permanently coincident trajectories not in pair events.
            while left > 0 and z[order[left - 1]] + v[order[left - 1]] * t == y:
                left -= 1
            while right < n - 1 and z[order[right + 1]] + v[order[right + 1]] * t == y:
                right += 1
            assert all(z[order[r]] + v[order[r]] * t == y for r in range(left, right + 1))
            before = sum((w[r] * v[order[r]] for r in range(left, right + 1)), F(0))
            order[left:right + 1] = sorted(order[left:right + 1], key=lambda i: (-v[i], i))
            for r in range(left, right + 1):
                pos[order[r]] = r
            after = sum((w[r] * v[order[r]] for r in range(left, right + 1)), F(0))
            s += after - before
        if s != knots[-1][2]:
            knots.append((t, g, s))
        last = t
    return knots


def profile_value(knots, t):
    t = rational(t)
    ix = bisect_right(knots, t, key=lambda knot: knot[0]) - 1
    if ix < 0:
        raise ValueError('t precedes interval')
    tau, g, s = knots[ix]
    return g + s * (t - tau)


def dominance_kinetic(a, b, v, w, T):
    pa = build_profile(a, v, w, T)
    pb = build_profile(b, v, w, T)
    ticks = sorted({F(0), rational(T)} | {e[0] for e in pb})
    return all(profile_value(pa, t) <= profile_value(pb, t) for t in ticks), pa, pb


def dominance_direct(a, b, v, w, T):
    a = list(map(rational, a))
    b = list(map(rational, b))
    v = list(map(rational, v))
    w = list(map(rational, w))
    T = rational(T)
    ticks = {F(0), T}
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if v[i] != v[j]:
                t = (b[j] - b[i]) / (v[i] - v[j])
                if 0 < t < T:
                    ticks.add(t)

    def risk(z, t):
        return sum((x * y for x, y in zip(w, sorted((z[i] + v[i] * t for i in range(len(z))), reverse=True))), F(0))

    return all(risk(a, t) <= risk(b, t) for t in ticks)

