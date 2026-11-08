"""Independent exact arithmetic reference procedures for GCVaR.

All exact certificates use Python Fraction. Scenario IDs preserve duplicates.
"""
from __future__ import annotations
from fractions import Fraction as F
from typing import Sequence


def rational(value):
    if isinstance(value, F):
        return value
    return F(str(value))


def ordered_exact(losses: Sequence):
    return sorted([rational(x) for x in losses], reverse=True)


def topk_exact(losses: Sequence, k: int):
    if not 1 <= k <= len(losses):
        raise ValueError('k outside [1,N]')
    return ordered_exact(losses)[:k]


def max_affine_exact(scenarios, slopes, intercepts):
    if len(slopes) != len(intercepts) or not slopes:
        raise ValueError('slopes and intercepts mismatch')
    out = []
    for x in scenarios:
        out.append(max(sum(rational(a) * rational(v) for a, v in zip(s, x)) + rational(b)
                       for s, b in zip(slopes, intercepts)))
    return out


def skyband_2d_exact(points, k):
    """A literal O(N^2) independent reference, not the optimized algorithm."""
    pts = [(rational(x), rational(y)) for x, y in points]
    if not 1 <= k <= len(pts):
        raise ValueError('invalid k')
    ids = sorted(range(len(pts)), key=lambda i: (-pts[i][0], -pts[i][1], i))
    retained = []
    for pos, i in enumerate(ids):
        d = sum(pts[h][0] >= pts[i][0] and pts[h][1] >= pts[i][1]
                for h in ids[:pos])
        if d < k:
            retained.append(i)
    return sorted(retained)


def topmean_exact(losses, k):
    return sum(topk_exact(losses, k), F(0)) / k


def cvar_fractional_exact(losses, alpha):
    n = len(losses)
    q = (F(1) - rational(alpha)) * n
    if not 0 < q <= n:
        raise ValueError('invalid alpha')
    m = q.numerator // q.denominator
    theta = q - m
    values = ordered_exact(losses)
    s = sum(values[:m], F(0))
    if theta:
        s += theta * values[m]
    return s / q


def spectral_exact(losses, weights):
    if not 1 <= len(weights) <= len(losses):
        raise ValueError('weights support invalid')
    return sum((rational(w) * y for w, y in zip(weights, topk_exact(losses, len(weights)))), F(0))


def exact_support_cut(x0, factors, coefficients, weights):
    """Return rational cut (constant, slope), with H(x)=constant+slope@x.

    L_i(x)=max(0, coefficient[j] dot factors[i] - x_j).
    """
    x0 = tuple(map(rational, x0))
    factors = [tuple(map(rational, x)) for x in factors]
    coeff = [tuple(map(rational, c)) for c in coefficients]
    k = len(weights)
    scores = []
    for i, vec in enumerate(factors):
        vals = [F(0)] + [sum(c * t for c, t in zip(row, vec)) - x0[j]
                         for j, row in enumerate(coeff)]
        j = max(range(len(vals)), key=lambda j: (vals[j], -j))
        scores.append((vals[j], i, j))
    scores.sort(key=lambda t: (-t[0], t[1]))
    slope = [F(0)] * len(coeff)
    const = F(0)
    for w, (_, i, j) in zip(weights, scores[:k]):
        w = rational(w)
        if j:
            const += w * sum(u * v for u, v in zip(coeff[j - 1], factors[i]))
            slope[j - 1] -= w
    return const, tuple(slope)


def eval_cut(cut, x):
    c, slope = cut
    return c + sum(a * rational(b) for a, b in zip(slope, x))

