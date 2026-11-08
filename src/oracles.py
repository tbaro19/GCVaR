"""Comparable first-stage baselines; pure-Python exact and NumPy full scan.

SORTED-UNION and SORTED-LAZY explicitly charge their p full per-piece sorts:
they are not represented as geometric index algorithms.
"""
from __future__ import annotations
import heapq
import numpy as np


def affine_scores(x, a, b):
    return np.asarray(x, dtype=float) @ np.asarray(a, dtype=float).T + np.asarray(b, dtype=float)


def max_affine_numpy(x, a, b):
    return np.max(affine_scores(x, a, b), axis=1)


def full_sort(x, a, b, k):
    return np.sort(max_affine_numpy(x, a, b))[::-1][:k].copy()


def full_select(x, a, b, k):
    scores = max_affine_numpy(x, a, b)
    if k == len(scores):
        return np.sort(scores)[::-1]
    return np.sort(np.partition(scores, len(scores) - k)[-k:])[::-1]


def component_union(x, a, b, k):
    """Sorted-array reference for explicit candidate union (p sorts included)."""
    scores = affine_scores(x, a, b)
    n, p = scores.shape
    if not 1 <= k <= n:
        raise ValueError('invalid k')
    ids = set()
    for j in range(p):
        ids.update(sorted(range(n), key=lambda i: (-scores[i, j], i))[:k])
    return np.sort(np.max(scores[sorted(ids)], axis=1))[::-1][:k]


def lazy_merge_from_streams(streams, k):
    """Merge arbitrary streams yielding (score, scenario_id).

    Returns (ordered_topk_values, M primitive removals), without a new
    max-affine evaluation for duplicate scenario identifiers.
    """
    streams = [iter(stream) for stream in streams]
    heap = []
    for j, s in enumerate(streams):
        item = next(s, None)
        if item is not None:
            score, i = item
            heapq.heappush(heap, (-score, i, j))
    used = set()
    out = []
    M = 0
    while len(out) < k:
        if not heap:
            raise ValueError('insufficient scenario records')
        neg, i, j = heapq.heappop(heap)
        M += 1
        if i not in used:
            used.add(i)
            out.append(-neg)
        if len(out) < k:
            item = next(streams[j], None)
            if item is not None:
                score, h = item
                heapq.heappush(heap, (-score, h, j))
    return out, M


def sorted_lazy(x, a, b, k):
    """Baseline with full primitive sort charged, NOT an indexed onion oracle."""
    scores = affine_scores(x, a, b)
    n, p = scores.shape

    def ordered_stream(j):
        ids = sorted(range(n), key=lambda i: (-scores[i, j], i))
        return ((scores[i, j], i) for i in ids)

    streams = [ordered_stream(j) for j in range(p)]
    return lazy_merge_from_streams(streams, k)


def weights_topmean(k):
    return np.ones(k, dtype=float) / k


def weights_mix(k1, k2, lam=0.6):
    if k1 > k2 or k1 < 1:
        raise ValueError('expect 1 <= k1 <= k2')
    w = np.full(k2, (1 - lam) / k2)
    w[:k1] += lam / k1
    return w


def spectral_risk(values, weights):
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.dot(w, vals[:len(w)]))

