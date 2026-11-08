"""Independent extensive-form LP and finite cutting-plane OA baselines.

Model: demand_ij = A_j @ factors_i, L_i(x)=max(0,max_j(demand_ij-x_j)).
Weights are a convex combination of top-k means with fixed ORIGINAL k.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass
class Solution:
    x: np.ndarray
    objective: float
    original_risk: float
    status: str
    iterations: int
    wall_s: float
    lp_rows: int
    lp_cols: int
    cuts: int


def instance_demands(factors, coeff):
    factors = np.asarray(factors, dtype=float)
    coeff = np.asarray(coeff, dtype=float)
    if factors.ndim != 2 or coeff.ndim != 2 or factors.shape[1] != coeff.shape[1]:
        raise ValueError('invalid dimensions')
    if (coeff < 0).any():
        raise ValueError('common positive cone violated')
    return factors @ coeff.T


def top_risk_from_demands(demand, x, means):
    L = np.maximum(0., np.max(demand - np.asarray(x)[None, :], axis=1))
    vals = np.sort(L)[::-1]
    return sum(lam * float(vals[:k].mean()) for k, lam in means)


def weighted_ranks(n, means):
    kmax = max(k for k, _ in means)
    w = np.zeros(kmax)
    for k, lam in means:
        if not 1 <= k <= n or lam < 0:
            raise ValueError('invalid supported topmean')
        w[:k] += lam / k
    if not np.isclose(sum(lam for _, lam in means), 1, atol=1e-12):
        raise ValueError('mixture weights do not sum to 1')
    return w


def bound_vector(demand):
    return np.maximum(1., np.max(demand, axis=0) + 10.)


def solve_extensive(demand, cost, means, budget, ub=None, *, solver_options=None):
    """Build exact mathematical extensive LP (HiGHS numerical solution).

    Each top-k term: min_t[t+1/k sum_i (u_i-t)_+]. Core rows use
    unchanged original k, and no artificial mass on removed scenarios.
    """
    tic = perf_counter()
    D = np.asarray(demand, float)
    n, q = D.shape
    c = np.asarray(cost, float)
    assert c.shape == (q,)
    if ub is None:
        ub = bound_vector(D)
    else:
        ub = np.asarray(ub, float)
    weighted_ranks(n, means)
    g = len(means)
    t0 = q + n
    s0 = t0 + g
    total = s0 + g * n
    obj = np.zeros(total)
    obj[:q] = c
    bounds = [(0, float(ub[j])) for j in range(q)] + [(0, None)] * n + [(None, None)] * g + [(0, None)] * (g * n)
    rows = []
    cols = []
    values = []
    rhs = []

    def row(terms, b):
        r = len(rhs)
        for j, val in terms:
            if val:
                rows.append(r)
                cols.append(j)
                values.append(val)
        rhs.append(b)

    # demand_ij - x_j <= u_i
    for i in range(n):
        for j in range(q):
            row([(j, -1), (q + i, -1)], -D[i, j])

    # u_i - t_g <= s_ig
    for gidx in range(g):
        for i in range(n):
            row([(q + i, 1), (t0 + gidx, -1), (s0 + gidx * n + i, -1)], 0.)

    row([(t0 + gidx, lam) for gidx, (k, lam) in enumerate(means)] + [
        (s0 + gidx * n + i, lam / k) for gidx, (k, lam) in enumerate(means) for i in range(n)], budget)

    A = coo_matrix((values, (rows, cols)), shape=(len(rhs), total)).tocsr()
    res = linprog(obj, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method='highs', options=solver_options)
    if not res.success:
        raise RuntimeError(f'extensive LP: {res.message}')
    x = res.x[:q]
    return Solution(x, float(c @ x), top_risk_from_demands(D, x, means), 'optimal_numeric', int(res.nit),
                    perf_counter() - tic, len(rhs), total, 0)


def separate_supporting_cut(D, x, means):
    n, q = D.shape
    w = weighted_ranks(n, means)
    losses = np.maximum(0., np.max(D - x[None, :], axis=1))
    ids = sorted(range(n), key=lambda i: (-losses[i], i))[:len(w)]
    Hc = 0.
    slope = np.zeros(q)
    for r, i in enumerate(ids):
        # Choose zero branch on ties to ensure a mathematically active affine piece.
        j = int(np.argmax(D[i] - x))
        if D[i, j] - x[j] > 0:
            Hc += w[r] * D[i, j]
            slope[j] -= w[r]
    return Hc, slope, float(np.dot(w, losses[ids]))


def solve_outer_approx(demand, cost, means, budget, ub=None, *, tol=1e-7, max_iter=250, solver_options=None):
    tic = perf_counter()
    D = np.asarray(demand, dtype=float)
    n, q = D.shape
    c = np.asarray(cost, float)
    if ub is None:
        ub = bound_vector(D)
    weighted_ranks(n, means)
    cuts = []
    for iteration in range(max_iter + 1):
        if cuts:
            A = np.stack([item[1] for item in cuts])
            b = np.array([budget - item[0] for item in cuts])
        else:
            A = None
            b = None
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, float(e)) for e in ub], method='highs', options=solver_options)
        if not res.success:
            raise RuntimeError(f'OA master: {res.message}')
        x = res.x
        Hc, slope, risk = separate_supporting_cut(D, x, means)
        if risk <= budget + tol:
            return Solution(x, float(c @ x), top_risk_from_demands(D, x, means), 'optimal_numeric', iteration + 1,
                            perf_counter() - tic, len(cuts), q, len(cuts))
        if iteration == max_iter:
            raise RuntimeError('max_iter reached')
        cuts.append((Hc, slope))
    raise AssertionError('unreachable')


def fractional_cvar_topmeans(original_n, alpha):
    """Fractional empirical CVaR as convex mixture of adjacent top means.

    This returns coefficients based on N_original, never core size.
    """
    from fractions import Fraction as F
    q = (F(1) - F(str(alpha))) * original_n
    if not 0 < q <= original_n:
        raise ValueError('alpha must be in [0,1)')
    m = q.numerator // q.denominator
    theta = q - m
    if theta == 0:
        return [(m, 1.)]
    if m == 0:
        return [(1, 1.)]
    return [(m, float(m * (1 - theta) / q)), (m + 1, float((m + 1) * theta / q))]

