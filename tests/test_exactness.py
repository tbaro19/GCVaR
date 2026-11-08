"""Exactness and Mathematical Verification Suite for GCVaR.

Verifies:
1. Exact top-k degeneracy fuzzing (>=10,500 tests).
2. Uniform planar 2D skyband screening correctness.
3. Cone eligibility and cone invariance.
4. Adversarial onion iterator and convex layers.
5. Numerical LP equivalence: EF-F == EF-C == OA-F == OA-C (zero optimality gap).
6. Operational dataset adapters (Bike, Tetouan, Electricity).
"""
import sys
import random
from fractions import Fraction as F
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest

from exact import (max_affine_exact, topk_exact, spectral_exact, skyband_2d_exact,
                   cvar_fractional_exact, exact_support_cut, eval_cut)
from core import skyband_2d
from onion import OnionIndex, OnionIterator, component_iterator
from oracles import (full_sort, full_select, component_union, sorted_lazy,
                     lazy_merge_from_streams, weights_mix)
from kinetic import dominance_kinetic, dominance_direct, build_profile, profile_value
from optimization import (instance_demands, solve_extensive, solve_outer_approx,
                         top_risk_from_demands, bound_vector, fractional_cvar_topmeans)
from datasets import load_local, chronological_split


def test_exact_oracle_degeneracy_fuzz():
    # >=10k independent input/direction combinations across ties/duplicates.
    trials = 10500
    for seed in range(trials):
        rng = random.Random(seed)
        N = rng.randrange(1, 19)
        p = rng.randrange(1, 7)
        k = rng.randrange(1, N + 1)
        pts = [(rng.randrange(-3, 4), rng.randrange(-3, 4)) for _ in range(N)]
        slopes = [(rng.randrange(-3, 4), rng.randrange(-3, 4)) for _ in range(p)]
        offsets = [rng.randrange(-3, 4) for _ in range(p)]
        ref = max_affine_exact(pts, slopes, offsets)
        keys = topk_exact(ref, k)
        sets = []
        for a, b in zip(slopes, offsets):
            component = [a[0] * x + a[1] * y + b for x, y in pts]
            sets.append(sorted(range(N), key=lambda i: (-component[i], rng.random()))[:k])
        union = set().union(*map(set, sets))
        assert topk_exact([ref[i] for i in union], k) == keys
        streams = []
        for a, b in zip(slopes, offsets):
            sorted_ids = sorted(range(N), key=lambda i: (-(a[0] * pts[i][0] + a[1] * pts[i][1] + b), i))
            streams.append([(a[0] * pts[i][0] + a[1] * pts[i][1] + b, i) for i in sorted_ids])
        merged, M = lazy_merge_from_streams(streams, k)
        assert merged == keys
        assert k <= M <= p * (k - 1) + 1


def test_core_chain_antichain_and_duplicates():
    for n in (1, 2, 8, 20):
        for k in range(1, n + 1):
            chain = [(i, i) for i in range(n)]
            antichain = [(i, n - i) for i in range(n)]
            dup = [(0, 0)] * n
            assert len(skyband_2d(chain, k)) == k
            assert len(skyband_2d(antichain, k)) == n
            assert len(skyband_2d(dup, k)) == k


def test_uniform_core_exact_rational():
    for seed in range(450):
        rng = random.Random(seed)
        n = rng.randrange(1, 40)
        k = rng.randrange(1, n + 1)
        pts = [(rng.randrange(-5, 6), rng.randrange(-5, 6)) for _ in range(n)]
        fast = skyband_2d(pts, k).tolist()
        independent = skyband_2d_exact(pts, k)
        assert fast == independent
        for _ in range(4):
            a = [(rng.randrange(0, 5), rng.randrange(0, 5)) for j in range(rng.randrange(1, 5))]
            b = [rng.randrange(-8, 9) for j in range(len(a))]
            full = max_affine_exact(pts, a, b)
            subset = [full[i] for i in fast]
            assert topk_exact(full, k) == topk_exact(subset, k)


def test_cone_ineligibility_is_real():
    points = [(0, 0), (1, 1)]
    kept = skyband_2d(points, 1).tolist()
    assert kept == [1]
    # A slope outside the cone reverses which scenario is worst.
    loss = max_affine_exact(points, [(-1, -1)], [0])
    assert topk_exact(loss, 1) != topk_exact([loss[i] for i in kept], 1)


def test_onion_iterator_adversarial():
    for seed in range(125):
        rng = random.Random(seed)
        n = rng.randrange(1, 65)
        pts = [(rng.randrange(-8, 9), rng.randrange(-8, 9)) for _ in range(n)]
        idx = OnionIndex(pts)
        for rep in range(5):
            a = (rng.randrange(-5, 6), rng.randrange(-5, 6))
            if a == (0, 0):
                continue
            results = list(OnionIterator(idx, a))
            assert len(results) == n
            assert len(set(i for score, i in results)) == n
            assert [score for score, i in results] == sorted([a[0] * x + a[1] * y for x, y in pts], reverse=True)
            for lid in range(len(idx.layers)):
                v = idx.extreme(lid, a)
                assert all(a[0] * idx.points[v][0] + a[1] * idx.points[v][1] >= a[0] * idx.points[h][0] + a[1] * idx.points[h][1]
                           for h in idx.layers[lid]['ids'])


def test_floating_baselines_ties():
    rng = np.random.default_rng(912)
    for seed in range(36):
        n = int(rng.integers(10, 170))
        p = int(rng.integers(1, 8))
        k = int(rng.integers(1, n + 1))
        x = rng.integers(-3, 4, size=(n, 2))
        a = rng.integers(-5, 6, size=(p, 2))
        b = rng.integers(-5, 6, size=p)
        ref = full_sort(x, a, b, k)
        for vals in (full_select(x, a, b, k), component_union(x, a, b, k), sorted_lazy(x, a, b, k)[0]):
            assert np.array_equal(ref, vals)


def test_fractional_cvar_and_spectral():
    assert cvar_fractional_exact([10, 8, 4, 0, 0], F('0.7')) == F(28, 3)  # q=1.5
    assert spectral_exact([10, 8, 4, 0], [F(1, 2), F(1, 3), F(1, 6)]) == F(25, 3)


def test_supporting_cut_exact_grid():
    pts = [(3, 4), (7, 2), (4, 9), (3, 4)]
    coeff = [(1, 0), (0, 1), (1, 1)]
    weights = [F(2, 3), F(1, 3)]
    for x0 in [(0, 0, 0), (4, 5, 2), (10, 1, 2)]:
        cut = exact_support_cut(x0, pts, coeff, weights)

        def losses(x):
            return [max([F(0)] + [F(a * u + b * v) - F(x[j]) for j, (a, b) in enumerate(coeff)]) for u, v in pts]

        assert eval_cut(cut, x0) == spectral_exact(losses(x0), weights)
        for x in [(a, b, c) for a in (0, 1, 5) for b in (0, 4) for c in (0, 5)]:
            assert eval_cut(cut, x) <= spectral_exact(losses(x), weights)


def test_kinetic_rational_against_direct():
    for seed in range(140):
        rng = random.Random(seed)
        n = rng.randrange(2, 11)
        a = [rng.randrange(-4, 5) for _ in range(n)]
        b = [rng.randrange(-4, 5) for _ in range(n)]
        v = [rng.randrange(-3, 4) for _ in range(n)]
        w = [F(n - i, n * (n + 1) // 2) for i in range(n)]
        ans, pa, pb = dominance_kinetic(a, b, v, w, 4)
        assert ans == dominance_direct(a, b, v, w, 4)
        for j in range(5):
            t = F(rng.randrange(0, 400), 100)
            ref = sum((weight * y for weight, y in zip(w, sorted([a[i] + v[i] * t for i in range(n)], reverse=True))), F(0))
            assert profile_value(pa, t) == ref


@pytest.mark.parametrize('seed', [801, 802, 803])
@pytest.mark.parametrize('means', [[(5, 1.)], [(5, .6), (10, .4)]])
def test_ef_oa_core_equivalence(seed, means):
    rng = np.random.default_rng(seed)
    factors = rng.gamma(3., 18., size=(55, 2))
    coeff = rng.uniform(.2, 1.3, size=(10, 2))
    cost = rng.uniform(.8, 1.2, size=10)
    D = instance_demands(factors, coeff)
    core = skyband_2d(factors, max(k for k, lam in means))
    ub = bound_vector(D)
    B = 60.
    ef = solve_extensive(D, cost, means, B, ub=ub)
    oa = solve_outer_approx(D, cost, means, B, ub=ub)
    efc = solve_extensive(D[core], cost, means, B, ub=ub)
    oac = solve_outer_approx(D[core], cost, means, B, ub=ub)
    for solution in (oa, efc, oac):
        assert abs(ef.objective - solution.objective) <= 1e-5
        assert top_risk_from_demands(D, solution.x, means) <= B + 1e-6


def test_zero_direction_component_stream():
    pts = [(0, 0), (2, 3), (2, 3), (4, -1)]
    idx = OnionIndex(pts)
    zero = list(component_iterator(idx, (0, 0)))
    assert zero == [(F(0), i) for i in range(len(pts))]
    streams = [((s + 4, i) for s, i in zero), component_iterator(idx, (1, 0))]
    merged, M = lazy_merge_from_streams(streams, 3)
    vals = max_affine_exact(pts, [(0, 0), (1, 0)], [4, 0])
    assert merged == topk_exact(vals, 3)


def test_bike_local_adapter(tmp_path):
    src = tmp_path / 'hour.csv'
    src.write_text('dteday,hr,casual,registered,cnt\n2011-01-02,2,1,3,4\n2011-01-01,1,2,4,6\n2011-01-03,3,6,7,13\n2011-01-04,4,6,7,13\n2011-01-05,5,6,7,13\n')
    df, manifest = load_local('bike', src)
    assert df.iloc[0]['factor_1'] == 2
    assert len(manifest['raw_sha256']) == 64
    sp = chronological_split(df)
    assert [len(sp[t]) for t in ('train', 'validation', 'holdout')] == [3, 1, 1]


def test_tetouan_local_adapter(tmp_path):
    src = tmp_path / 'Tetuan City power consumption.csv'
    src.write_text('DateTime,Zone 1 Power Consumption,Zone 2 Power Consumption,Zone 3 Power Consumption\n02/01/2017 00:00,100,200,300\n01/01/2017 00:00,80,160,240\n')
    df, manifest = load_local('tetouan', src)
    assert len(df) == 2 and list(df['factor_1']) == [80, 100]
    assert manifest['dataset'] == 'tetouan'


def test_electricity_local_adapter(tmp_path):
    src = tmp_path / 'LD2011_2014.txt'
    cols = [f'MT_{i:03d}' for i in range(1, 22)]
    lines = [';'.join(['Unnamed: 0'] + cols)]
    for k in range(3):
        lines.append(';'.join([f'2011-01-01 00:{k*15:02d}:00'] + ['1,5'] * 21))
    src.write_text('\n'.join(lines) + '\n')
    df, manifest = load_local('electricity', src)
    assert df.shape[0] == 3
    assert float(df.iloc[0].factor_1) == 15.0
    assert float(df.iloc[0].factor_2) == 15.0


def test_fractional_cvar_lp_mixture_exact_rank_weights():
    for n in (5, 9, 11, 35):
        for alpha in (F('0.7'), F('0.9'), F('0.95'), F('0.99')):
            means = fractional_cvar_topmeans(n, alpha)
            values = [F(i * i - 3 * i, 7) for i in range(n)]
            target = cvar_fractional_exact(values, alpha)
            actual = sum(F(str(lam)) * sum(topk_exact(values, k), F(0)) / k for k, lam in means)
            assert abs(float(actual - target)) < 1e-12


if __name__ == '__main__':
    sys.exit(pytest.main(['-q', str(Path(__file__).resolve())]))

