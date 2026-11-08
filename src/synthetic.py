"""Deterministic bivariate scenario generators; no use of holdout data."""
from __future__ import annotations
import numpy as np


def make_cloud(seed, n, family='independent'):
    if n < 1:
        raise ValueError('N must be positive')
    rng = np.random.default_rng(seed)
    if family == 'independent':
        data = rng.gamma(3.2, 22., size=(n, 2))
    elif family in ('rho_-0.9', 'rho_+0.9'):
        rho = -.9 if family == 'rho_-0.9' else .9
        cov = np.array([[1, rho], [rho, 1]])
        z = rng.multivariate_normal(np.zeros(2), cov, size=n)
        # Lognormal with prespecified positive scale; bounded exponent for safety.
        data = 55 * np.exp(np.clip(.65 * z, -5, 5))
    elif family == 't3':
        z = np.clip(rng.standard_t(3, size=(n, 2)), -8, 8)
        data = 55 * np.exp(.35 * z)
    elif family == 'four_mixture':
        centers = np.array([[30, 30], [40, 140], [130, 40], [150, 150]])
        assignment = rng.integers(0, 4, size=n)
        data = centers[assignment] + rng.normal(0, 9, (n, 2))
    elif family == 'near_chain':
        t = np.linspace(0, 200, n)
        data = np.stack([t, t], axis=1) + rng.normal(0, .35, (n, 2))
    elif family == 'near_antichain':
        t = np.linspace(0, 200, n)
        data = np.stack([t, 200 - t], axis=1) + rng.normal(0, .35, (n, 2))
    elif family == 'chain':
        return np.repeat(np.arange(n, dtype=np.int64)[:, None], 2, axis=1)
    elif family == 'antichain':
        return np.stack([np.arange(n), np.arange(n - 1, -1, -1)], axis=1).astype(np.int64)
    elif family == 'ties_duplicates':
        data = rng.integers(0, 9, size=(n, 2))
    else:
        raise ValueError(f'unknown synthetic geometry: {family}')
    return np.rint(np.maximum(data, 0)).astype(np.int64)

