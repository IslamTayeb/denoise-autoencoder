from __future__ import annotations

import numpy as np

from src.data.residuals import characterize, compute_residuals
from src.data.sine_fit import SineFit


def test_compute_residuals_shape_and_values():
    n = 100
    fit = SineFit(c=0.0, m=0.0, A=1.0, f=1.0 / 50.0, phi=0.0, period_weeks=50.0, r_squared=1.0)
    t = np.arange(n, dtype=float)
    y = np.sin(2 * np.pi * t / 50.0)
    r = compute_residuals(y.astype(np.float32), fit)
    assert r.shape == (n,)
    assert np.allclose(r, 0.0, atol=1e-5)


def test_characterize_gaussian_residuals():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=1.0, size=10_000)
    stats = characterize(r.astype(np.float32))
    assert abs(stats.std - 1.0) < 0.05
    assert abs(stats.kurtosis) < 0.2
    assert 0.001 < stats.fraction_outliers < 0.01


def test_characterize_sparse_spikes():
    rng = np.random.default_rng(1)
    r = np.zeros(10_000)
    spike_idx = rng.choice(r.size, size=100, replace=False)
    r[spike_idx] = rng.choice([-5.0, 5.0], size=100)
    stats = characterize(r.astype(np.float32))
    assert stats.kurtosis > 5.0
    assert stats.fraction_near_zero > 0.5
