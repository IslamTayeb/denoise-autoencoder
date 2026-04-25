from __future__ import annotations

import numpy as np

from src.data.noise import (
    GaussianNoiseParams,
    ImpulseNoiseParams,
    MaskingNoiseParams,
    apply_gaussian,
    apply_impulse,
    apply_masking,
    fit_gaussian,
    fit_impulse,
    fit_masking,
    score_noise_fit,
    synthesize_residuals,
)


def test_gaussian_shape_and_std():
    rng = np.random.default_rng(0)
    clean = np.zeros((50, 128), dtype=np.float32)
    out = apply_gaussian(clean, GaussianNoiseParams(sigma=0.5), rng)
    assert out.shape == clean.shape
    assert abs(float(out.std()) - 0.5) < 0.05


def test_masking_creates_zero_runs():
    rng = np.random.default_rng(0)
    clean = np.ones((10, 200), dtype=np.float32)  # local mean = 1
    out = apply_masking(clean, MaskingNoiseParams(seg_rate=3.0, seg_len_mean=10.0), rng)
    assert out.shape == clean.shape
    diffs = np.abs(out - clean)
    assert diffs.max() < 1e-5  # masked = local mean = 1 here, so identical
    clean2 = np.linspace(-1, 1, 200, dtype=np.float32)[None, :].repeat(10, axis=0)
    rng2 = np.random.default_rng(0)
    out2 = apply_masking(clean2, MaskingNoiseParams(seg_rate=3.0, seg_len_mean=10.0), rng2)
    assert (np.abs(out2 - clean2) > 1e-3).any()


def test_impulse_creates_outliers():
    rng = np.random.default_rng(0)
    clean = np.zeros((50, 128), dtype=np.float32)
    out = apply_impulse(clean, ImpulseNoiseParams(spike_rate=0.05, magnitude_scale=3.0), rng)
    sigma_clean = float(clean.std()) or 1.0
    n_spikes = int(np.sum(np.abs(out) > 2.0 * sigma_clean))
    assert n_spikes > 0
    assert out.shape == clean.shape


def test_fit_gaussian_recovers_sigma():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.7, size=10_000).astype(np.float32)
    p = fit_gaussian(r)
    assert abs(p.sigma - 0.7) < 0.05


def test_fit_impulse_finds_outliers():
    rng = np.random.default_rng(0)
    r = np.zeros(10_000, dtype=np.float32)
    spike_idx = rng.choice(r.size, size=100, replace=False)
    r[spike_idx] = rng.choice([-5.0, 5.0], size=100).astype(np.float32)
    p = fit_impulse(r, outlier_thresh=3.0)
    assert p.spike_rate > 0.005
    assert p.magnitude_scale > 1.0


def test_score_noise_fit_rewards_match():
    rng = np.random.default_rng(0)
    obs = rng.normal(scale=1.0, size=5000)
    syn_match = rng.normal(scale=1.0, size=5000)
    syn_off = rng.normal(scale=3.0, size=5000)
    s_match = score_noise_fit(obs, syn_match)
    s_off = score_noise_fit(obs, syn_off)
    assert s_match["wasserstein"] < s_off["wasserstein"]


def test_synthesize_residuals_shape():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 4 * np.pi, 256)).astype(np.float32)
    out = synthesize_residuals("gaussian", GaussianNoiseParams(sigma=0.3), clean, seed=0)
    assert out.shape == clean.shape
