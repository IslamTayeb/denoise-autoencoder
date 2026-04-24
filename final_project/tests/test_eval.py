from __future__ import annotations

import numpy as np
import torch

from src.eval import mse, snr_db, snr_improvement_db


def test_mse_zero_when_equal():
    a = torch.randn(3, 1, 32)
    assert mse(a, a) == 0.0


def test_snr_large_when_recon_close():
    clean = torch.randn(8, 1, 64)
    recon = clean + 1e-6 * torch.randn_like(clean)
    val = snr_db(clean, recon)
    assert val > 50.0


def test_snr_improvement_positive_when_recon_better():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((4, 1, 64)).astype(np.float32)
    noise = rng.standard_normal(clean.shape).astype(np.float32) * 0.5
    noisy = clean + noise
    recon = clean + 0.05 * rng.standard_normal(clean.shape).astype(np.float32)
    imp = snr_improvement_db(clean, noisy, recon)
    assert imp > 0


def test_snr_improvement_negative_when_recon_worse():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((4, 1, 64)).astype(np.float32)
    noisy = clean + 0.1 * rng.standard_normal(clean.shape).astype(np.float32)
    recon = clean + 0.5 * rng.standard_normal(clean.shape).astype(np.float32)
    imp = snr_improvement_db(clean, noisy, recon)
    assert imp < 0
