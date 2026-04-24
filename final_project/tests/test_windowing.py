from __future__ import annotations

import numpy as np

from src.data.windowing import from_windows, to_windows


def test_to_windows_basic():
    series = np.arange(20, dtype=np.float32)
    windows, means, stds = to_windows(series, window_len=5, stride=1)
    assert windows.shape == (16, 5)
    assert means.shape == (16,)
    assert stds.shape == (16,)
    assert np.allclose(windows.mean(axis=1), 0.0, atol=1e-5)


def test_round_trip_recovers_constant_signal():
    series = 3.0 * np.ones(64, dtype=np.float32)
    windows, means, stds = to_windows(series, window_len=8, stride=1)
    recovered = from_windows(windows, means, stds, series_len=64, stride=1)
    assert np.allclose(recovered, series, atol=1e-4)


def test_round_trip_recovers_signal_after_identity_recon():
    rng = np.random.default_rng(0)
    series = rng.standard_normal(50).astype(np.float32) + 5.0
    windows, means, stds = to_windows(series, window_len=10, stride=1)
    recovered = from_windows(windows, means, stds, series_len=50, stride=1)
    assert np.allclose(recovered, series, atol=1e-4)
