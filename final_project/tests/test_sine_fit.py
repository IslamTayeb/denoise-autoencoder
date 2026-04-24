from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.sine_fit import fit_all, fit_sine, predict_sine


def test_fit_recovers_known_sine():
    n = 200
    t = np.arange(n, dtype=float)
    A_true, f_true, phi_true, c_true, m_true = 1.7, 1.0 / 52.0, 0.6, 3.0, 0.005
    y = c_true + m_true * t + A_true * np.sin(2.0 * np.pi * f_true * t + phi_true)
    rng = np.random.default_rng(0)
    y_noisy = y + rng.normal(scale=0.05, size=n)

    fit = fit_sine(y_noisy)
    assert fit.r_squared > 0.95
    assert abs(fit.A - A_true) < 0.2
    assert abs(fit.f - f_true) / f_true < 0.1
    assert abs(fit.period_weeks - 52.0) < 5.0


def test_predict_sine_matches_components():
    fit = fit_sine(np.sin(np.arange(200) * 2 * np.pi / 50.0))
    t = np.arange(200, dtype=float)
    y = predict_sine(fit, t)
    assert y.shape == (200,)
    assert np.isfinite(y).all()


def test_fit_all_dataframe():
    n = 300
    t = np.arange(n, dtype=float)
    df = pd.DataFrame({
        "a": np.sin(2 * np.pi * t / 52.0),
        "b": 2.0 + 0.5 * np.sin(2 * np.pi * t / 26.0 + 0.4),
    })
    fits = fit_all(df)
    assert set(fits.keys()) == {"a", "b"}
    for fit in fits.values():
        assert fit.r_squared > 0.9
