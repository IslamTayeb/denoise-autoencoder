from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class SineFit:
    c: float
    m: float
    A: float
    f: float
    phi: float
    period_weeks: float
    r_squared: float

    def to_dict(self) -> dict:
        return asdict(self)


def _model(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    c, m, A, f, phi = params
    return c + m * t + A * np.sin(2.0 * np.pi * f * t + phi)


def _residual_fn(params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _model(params, t) - y


def _initial_guess(y: np.ndarray) -> tuple[float, float, float, float, float]:
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    detrended = y - (slope * t + intercept)

    n = len(detrended)
    spectrum = np.fft.rfft(detrended * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0)
    if len(spectrum) <= 1:
        return float(intercept), float(slope), float(np.std(detrended)), 1.0 / max(n, 2), 0.0

    mags = np.abs(spectrum)
    mags[0] = 0.0
    k_peak = int(np.argmax(mags))
    f0 = float(freqs[k_peak]) if freqs[k_peak] > 0 else 1.0 / max(n, 2)
    A0 = float(2.0 * mags[k_peak] / max(n, 1))
    if A0 == 0.0:
        A0 = float(np.std(detrended) + 1e-6)
    phi0 = float(np.angle(spectrum[k_peak]) + np.pi / 2.0)
    return float(intercept), float(slope), A0, f0, phi0


def fit_sine(y: np.ndarray) -> SineFit:
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 8:
        raise ValueError("series too short to fit (need at least 8 samples)")
    t = np.arange(y.size, dtype=float)

    c0, m0, A0, f0, phi0 = _initial_guess(y)

    f_lo = max(f0 * 0.5, 1.0 / (4.0 * y.size))
    f_hi = min(f0 * 1.5, 0.49)
    if f_hi <= f_lo:
        f_lo = max(1.0 / (4.0 * y.size), 1e-4)
        f_hi = 0.49

    A_bound = max(abs(A0) * 5.0, float(np.std(y)) * 5.0, 1e-3)
    c_bound = max(abs(c0) * 5.0, float(np.max(np.abs(y))) * 2.0, 1.0)
    m_bound = max(abs(m0) * 100.0, 1.0)

    bounds_lo = np.array([-c_bound, -m_bound, 0.0, f_lo, -4.0 * np.pi])
    bounds_hi = np.array([c_bound, m_bound, A_bound, f_hi, 4.0 * np.pi])

    x0 = np.array([
        np.clip(c0, bounds_lo[0], bounds_hi[0]),
        np.clip(m0, bounds_lo[1], bounds_hi[1]),
        np.clip(abs(A0), bounds_lo[2], bounds_hi[2]),
        np.clip(f0, bounds_lo[3], bounds_hi[3]),
        np.clip(phi0, bounds_lo[4], bounds_hi[4]),
    ])

    result = least_squares(
        _residual_fn,
        x0=x0,
        bounds=(bounds_lo, bounds_hi),
        args=(t, y),
        max_nfev=5000,
    )

    c, m, A, f, phi = result.x
    phi = float(np.mod(phi + np.pi, 2.0 * np.pi) - np.pi)

    y_hat = _model(result.x, t)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    r_squared = 1.0 - ss_res / ss_tot

    period = float("inf") if f <= 0 else 1.0 / float(f)

    return SineFit(
        c=float(c),
        m=float(m),
        A=float(A),
        f=float(f),
        phi=float(phi),
        period_weeks=period,
        r_squared=float(r_squared),
    )


def predict_sine(fit: SineFit, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return (fit.c + fit.m * t + fit.A * np.sin(2.0 * np.pi * fit.f * t + fit.phi)).astype(np.float32)


def fit_all(df: pd.DataFrame) -> dict[str, SineFit]:
    return {col: fit_sine(df[col].to_numpy()) for col in df.columns}
