from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from .sine_fit import SineFit, predict_sine


@dataclass
class ResidualStats:
    mean: float
    std: float
    skew: float
    kurtosis: float
    fraction_outliers: float
    fraction_near_zero: float
    autocorr_lag1: float
    max_run_below_thresh: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_residuals(y: np.ndarray, fit: SineFit) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).ravel()
    t = np.arange(y.size, dtype=float)
    y_hat = predict_sine(fit, t)
    return (y - y_hat).astype(np.float32)


def _max_run_below(values: np.ndarray, thresh: float) -> int:
    below = np.abs(values) < thresh
    if not below.any():
        return 0
    best = run = 0
    for b in below:
        if b:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


def _autocorr_lag1(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    v = values - values.mean()
    denom = float(np.dot(v, v))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v[:-1], v[1:]) / denom)


def characterize(residuals: np.ndarray) -> ResidualStats:
    r = np.asarray(residuals, dtype=np.float64).ravel()
    if r.size < 4:
        raise ValueError("need at least 4 residual samples to characterize")

    sigma = float(np.std(r))
    mean = float(np.mean(r))
    skew = float(stats.skew(r, bias=False)) if sigma > 0 else 0.0
    kurt = float(stats.kurtosis(r, fisher=True, bias=False)) if sigma > 0 else 0.0

    if sigma > 0:
        frac_outliers = float(np.mean(np.abs(r) > 3.0 * sigma))
        frac_near_zero = float(np.mean(np.abs(r) < 0.05 * sigma))
        max_run = _max_run_below(r, 0.1 * sigma)
    else:
        frac_outliers = 0.0
        frac_near_zero = 1.0
        max_run = int(r.size)

    return ResidualStats(
        mean=mean,
        std=sigma,
        skew=skew,
        kurtosis=kurt,
        fraction_outliers=frac_outliers,
        fraction_near_zero=frac_near_zero,
        autocorr_lag1=_autocorr_lag1(r),
        max_run_below_thresh=max_run,
    )
