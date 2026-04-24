from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Union

import numpy as np
from scipy import stats

NoiseParams = Union["GaussianNoiseParams", "MaskingNoiseParams", "ImpulseNoiseParams"]


@dataclass
class GaussianNoiseParams:
    sigma: float

    def to_dict(self) -> dict:
        return {"kind": "gaussian", **asdict(self)}


@dataclass
class MaskingNoiseParams:
    seg_rate: float
    seg_len_mean: float

    def to_dict(self) -> dict:
        return {"kind": "masking", **asdict(self)}


@dataclass
class ImpulseNoiseParams:
    spike_rate: float
    magnitude_scale: float

    def to_dict(self) -> dict:
        return {"kind": "impulse", **asdict(self)}


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"expected 1D or 2D array; got shape {arr.shape}")


def apply_gaussian(clean: np.ndarray, p: GaussianNoiseParams, rng: np.random.Generator) -> np.ndarray:
    arr, was_1d = _as_2d(clean)
    noise = rng.normal(loc=0.0, scale=p.sigma, size=arr.shape).astype(arr.dtype)
    out = arr + noise
    return out[0] if was_1d else out


def apply_masking(clean: np.ndarray, p: MaskingNoiseParams, rng: np.random.Generator) -> np.ndarray:
    arr, was_1d = _as_2d(clean)
    out = arr.copy()
    n, L = out.shape
    seg_len_mean = max(p.seg_len_mean, 1.0)
    rate = max(p.seg_rate, 0.0)

    for i in range(n):
        n_segs = int(rng.poisson(lam=rate))
        local_mean = float(out[i].mean())
        for _ in range(n_segs):
            seg_len = max(1, int(rng.poisson(lam=seg_len_mean)))
            seg_len = min(seg_len, L)
            start = int(rng.integers(0, L - seg_len + 1))
            out[i, start:start + seg_len] = local_mean
    return out[0] if was_1d else out


def apply_impulse(clean: np.ndarray, p: ImpulseNoiseParams, rng: np.random.Generator) -> np.ndarray:
    arr, was_1d = _as_2d(clean)
    out = arr.copy()
    n, L = out.shape
    rate = float(np.clip(p.spike_rate, 0.0, 1.0))
    if rate <= 0.0:
        return out[0] if was_1d else out

    base_std = float(np.std(arr)) or 1.0
    mag = max(p.magnitude_scale, 0.0) * base_std

    mask = rng.random(size=out.shape) < rate
    signs = rng.choice([-1.0, 1.0], size=out.shape)
    magnitudes = mag * (1.0 + 0.5 * rng.standard_normal(size=out.shape))
    out = out + (mask * signs * magnitudes).astype(out.dtype)
    return out[0] if was_1d else out


def apply_noise(clean: np.ndarray, kind: str, params: NoiseParams,
                rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return apply_gaussian(clean, params, rng)
    if kind == "masking":
        return apply_masking(clean, params, rng)
    if kind == "impulse":
        return apply_impulse(clean, params, rng)
    raise ValueError(f"unknown noise kind {kind!r}")


def fit_gaussian(residuals: np.ndarray) -> GaussianNoiseParams:
    sigma = float(np.std(np.asarray(residuals, dtype=np.float64)))
    return GaussianNoiseParams(sigma=max(sigma, 1e-6))


def fit_masking(residuals: np.ndarray, near_zero_thresh: float = 0.1) -> MaskingNoiseParams:
    r = np.asarray(residuals, dtype=np.float64).ravel()
    sigma = float(np.std(r)) or 1e-6
    below = np.abs(r) < near_zero_thresh * sigma

    runs: list[int] = []
    current = 0
    for b in below:
        if b:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    if not runs:
        return MaskingNoiseParams(seg_rate=0.0, seg_len_mean=1.0)

    n_windows_equiv = max(r.size / 128.0, 1.0)
    seg_rate = float(len(runs)) / n_windows_equiv
    seg_len_mean = float(np.mean(runs))
    return MaskingNoiseParams(seg_rate=seg_rate, seg_len_mean=seg_len_mean)


def fit_impulse(residuals: np.ndarray, outlier_thresh: float = 3.0) -> ImpulseNoiseParams:
    r = np.asarray(residuals, dtype=np.float64).ravel()
    sigma = float(np.std(r)) or 1e-6
    spikes = np.abs(r) > outlier_thresh * sigma
    spike_rate = float(np.mean(spikes))
    if spikes.any():
        magnitude_scale = float(np.mean(np.abs(r[spikes])) / sigma)
    else:
        magnitude_scale = float(outlier_thresh)
    return ImpulseNoiseParams(spike_rate=spike_rate, magnitude_scale=magnitude_scale)


def fit_noise(residuals: np.ndarray, kind: str) -> NoiseParams:
    if kind == "gaussian":
        return fit_gaussian(residuals)
    if kind == "masking":
        return fit_masking(residuals)
    if kind == "impulse":
        return fit_impulse(residuals)
    raise ValueError(f"unknown noise kind {kind!r}")


def score_noise_fit(observed_residuals: np.ndarray,
                    synthetic_residuals: np.ndarray) -> dict:
    obs = np.asarray(observed_residuals, dtype=np.float64).ravel()
    syn = np.asarray(synthetic_residuals, dtype=np.float64).ravel()

    wasserstein = float(stats.wasserstein_distance(obs, syn))
    ks_stat, ks_pvalue = stats.ks_2samp(obs, syn)

    obs_std = float(np.std(obs))
    syn_std = float(np.std(syn))
    obs_kurt = float(stats.kurtosis(obs, fisher=True, bias=False)) if obs_std > 0 else 0.0
    syn_kurt = float(stats.kurtosis(syn, fisher=True, bias=False)) if syn_std > 0 else 0.0

    obs_near = float(np.mean(np.abs(obs) < 0.05 * (obs_std or 1e-12)))
    syn_near = float(np.mean(np.abs(syn) < 0.05 * (syn_std or 1e-12)))

    return {
        "wasserstein": wasserstein,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "kurtosis_diff": float(syn_kurt - obs_kurt),
        "std_ratio": float(syn_std / (obs_std or 1e-12)),
        "near_zero_diff": float(syn_near - obs_near),
    }


def synthesize_residuals(kind: str, params: NoiseParams,
                        clean_signal: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply a noise model to a clean reference signal and return the residual stream."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(clean_signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    noisy = apply_noise(arr, kind, params, rng)
    return (noisy - arr).reshape(-1)
