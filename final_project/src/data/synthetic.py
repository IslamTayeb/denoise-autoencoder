from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .noise import NoiseParams, apply_noise
from .sine_fit import SineFit

WINDOW_LEN = 128
TRAIN_N = 8000
VAL_N = 1000
TEST_N = 1000
DEFAULT_BATCH = 64


@dataclass
class SineFitDistribution:
    A_range: tuple[float, float]
    f_range: tuple[float, float]
    phi_range: tuple[float, float]
    m_range: tuple[float, float]
    c_range: tuple[float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _expand_range(values: list[float], min_span: float = 1e-6, jitter: float = 0.0) -> tuple[float, float]:
    if not values:
        return (-1.0, 1.0)
    lo = float(min(values))
    hi = float(max(values))
    if hi - lo < min_span:
        center = 0.5 * (lo + hi)
        half = max(min_span * 0.5, abs(center) * 0.05 + min_span)
        lo, hi = center - half, center + half
    if jitter > 0.0:
        span = hi - lo
        lo -= jitter * span
        hi += jitter * span
    return (lo, hi)


def distribution_from_fits(
    fits: dict[str, SineFit],
    real_series_len: int,
    window_len: int = WINDOW_LEN,
    freq_jitter: float = 0.5,
) -> SineFitDistribution:
    if not fits:
        raise ValueError("need at least one fit to build a distribution")

    amplitudes = [f.A for f in fits.values()]
    intercepts = [f.c for f in fits.values()]
    slopes = [f.m for f in fits.values()]

    cycles_in_real = [f.f * real_series_len for f in fits.values()]
    cycles_in_window = [c * (window_len / real_series_len) for c in cycles_in_real]
    f_per_window_sample = [c / window_len for c in cycles_in_window]

    return SineFitDistribution(
        A_range=_expand_range(amplitudes, min_span=0.05, jitter=0.25),
        f_range=_expand_range(f_per_window_sample, min_span=1e-4, jitter=freq_jitter),
        phi_range=(0.0, 2.0 * np.pi),
        m_range=_expand_range(slopes, min_span=1e-4, jitter=0.5),
        c_range=_expand_range(intercepts, min_span=0.1, jitter=0.25),
    )


def make_clean_dataset(n: int, dist: SineFitDistribution,
                       window_len: int = WINDOW_LEN, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.uniform(max(dist.A_range[0], 0.0), max(dist.A_range[1], 1e-3), size=n)
    f = rng.uniform(max(dist.f_range[0], 1e-5), max(dist.f_range[1], 2e-5), size=n)
    phi = rng.uniform(dist.phi_range[0], dist.phi_range[1], size=n)
    m = rng.uniform(dist.m_range[0], dist.m_range[1], size=n)
    c = rng.uniform(dist.c_range[0], dist.c_range[1], size=n)

    t = np.arange(window_len, dtype=np.float32)
    out = np.empty((n, window_len), dtype=np.float32)
    for i in range(n):
        sig = c[i] + m[i] * t + A[i] * np.sin(2.0 * np.pi * f[i] * t + phi[i])
        std = float(sig.std())
        if std > 1e-8:
            sig = (sig - sig.mean()) / std
        else:
            sig = sig - sig.mean()
        out[i] = sig.astype(np.float32)
    return out


def make_noisy_dataset(clean: np.ndarray, noise_kind: str, params: NoiseParams,
                       seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return apply_noise(clean, noise_kind, params, rng).astype(np.float32)


def _to_loader(clean: np.ndarray, noisy: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    clean_t = torch.from_numpy(clean).float().unsqueeze(1)   # (N, 1, L)
    noisy_t = torch.from_numpy(noisy).float().unsqueeze(1)
    ds = TensorDataset(noisy_t, clean_t)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def build_loaders(
    dist: SineFitDistribution,
    noise_kind: str,
    params: NoiseParams,
    seed: int,
    batch: int = DEFAULT_BATCH,
    train_n: int = TRAIN_N,
    val_n: int = VAL_N,
    test_n: int = TEST_N,
    window_len: int = WINDOW_LEN,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_clean = make_clean_dataset(train_n, dist, window_len, seed=seed)
    val_clean = make_clean_dataset(val_n, dist, window_len, seed=seed + 1)
    test_clean = make_clean_dataset(test_n, dist, window_len, seed=seed + 2)

    train_noisy = make_noisy_dataset(train_clean, noise_kind, params, seed=seed + 100)
    val_noisy = make_noisy_dataset(val_clean, noise_kind, params, seed=seed + 101)
    test_noisy = make_noisy_dataset(test_clean, noise_kind, params, seed=seed + 102)

    return (
        _to_loader(train_clean, train_noisy, batch, shuffle=True),
        _to_loader(val_clean, val_noisy, batch, shuffle=False),
        _to_loader(test_clean, test_noisy, batch, shuffle=False),
    )


def build_test_loader(
    dist: SineFitDistribution,
    noise_kind: str,
    params: NoiseParams,
    seed: int,
    batch: int = DEFAULT_BATCH,
    n: int = TEST_N,
    window_len: int = WINDOW_LEN,
) -> DataLoader:
    clean = make_clean_dataset(n, dist, window_len, seed=seed + 2)
    noisy = make_noisy_dataset(clean, noise_kind, params, seed=seed + 102)
    return _to_loader(clean, noisy, batch, shuffle=False)
