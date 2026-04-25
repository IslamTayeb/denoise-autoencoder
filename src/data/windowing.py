from __future__ import annotations

import numpy as np


def to_windows(series: np.ndarray, window_len: int = 128, stride: int = 1
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(series, dtype=np.float32).ravel()
    if s.size < window_len:
        raise ValueError(f"series of length {s.size} shorter than window_len={window_len}")
    starts = np.arange(0, s.size - window_len + 1, stride)
    windows = np.stack([s[i:i + window_len] for i in starts], axis=0)

    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    safe_stds = np.where(stds < 1e-8, 1.0, stds)
    norm = (windows - means) / safe_stds

    return norm.astype(np.float32), means.squeeze(1).astype(np.float32), safe_stds.squeeze(1).astype(np.float32)


def from_windows(windows: np.ndarray, means: np.ndarray, stds: np.ndarray,
                series_len: int, stride: int = 1) -> np.ndarray:
    w = np.asarray(windows, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected 2D windows array; got shape {w.shape}")
    n_windows, window_len = w.shape

    denorm = w * stds[:, None] + means[:, None]

    accum = np.zeros(series_len, dtype=np.float64)
    counts = np.zeros(series_len, dtype=np.float64)
    for i in range(n_windows):
        start = i * stride
        end = start + window_len
        if end > series_len:
            end = series_len
            valid = end - start
        else:
            valid = window_len
        accum[start:end] += denorm[i, :valid]
        counts[start:end] += 1.0
    counts = np.where(counts == 0.0, 1.0, counts)
    return (accum / counts).astype(np.float32)
