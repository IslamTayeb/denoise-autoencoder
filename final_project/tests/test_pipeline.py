"""Pipeline-level smoke test that mocks the Wikipedia HTTP layer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import wiki
from src.pipeline import build_pipeline_artifacts


def _fake_seasonal_payload(article: str, n_days: int = 1500, freq: float = 1.0 / 365.0,
                           amp: float = 50.0, base: float = 100.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2019-01-07")
    items = []
    for i in range(n_days):
        ts = (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d") + "00"
        seasonal = base + amp * np.sin(2 * np.pi * freq * i)
        noise = rng.normal(scale=5.0)
        views = max(1.0, seasonal + noise)
        items.append({
            "project": "en.wikipedia",
            "article": article,
            "granularity": "daily",
            "timestamp": ts,
            "access": "all-access",
            "agent": "user",
            "views": float(views),
        })
    return {"items": items}


def test_pipeline_smoke(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    fits_dir = tmp_path / "fits"
    cache_dir.mkdir()
    fits_dir.mkdir()

    articles = ["Synth_A", "Synth_B"]
    start = "2019-01-01"
    end = wiki._resolve_end(None)

    for art in articles:
        payload = _fake_seasonal_payload(art)
        cache_path = cache_dir / f"wiki_{art}_{start}_{end}.json"
        cache_path.write_text(json.dumps(payload))

    def _no_network(*args, **kwargs):
        raise RuntimeError("HTTP should not be called when cache exists")

    monkeypatch.setattr(wiki.requests, "get", _no_network)

    cfg = {
        "seed": 0,
        "data": {"articles": articles, "start": start, "end": None},
    }
    artifacts = build_pipeline_artifacts(cfg, fits_dir=fits_dir, cache_dir=cache_dir, force=True)

    assert set(artifacts.fits.keys()) == set(articles)
    assert artifacts.residuals_pooled.size > 0
    assert artifacts.best_noise_kind in ("gaussian", "masking", "impulse")
    assert artifacts.distribution.A_range[0] <= artifacts.distribution.A_range[1]
    assert (fits_dir / "pipeline_artifacts.pkl").exists()
    assert (fits_dir / "noise_fits.json").exists()

    nf = json.loads((fits_dir / "noise_fits.json").read_text())
    assert nf["best_noise_kind"] == artifacts.best_noise_kind
