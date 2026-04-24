from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import wiki


def _fake_json_for(article: str, n_days: int = 200) -> dict:
    base = pd.Timestamp("2021-01-04")
    items = []
    for i in range(n_days):
        ts = (base + pd.Timedelta(days=i)).strftime("%Y%m%d") + "00"
        items.append({
            "project": "en.wikipedia",
            "article": article,
            "granularity": "daily",
            "timestamp": ts,
            "access": "all-access",
            "agent": "user",
            "views": 100 + (i % 10),
        })
    return {"items": items}


def test_load_wiki_weekly_uses_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path
    article = "TestArticle"
    payload = _fake_json_for(article, n_days=210)
    cache_path = cache_dir / f"wiki_{article}_2021-01-01_2021-08-01.json"
    cache_path.write_text(json.dumps(payload))

    def _no_network(*args, **kwargs):
        raise RuntimeError("HTTP should not be called when cache exists")

    monkeypatch.setattr(wiki.requests, "get", _no_network)

    df = wiki.load_wiki_weekly(
        articles=[article], start="2021-01-01", end="2021-08-01", cache_dir=cache_dir,
    )
    assert article in df.columns
    assert df.index.freqstr is not None or df.index.is_monotonic_increasing
    assert df.dtypes[article] == "float32"
    assert (df[article] > 0).all()


def test_to_log_is_log1p():
    df = pd.DataFrame({"a": [0.0, 1.0, 9.0]})
    out = wiki.to_log(df)
    import numpy as np
    np.testing.assert_allclose(out["a"].values, np.log1p([0.0, 1.0, 9.0]).astype("float32"), rtol=1e-5)
