from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

WIKI_DEFAULT_ARTICLES = ["Influenza", "Common_cold", "Fever", "Cough", "Christmas"]
WIKI_DEFAULT_START = "2018-01-01"
WIKI_DEFAULT_END: str | None = None

_USER_AGENT = "cs675-denoising-project/1.0 (educational; contact: nikhil.pesaladinne@duke.edu)"
_API_TEMPLATE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
)


def _fmt_date(d: str | date | datetime) -> str:
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    elif isinstance(d, datetime):
        d = d.date()
    return d.strftime("%Y%m%d")


def _resolve_end(end: str | None) -> str:
    if end is not None:
        return end
    yesterday = date.today() - timedelta(days=2)
    return yesterday.strftime("%Y-%m-%d")


def _fetch_article_json(article: str, start: str, end: str, cache_dir: Path,
                       max_retries: int = 3, backoff: float = 1.5) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_article = article.replace("/", "_")
    cache_path = cache_dir / f"wiki_{safe_article}_{start}_{end}.json"
    if cache_path.exists():
        with cache_path.open("r") as f:
            return json.load(f)

    url = _API_TEMPLATE.format(
        article=quote(article, safe=""),
        start=_fmt_date(start),
        end=_fmt_date(end),
    )
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 429:
                time.sleep(backoff ** (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            with cache_path.open("w") as f:
                json.dump(data, f)
            return data
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff ** (attempt + 1))
    raise RuntimeError(f"failed to fetch {article} from Wikimedia after {max_retries} retries: {last_err}")


def _json_to_daily_series(data: dict, article: str) -> pd.Series:
    items = data.get("items", [])
    if not items:
        raise ValueError(f"no pageview items returned for {article}")
    rows = []
    for it in items:
        ts = it["timestamp"]
        d = datetime.strptime(ts[:8], "%Y%m%d").date()
        rows.append((d, float(it["views"])))
    s = pd.Series(
        [v for _, v in rows],
        index=pd.DatetimeIndex([d for d, _ in rows], name="date"),
        name=article,
        dtype="float32",
    )
    return s.sort_index()


def load_wiki_weekly(
    articles: list[str] = WIKI_DEFAULT_ARTICLES,
    start: str = WIKI_DEFAULT_START,
    end: str | None = WIKI_DEFAULT_END,
    cache_dir: Path | str = Path("data_cache"),
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    end_resolved = _resolve_end(end)
    series_per_article: dict[str, pd.Series] = {}
    for article in articles:
        data = _fetch_article_json(article, start, end_resolved, cache_dir)
        daily = _json_to_daily_series(data, article)
        weekly = daily.resample("W-MON", label="left", closed="left").sum().astype("float32")
        series_per_article[article] = weekly
    df = pd.concat(series_per_article, axis=1).dropna(how="any")
    df.index.name = "week"
    return df.astype("float32")


def to_log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(df).astype("float32")
