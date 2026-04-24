from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.noise import (
    NoiseParams,
    fit_gaussian,
    fit_impulse,
    fit_masking,
    score_noise_fit,
    synthesize_residuals,
)
from .data.residuals import ResidualStats, characterize, compute_residuals
from .data.sine_fit import SineFit, fit_all
from .data.synthetic import SineFitDistribution, distribution_from_fits
from .data.wiki import load_wiki_weekly, to_log

NOISE_KINDS = ("gaussian", "masking", "impulse")
ARTIFACTS_FILENAME = "pipeline_artifacts.pkl"


@dataclass
class NoiseFitResult:
    kind: str
    params: NoiseParams
    score: dict[str, float]


@dataclass
class PipelineArtifacts:
    real_df: pd.DataFrame                       # raw weekly pageviews
    log_df: pd.DataFrame                        # log1p-transformed
    fits: dict[str, SineFit]
    residuals_per_article: dict[str, np.ndarray]
    residuals_pooled: np.ndarray
    residual_stats: dict[str, ResidualStats]
    noise_fits: dict[str, NoiseFitResult]       # one per kind
    best_noise_kind: str
    distribution: SineFitDistribution
    config_snapshot: dict[str, Any]


def _representative_clean_signal(fits: dict[str, SineFit], log_df: pd.DataFrame) -> np.ndarray:
    pieces = []
    for col, fit in fits.items():
        n = len(log_df[col])
        t = np.arange(n, dtype=float)
        pieces.append(fit.c + fit.m * t + fit.A * np.sin(2.0 * np.pi * fit.f * t + fit.phi))
    return np.concatenate(pieces).astype(np.float32)


def _save_artifacts(artifacts: PipelineArtifacts, fits_dir: Path) -> None:
    fits_dir.mkdir(parents=True, exist_ok=True)
    with (fits_dir / ARTIFACTS_FILENAME).open("wb") as f:
        pickle.dump(artifacts, f)
    with (fits_dir / "sine_fits.pkl").open("wb") as f:
        pickle.dump(artifacts.fits, f)
    np.save(fits_dir / "residuals.npy", artifacts.residuals_pooled)
    with (fits_dir / "noise_fits.json").open("w") as f:
        json.dump(
            {
                "best_noise_kind": artifacts.best_noise_kind,
                "fits": {
                    k: {"params": v.params.to_dict(), "score": v.score}
                    for k, v in artifacts.noise_fits.items()
                },
            },
            f,
            indent=2,
        )


def load_artifacts(fits_dir: Path | str = Path("results/fits")) -> PipelineArtifacts:
    fits_dir = Path(fits_dir)
    path = fits_dir / ARTIFACTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no cached pipeline artifacts at {path}; run experiment 1 first")
    with path.open("rb") as f:
        return pickle.load(f)


def build_pipeline_artifacts(cfg: dict, fits_dir: Path | str = Path("results/fits"),
                             cache_dir: Path | str = Path("data_cache"),
                             force: bool = False) -> PipelineArtifacts:
    fits_dir = Path(fits_dir)
    cache_dir = Path(cache_dir)

    cached = fits_dir / ARTIFACTS_FILENAME
    if cached.exists() and not force:
        return load_artifacts(fits_dir)

    data_cfg = cfg.get("data", {})
    articles = list(data_cfg.get("articles", []))
    if not articles:
        raise ValueError("config 'data.articles' must be a non-empty list")
    start = data_cfg.get("start", "2018-01-01")
    end = data_cfg.get("end", None)

    real_df = load_wiki_weekly(articles=articles, start=start, end=end, cache_dir=cache_dir)
    log_df = to_log(real_df)

    fits = fit_all(log_df)

    residuals_per_article: dict[str, np.ndarray] = {}
    residual_stats: dict[str, ResidualStats] = {}
    for col in log_df.columns:
        r = compute_residuals(log_df[col].to_numpy(), fits[col])
        residuals_per_article[col] = r
        residual_stats[col] = characterize(r)

    pooled = np.concatenate(list(residuals_per_article.values())).astype(np.float32)

    fitters = {"gaussian": fit_gaussian, "masking": fit_masking, "impulse": fit_impulse}
    representative_clean = _representative_clean_signal(fits, log_df)

    noise_fits: dict[str, NoiseFitResult] = {}
    for kind in NOISE_KINDS:
        params = fitters[kind](pooled)
        synth = synthesize_residuals(kind, params, representative_clean, seed=cfg.get("seed", 0))
        score = score_noise_fit(pooled, synth)
        noise_fits[kind] = NoiseFitResult(kind=kind, params=params, score=score)

    best_kind = min(noise_fits, key=lambda k: noise_fits[k].score["wasserstein"])

    real_series_len = int(np.median([len(log_df[c]) for c in log_df.columns]))
    distribution = distribution_from_fits(fits, real_series_len=real_series_len)

    artifacts = PipelineArtifacts(
        real_df=real_df,
        log_df=log_df,
        fits=fits,
        residuals_per_article=residuals_per_article,
        residuals_pooled=pooled,
        residual_stats=residual_stats,
        noise_fits=noise_fits,
        best_noise_kind=best_kind,
        distribution=distribution,
        config_snapshot=dict(cfg),
    )
    _save_artifacts(artifacts, fits_dir)
    return artifacts
