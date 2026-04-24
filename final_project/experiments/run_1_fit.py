from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._common import ensure_dirs, load_config, parse_args, set_seed
from src.pipeline import build_pipeline_artifacts
from src.viz import plot_sine_fit


def main() -> int:
    args = parse_args(default_config="configs/exp_1_fit.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_1_fit"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"], force=True)

    sine_rows = []
    for col, fit in artifacts.fits.items():
        sine_rows.append({"article": col, **fit.to_dict()})
    sine_df = pd.DataFrame(sine_rows)
    sine_df.to_csv(paths["tables"] / "sine_fits.csv", index=False)

    stats_rows = []
    for col, st in artifacts.residual_stats.items():
        stats_rows.append({"article": col, **st.to_dict()})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(paths["tables"] / "residual_stats.csv", index=False)

    for col in artifacts.log_df.columns:
        y = artifacts.log_df[col].to_numpy()
        t = np.arange(y.size)
        fit = artifacts.fits[col]
        y_hat = fit.c + fit.m * t + fit.A * np.sin(2.0 * np.pi * fit.f * t + fit.phi)
        plot_sine_fit(t, y, y_hat,
                      out_path=paths["figures"] / f"{col}_sine_fit.png",
                      title=f"{col}: sine fit  (R²={fit.r_squared:.3f}, period≈{fit.period_weeks:.1f}w)")

    print(
        "EXP1 done | articles=",
        list(artifacts.fits.keys()),
        " | mean R²=",
        f"{float(np.mean([f.r_squared for f in artifacts.fits.values()])):.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
