from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._common import ensure_dirs, load_config, parse_args, set_seed
from src.data.noise import synthesize_residuals
from src.pipeline import build_pipeline_artifacts
from src.viz import plot_qq, plot_residual_hist


def main() -> int:
    args = parse_args(default_config="configs/exp_2_noise_select.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_2_noise_select"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"])

    rows = []
    rank_order = sorted(artifacts.noise_fits.values(), key=lambda r: r.score["wasserstein"])
    rank_lookup = {r.kind: i + 1 for i, r in enumerate(rank_order)}

    for kind, result in artifacts.noise_fits.items():
        rows.append({
            "noise_kind": kind,
            "params_json": json.dumps(result.params.to_dict()),
            **result.score,
            "rank": rank_lookup[kind],
        })
    df = pd.DataFrame(rows).sort_values("rank")
    df.to_csv(paths["tables"] / "noise_selection.csv", index=False)

    log_concat = artifacts.residuals_pooled
    representative = []
    for col, fit in artifacts.fits.items():
        n = len(artifacts.log_df[col])
        t = np.arange(n, dtype=float)
        representative.append(fit.c + fit.m * t + fit.A * np.sin(2.0 * np.pi * fit.f * t + fit.phi))
    representative = np.concatenate(representative).astype(np.float32)

    synthetics = {}
    for kind, result in artifacts.noise_fits.items():
        syn = synthesize_residuals(kind, result.params, representative, seed=seed + 13)
        synthetics[kind] = syn

    plot_residual_hist(log_concat, synthetics,
                       out_path=paths["figures"] / "residual_hist_overlay.png",
                       title="observed vs synthetic residual distributions")
    for kind, syn in synthetics.items():
        plot_qq(log_concat, syn,
                out_path=paths["figures"] / f"qq_{kind}.png",
                title=f"QQ: observed vs {kind}")

    print(f"EXP2 done | BEST_NOISE={artifacts.best_noise_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
