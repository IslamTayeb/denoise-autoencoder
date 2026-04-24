from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._common import (
    ensure_dirs,
    load_config,
    parse_args,
    resolve_model_name,
    resolve_noise_kind,
    set_seed,
)
from src.data.noise import (
    GaussianNoiseParams,
    ImpulseNoiseParams,
    MaskingNoiseParams,
)
from src.data.synthetic import build_loaders, build_test_loader
from src.eval import evaluate
from src.models import build_model
from src.pipeline import build_pipeline_artifacts
from src.train import train
from src.viz import plot_metric_vs_x


def _scale_params(params, multiplier: float):
    if isinstance(params, GaussianNoiseParams):
        return GaussianNoiseParams(sigma=params.sigma * multiplier)
    if isinstance(params, MaskingNoiseParams):
        return MaskingNoiseParams(
            seg_rate=params.seg_rate * multiplier,
            seg_len_mean=params.seg_len_mean,
        )
    if isinstance(params, ImpulseNoiseParams):
        return ImpulseNoiseParams(
            spike_rate=min(1.0, params.spike_rate * multiplier),
            magnitude_scale=params.magnitude_scale,
        )
    raise TypeError(f"unknown params type {type(params)!r}")


def main() -> int:
    args = parse_args(default_config="configs/exp_6_noise_levels.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_6_noise_levels"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"])
    noise_kind = resolve_noise_kind(cfg, artifacts)
    base_params = artifacts.noise_fits[noise_kind].params
    model_name = resolve_model_name(cfg, paths["tables"])
    latent_dim = int(cfg.get("model", {}).get("latent_dim", 16))
    window_len = cfg.get("window_len", 128)

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    batch = train_cfg.get("batch", 64)
    lr = train_cfg.get("lr", 1e-3)
    patience = train_cfg.get("patience", 10)
    train_n = train_cfg.get("train_n", 8000)
    val_n = train_cfg.get("val_n", 1000)
    test_n = train_cfg.get("test_n", 1000)

    multipliers = cfg.get("intensity_multipliers", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])

    train_loader, val_loader, _ = build_loaders(
        artifacts.distribution, noise_kind, base_params, seed=seed,
        batch=batch, train_n=train_n, val_n=val_n, test_n=test_n, window_len=window_len,
    )
    model = build_model(model_name, latent_dim=latent_dim, window_len=window_len)
    train(model, train_loader, val_loader, epochs=epochs, lr=lr,
          patience=patience, log_path=paths["checkpoints"])

    rows = []
    for mult in multipliers:
        eval_params = _scale_params(base_params, float(mult))
        test_loader = build_test_loader(
            artifacts.distribution, noise_kind, eval_params, seed=seed + 999,
            batch=batch, n=test_n, window_len=window_len,
        )
        res = evaluate(model, test_loader)
        rows.append({
            "intensity_multiplier": float(mult),
            "snr_input_db": res["snr_input_db"],
            "snr_output_db": res["snr_output_db"],
            "snr_improvement_db": res["snr_improvement_db"],
            "test_mse": res["mse"],
        })

    df = pd.DataFrame(rows).sort_values("intensity_multiplier")
    df.to_csv(paths["tables"] / "noise_levels.csv", index=False)

    plot_metric_vs_x(df["intensity_multiplier"].tolist(), df["snr_improvement_db"].tolist(),
                     xlabel="intensity multiplier (1.0 = train)",
                     ylabel="SNR improvement (dB)",
                     out_path=paths["figures"] / "snr_imp_vs_intensity.png",
                     title=f"{model_name} | noise={noise_kind}", vline=1.0)
    plot_metric_vs_x(df["intensity_multiplier"].tolist(), df["test_mse"].tolist(),
                     xlabel="intensity multiplier (1.0 = train)",
                     ylabel="test MSE",
                     out_path=paths["figures"] / "mse_vs_intensity.png",
                     title=f"{model_name} | noise={noise_kind}", vline=1.0)

    print(f"EXP6 done | model={model_name} | noise={noise_kind} | multipliers={multipliers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
