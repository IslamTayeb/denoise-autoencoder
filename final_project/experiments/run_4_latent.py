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
from src.data.synthetic import build_loaders
from src.eval import evaluate
from src.models import build_model
from src.pipeline import build_pipeline_artifacts
from src.train import count_parameters, train
from src.viz import plot_metric_vs_x


def main() -> int:
    args = parse_args(default_config="configs/exp_4_latent.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_4_latent"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"])
    noise_kind = resolve_noise_kind(cfg, artifacts)
    noise_params = artifacts.noise_fits[noise_kind].params
    model_name = resolve_model_name(cfg, paths["tables"])

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    batch = train_cfg.get("batch", 64)
    lr = train_cfg.get("lr", 1e-3)
    patience = train_cfg.get("patience", 10)
    train_n = train_cfg.get("train_n", 8000)
    val_n = train_cfg.get("val_n", 1000)
    test_n = train_cfg.get("test_n", 1000)
    window_len = cfg.get("window_len", 128)

    latents = cfg.get("model", {}).get("latents", [4, 8, 16, 32, 64])

    train_loader, val_loader, test_loader = build_loaders(
        artifacts.distribution, noise_kind, noise_params, seed=seed,
        batch=batch, train_n=train_n, val_n=val_n, test_n=test_n, window_len=window_len,
    )

    rows = []
    for latent_dim in latents:
        model = build_model(model_name, latent_dim=int(latent_dim), window_len=window_len)
        n_params = count_parameters(model)
        ckpt_dir = paths["checkpoints"] / f"latent_{latent_dim}"
        train_result = train(model, train_loader, val_loader, epochs=epochs, lr=lr,
                             patience=patience, log_path=ckpt_dir)
        eval_result = evaluate(model, test_loader)
        rows.append({
            "latent_dim": int(latent_dim),
            "n_params": n_params,
            "test_mse": eval_result["mse"],
            "snr_improvement_db": eval_result["snr_improvement_db"],
            "snr_output_db": eval_result["snr_output_db"],
            "best_epoch": train_result["best_epoch"],
        })

    df = pd.DataFrame(rows).sort_values("latent_dim")
    df.to_csv(paths["tables"] / "latent_sweep.csv", index=False)

    plot_metric_vs_x(df["latent_dim"].tolist(), df["test_mse"].tolist(),
                     xlabel="latent dim", ylabel="test MSE",
                     out_path=paths["figures"] / "mse_vs_latent.png",
                     title=f"{model_name} | noise={noise_kind}")
    plot_metric_vs_x(df["latent_dim"].tolist(), df["snr_improvement_db"].tolist(),
                     xlabel="latent dim", ylabel="SNR improvement (dB)",
                     out_path=paths["figures"] / "snr_imp_vs_latent.png",
                     title=f"{model_name} | noise={noise_kind}")

    best = df.sort_values("snr_improvement_db", ascending=False).iloc[0]
    print(f"EXP4 done | BEST_LATENT={int(best['latent_dim'])} | snr_imp={best['snr_improvement_db']:.2f}dB | model={model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
