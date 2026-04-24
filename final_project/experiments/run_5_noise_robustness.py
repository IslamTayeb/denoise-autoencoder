from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._common import (
    ensure_dirs,
    load_config,
    parse_args,
    resolve_model_name,
    set_seed,
)
from src.data.synthetic import build_loaders, build_test_loader
from src.eval import evaluate
from src.models import build_model
from src.pipeline import build_pipeline_artifacts
from src.train import train
from src.viz import plot_heatmap


def main() -> int:
    args = parse_args(default_config="configs/exp_5_noise_robustness.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_5_noise_robustness"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"])
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

    noise_kinds = cfg.get("noise_kinds", ["gaussian", "masking", "impulse"])

    trained_models = {}
    for kind in noise_kinds:
        params = artifacts.noise_fits[kind].params
        train_loader, val_loader, _ = build_loaders(
            artifacts.distribution, kind, params, seed=seed,
            batch=batch, train_n=train_n, val_n=val_n, test_n=test_n, window_len=window_len,
        )
        model = build_model(model_name, latent_dim=latent_dim, window_len=window_len)
        ckpt_dir = paths["checkpoints"] / kind
        train(model, train_loader, val_loader, epochs=epochs, lr=lr,
              patience=patience, log_path=ckpt_dir)
        trained_models[kind] = model

    rows = []
    matrix = np.zeros((len(noise_kinds), len(noise_kinds)), dtype=float)
    for i, train_kind in enumerate(noise_kinds):
        for j, eval_kind in enumerate(noise_kinds):
            eval_params = artifacts.noise_fits[eval_kind].params
            test_loader = build_test_loader(
                artifacts.distribution, eval_kind, eval_params, seed=seed + 999,
                batch=batch, n=test_n, window_len=window_len,
            )
            res = evaluate(trained_models[train_kind], test_loader)
            rows.append({
                "train_noise": train_kind,
                "eval_noise": eval_kind,
                "test_mse": res["mse"],
                "snr_improvement_db": res["snr_improvement_db"],
                "snr_output_db": res["snr_output_db"],
            })
            matrix[i, j] = res["snr_improvement_db"]

    df = pd.DataFrame(rows)
    df.to_csv(paths["tables"] / "noise_robustness.csv", index=False)

    plot_heatmap(matrix, row_labels=noise_kinds, col_labels=noise_kinds,
                 out_path=paths["figures"] / "robustness_heatmap.png",
                 title=f"SNR improvement (dB) | model={model_name}",
                 cbar_label="SNR improvement (dB)")

    diag_mean = float(np.mean(np.diag(matrix)))
    off_mean = float((matrix.sum() - np.trace(matrix)) / (matrix.size - len(noise_kinds)))
    print(f"EXP5 done | model={model_name} | diag_mean_snr_imp={diag_mean:.2f}dB | off_diag_mean_snr_imp={off_mean:.2f}dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
