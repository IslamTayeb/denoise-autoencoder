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
    resolve_noise_kind,
    set_seed,
)
from src.data.synthetic import build_loaders
from src.eval import collect_examples, evaluate
from src.models import build_model
from src.pipeline import build_pipeline_artifacts
from src.train import count_parameters, train
from src.viz import plot_histories_overlay, plot_triplet


def main() -> int:
    args = parse_args(default_config="configs/exp_3_arch.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_3_arch"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"])
    noise_kind = resolve_noise_kind(cfg, artifacts)
    noise_params = artifacts.noise_fits[noise_kind].params

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    batch = train_cfg.get("batch", 64)
    lr = train_cfg.get("lr", 1e-3)
    patience = train_cfg.get("patience", 10)
    train_n = train_cfg.get("train_n", 8000)
    val_n = train_cfg.get("val_n", 1000)
    test_n = train_cfg.get("test_n", 1000)

    window_len = cfg.get("window_len", 128)
    latent_dim = int(cfg.get("model", {}).get("latent_dim", 16))
    model_names = cfg.get("models", ["mlp", "cnn1d"])

    train_loader, val_loader, test_loader = build_loaders(
        artifacts.distribution, noise_kind, noise_params, seed=seed,
        batch=batch, train_n=train_n, val_n=val_n, test_n=test_n, window_len=window_len,
    )

    rows = []
    histories = {}
    triplet_examples = None

    for name in model_names:
        model = build_model(name, latent_dim=latent_dim, window_len=window_len)
        n_params = count_parameters(model)
        ckpt_dir = paths["checkpoints"] / name
        result = train(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, patience=patience, log_path=ckpt_dir,
        )
        eval_result = evaluate(model, test_loader)
        rows.append({
            "model": name,
            "n_params": n_params,
            "train_loss_final": result["history"]["train_loss"][-1],
            "val_loss_best": result["best_val"],
            "test_mse": eval_result["mse"],
            "snr_input_db": eval_result["snr_input_db"],
            "snr_output_db": eval_result["snr_output_db"],
            "snr_improvement_db": eval_result["snr_improvement_db"],
            "train_seconds": result["train_seconds"],
            "best_epoch": result["best_epoch"],
        })
        histories[name] = result["history"]

        examples = collect_examples(model, test_loader, n=4)
        if triplet_examples is None:
            triplet_examples = [(ex["clean"], ex["noisy"]) for ex in examples]
        for i, ex in enumerate(examples):
            plot_triplet(ex["clean"], ex["noisy"], ex["recon"],
                         out_path=paths["figures"] / f"triplet_{name}_{i}.png",
                         title=f"{name} | noise={noise_kind} | example {i}")

    df = pd.DataFrame(rows)
    df.to_csv(paths["tables"] / "arch_comparison.csv", index=False)

    plot_histories_overlay(histories, out_path=paths["figures"] / "training_curves.png",
                           title=f"training curves (noise={noise_kind})")

    best = df.sort_values("test_mse").iloc[0]
    print(f"EXP3 done | BEST_ARCH={best['model']} | test_mse={best['test_mse']:.5f} | snr_imp={best['snr_improvement_db']:.2f}dB | noise={noise_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
