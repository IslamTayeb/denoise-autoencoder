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
    resolve_noise_kind,
    set_seed,
)
from src.data.sine_fit import predict_sine
from src.data.windowing import from_windows, to_windows
from src.eval import mse, snr_db, snr_improvement_db
from src.models import build_model
from src.pipeline import build_pipeline_artifacts
from src.train import _resolve_device, load_checkpoint, read_checkpoint_meta
from src.viz import plot_real_eval

CLASS_TO_MODEL_NAME = {"MLPAE": "mlp", "CNN1DAE": "cnn1d", "RNNAE": "rnn"}

import torch


def _find_checkpoint(cfg: dict, noise_kind: str, paths: dict[str, Path]) -> Path:
    explicit = cfg.get("checkpoint", "auto")
    if explicit not in (None, "auto"):
        p = Path(explicit)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            return p

    candidates = [
        paths["checkpoints"].parent / "exp_5_noise_robustness" / noise_kind / "best.pt",
        paths["checkpoints"].parent / "exp_4_latent" / "best.pt",
        paths["checkpoints"].parent / "exp_3_arch" / "cnn1d" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "no usable checkpoint found; tried: " + ", ".join(str(c) for c in candidates)
    )


@torch.no_grad()
def _denoise_windows(model, windows: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(windows).float().unsqueeze(1).to(device)
    out = model(t).squeeze(1).detach().cpu().numpy()
    return out.astype(np.float32)


def main() -> int:
    args = parse_args(default_config="configs/exp_7_real_eval.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_7_real_eval"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"])

    noise_kind = resolve_noise_kind(cfg, artifacts)
    checkpoint_path = _find_checkpoint(cfg, noise_kind, paths)
    meta = read_checkpoint_meta(checkpoint_path)
    model_name = CLASS_TO_MODEL_NAME.get(meta["model_class"] or "")
    if model_name is None:
        raise ValueError(f"unknown model class in checkpoint: {meta!r}")
    window_len = int(meta["window_len"] or cfg.get("window_len", 128))
    latent_dim = int(meta["latent_dim"] or cfg.get("model", {}).get("latent_dim", 16))

    device = _resolve_device(None)
    model = build_model(model_name, latent_dim=latent_dim, window_len=window_len)
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    rows = []
    for col in artifacts.log_df.columns:
        y_real = artifacts.log_df[col].to_numpy(dtype=np.float32)
        if y_real.size < window_len:
            continue
        t_idx = np.arange(y_real.size, dtype=np.float32)
        y_fit = predict_sine(artifacts.fits[col], t_idx)

        windows, means, stds = to_windows(y_real, window_len=window_len, stride=1)
        recon_windows = _denoise_windows(model, windows, device)
        y_denoised = from_windows(recon_windows, means, stds, series_len=y_real.size, stride=1)

        recon_mse = mse(y_fit, y_denoised)
        snr_recon_vs_fit = snr_db(y_fit, y_denoised)
        snr_real_vs_fit = snr_db(y_fit, y_real)
        snr_imp = snr_recon_vs_fit - snr_real_vs_fit
        std_residual = float(np.std(y_real - y_denoised))

        rows.append({
            "article": col,
            "mse_recon_vs_sine": recon_mse,
            "snr_db_recon_vs_sine": snr_recon_vs_fit,
            "snr_db_real_vs_sine": snr_real_vs_fit,
            "snr_improvement_db": snr_imp,
            "std_residual_after_denoise": std_residual,
        })

        plot_real_eval(t_idx, y_real, y_fit, y_denoised,
                       out_path=paths["figures"] / f"{col}_real_eval.png",
                       title=f"{col} | model={model_name} | noise={noise_kind}")

    df = pd.DataFrame(rows)
    df.to_csv(paths["tables"] / "real_eval.csv", index=False)

    mean_imp = float(df["snr_improvement_db"].mean()) if not df.empty else float("nan")
    print(
        f"EXP7 done | model={model_name} | latent={latent_dim} | noise={noise_kind} "
        f"| mean_snr_imp_vs_fitted_sine={mean_imp:.2f}dB "
        f"| caveat: fitted sine is proxy ground truth, not true clean signal"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
