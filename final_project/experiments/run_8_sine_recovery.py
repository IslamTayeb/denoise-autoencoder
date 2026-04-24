from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch

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
from src.data.noise import (
    GaussianNoiseParams,
    ImpulseNoiseParams,
    MaskingNoiseParams,
    apply_noise,
)
from src.data.sine_fit import predict_sine
from src.data.windowing import from_windows, to_windows
from src.eval import mse, snr_db
from src.models import build_model
from src.pipeline import build_pipeline_artifacts
from src.train import _resolve_device, load_checkpoint, read_checkpoint_meta
from src.viz import plot_heatmap

CLASS_TO_MODEL_NAME = {"MLPAE": "mlp", "CNN1DAE": "cnn1d", "RNNAE": "rnn"}


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
def _denoise_series(model, series: np.ndarray, window_len: int, device) -> np.ndarray:
    windows, means, stds = to_windows(series, window_len=window_len, stride=1)
    t = torch.from_numpy(windows).float().unsqueeze(1).to(device)
    out = model(t).squeeze(1).detach().cpu().numpy().astype(np.float32)
    return from_windows(out, means, stds, series_len=series.size, stride=1)


def _plot_recovery(t, clean, noisy, recon, out_path: Path, title: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, noisy, label="noisy (sine + Gaussian)", color="tab:orange", alpha=0.55, linewidth=1)
    ax.plot(t, clean, label="ground truth (fitted sine)", color="tab:red", linewidth=2.2)
    ax.plot(t, recon, label="autoencoder reconstruction", color="tab:green", linewidth=2, alpha=0.9)
    ax.set_xlabel("week index")
    ax.set_ylabel("log(1 + pageviews)")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args(default_config="configs/exp_8_sine_recovery.yaml")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_seed(seed)

    paths = ensure_dirs(cfg.get("output_dir", "results/exp_8_sine_recovery"))
    artifacts = build_pipeline_artifacts(cfg, fits_dir=paths["fits"], cache_dir=paths["data_cache"])

    noise_kind = resolve_noise_kind(cfg, artifacts)
    base_params = artifacts.noise_fits[noise_kind].params
    checkpoint_path = _find_checkpoint(cfg, noise_kind, paths)
    meta = read_checkpoint_meta(checkpoint_path)
    model_name = CLASS_TO_MODEL_NAME.get(meta["model_class"] or "")
    if model_name is None:
        raise ValueError(f"unknown model class in checkpoint meta: {meta!r}")
    window_len = int(meta["window_len"] or cfg.get("window_len", 128))
    latent_dim = int(meta["latent_dim"] or 16)

    device = _resolve_device(None)
    model = build_model(model_name, latent_dim=latent_dim, window_len=window_len)
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    n_realizations = int(cfg.get("n_realizations", 20))
    multipliers = cfg.get("intensity_multipliers", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])

    main_rows = []
    sweep_rows = []

    log_df = artifacts.log_df

    for col in log_df.columns:
        T = len(log_df[col])
        if T < window_len:
            continue
        t_idx = np.arange(T, dtype=np.float32)
        y_fit = predict_sine(artifacts.fits[col], t_idx)

        # Baseline: clean sine itself through the AE (no noise added).
        clean_recon = _denoise_series(model, y_fit, window_len, device)
        mse_clean_pass = mse(y_fit, clean_recon)
        snr_clean_pass = snr_db(y_fit, clean_recon)

        # Main metric: corrupt with the fitted-intensity noise, average over realizations.
        per_real_mses_in, per_real_mses_out = [], []
        per_real_snr_in, per_real_snr_out = [], []
        last_noisy = last_recon = None
        for r in range(n_realizations):
            rng = np.random.default_rng(seed + 1000 + r)
            noisy = apply_noise(y_fit, noise_kind, base_params, rng).astype(np.float32)
            recon = _denoise_series(model, noisy, window_len, device)

            per_real_mses_in.append(mse(y_fit, noisy))
            per_real_mses_out.append(mse(y_fit, recon))
            per_real_snr_in.append(snr_db(y_fit, noisy))
            per_real_snr_out.append(snr_db(y_fit, recon))
            last_noisy, last_recon = noisy, recon

        snr_in_mean = float(np.mean(per_real_snr_in))
        snr_out_mean = float(np.mean(per_real_snr_out))
        main_rows.append({
            "article": col,
            "noise_kind": noise_kind,
            "n_realizations": n_realizations,
            "mse_clean_pass": mse_clean_pass,
            "snr_db_clean_pass": snr_clean_pass,
            "mse_input_mean": float(np.mean(per_real_mses_in)),
            "mse_recon_mean": float(np.mean(per_real_mses_out)),
            "mse_recon_std": float(np.std(per_real_mses_out)),
            "snr_input_db_mean": snr_in_mean,
            "snr_recon_db_mean": snr_out_mean,
            "snr_improvement_db": snr_out_mean - snr_in_mean,
        })

        if last_noisy is not None and last_recon is not None:
            _plot_recovery(t_idx, y_fit, last_noisy, last_recon,
                           out_path=paths["figures"] / f"{col}_sine_recovery.png",
                           title=f"{col}: sine recovery | model={model_name} | noise={noise_kind}")

        # Noise-level sweep (single realization per multiplier; cheaper, signal still clear)
        for mult in multipliers:
            scaled = _scale_params(base_params, float(mult))
            rng = np.random.default_rng(seed + 5000 + int(round(float(mult) * 1000)))
            noisy = apply_noise(y_fit, noise_kind, scaled, rng).astype(np.float32)
            recon = _denoise_series(model, noisy, window_len, device)
            snr_in = snr_db(y_fit, noisy)
            snr_out = snr_db(y_fit, recon)
            sweep_rows.append({
                "article": col,
                "intensity_multiplier": float(mult),
                "snr_input_db": snr_in,
                "snr_recon_db": snr_out,
                "snr_improvement_db": snr_out - snr_in,
                "mse_recon": mse(y_fit, recon),
            })

    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(paths["tables"] / "sine_recovery.csv", index=False)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(paths["tables"] / "sine_recovery_levels.csv", index=False)

    # Heatmap of SNR improvement: rows = articles, cols = multipliers.
    articles = list(main_df["article"])
    matrix = np.zeros((len(articles), len(multipliers)), dtype=float)
    for i, art in enumerate(articles):
        for j, mult in enumerate(multipliers):
            row = sweep_df[(sweep_df["article"] == art) & (sweep_df["intensity_multiplier"] == float(mult))]
            if len(row) == 1:
                matrix[i, j] = float(row.iloc[0]["snr_improvement_db"])
    plot_heatmap(
        matrix, row_labels=articles,
        col_labels=[f"{m:g}x" for m in multipliers],
        out_path=paths["figures"] / "sine_recovery_heatmap.png",
        title=f"SNR improvement (dB) | sine recovery sweep | model={model_name}",
        cbar_label="SNR improvement (dB)",
        xlabel="intensity multiplier (1.0 = train)",
        ylabel="article",
    )

    mean_imp = float(main_df["snr_improvement_db"].mean()) if not main_df.empty else float("nan")
    mean_clean = float(main_df["snr_db_clean_pass"].mean()) if not main_df.empty else float("nan")
    print(
        f"EXP8 done | model={model_name} | latent={latent_dim} | noise={noise_kind} "
        f"| mean_snr_imp={mean_imp:.2f}dB | mean_clean_pass_snr={mean_clean:.2f}dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
