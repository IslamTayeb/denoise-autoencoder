from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _ensure_parent(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_triplet(clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray,
                 out_path: Path | str, title: str = "") -> None:
    p = _ensure_parent(out_path)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(clean, color="tab:blue")
    axes[0].set_ylabel("clean")
    axes[1].plot(noisy, color="tab:orange")
    axes[1].set_ylabel("noisy")
    axes[2].plot(recon, color="tab:green")
    axes[2].set_ylabel("recon")
    axes[2].set_xlabel("sample")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_history(history: dict, out_path: Path | str, title: str = "") -> None:
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.get("train_loss", []), label="train")
    ax.plot(history.get("val_loss", []), label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_histories_overlay(histories: dict[str, dict], out_path: Path | str,
                           title: str = "") -> None:
    p = _ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for label, h in histories.items():
        axes[0].plot(h.get("train_loss", []), label=label)
        axes[1].plot(h.get("val_loss", []), label=label)
    axes[0].set_title("train")
    axes[1].set_title("val")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_metric_vs_x(xs, ys, xlabel: str, ylabel: str,
                     out_path: Path | str, title: str = "",
                     vline: float | None = None,
                     marker: str = "o") -> None:
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", label=f"train @ {vline}")
        ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_sine_fit(t, y_real, y_fit, out_path: Path | str, title: str = "") -> None:
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y_real, label="real (log-pageviews)", color="tab:blue", alpha=0.7)
    ax.plot(t, y_fit, label="fitted sine", color="tab:red", linewidth=2)
    ax.set_xlabel("week index")
    ax.set_ylabel("log(1 + pageviews)")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_residual_hist(observed: np.ndarray, synthetics: dict[str, np.ndarray],
                       out_path: Path | str, title: str = "", bins: int = 60) -> None:
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    all_vals = [observed] + list(synthetics.values())
    lo = float(min(np.min(v) for v in all_vals))
    hi = float(max(np.max(v) for v in all_vals))
    edges = np.linspace(lo, hi, bins + 1)

    ax.hist(observed, bins=edges, density=True, alpha=0.5, label="observed", color="black")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for i, (label, vals) in enumerate(synthetics.items()):
        ax.hist(vals, bins=edges, density=True, histtype="step", linewidth=2,
                color=colors[i % len(colors)], label=label)
    ax.set_xlabel("residual value")
    ax.set_ylabel("density")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_qq(observed: np.ndarray, synthetic: np.ndarray,
            out_path: Path | str, title: str = "") -> None:
    p = _ensure_parent(out_path)
    obs = np.sort(np.asarray(observed).ravel())
    syn = np.sort(np.asarray(synthetic).ravel())
    n = min(obs.size, syn.size)
    qs = np.linspace(0, 1, n)
    obs_q = np.quantile(obs, qs)
    syn_q = np.quantile(syn, qs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(obs_q, syn_q, marker=".", linestyle="none", alpha=0.6)
    lo = float(min(obs_q.min(), syn_q.min()))
    hi = float(max(obs_q.max(), syn_q.max()))
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--")
    ax.set_xlabel("observed quantile")
    ax.set_ylabel("synthetic quantile")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                 out_path: Path | str, title: str = "", cbar_label: str = "",
                 xlabel: str = "eval noise", ylabel: str = "train noise") -> None:
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * len(row_labels) + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.mean() else "black")
    fig.colorbar(im, ax=ax, label=cbar_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_real_eval(t, real, fitted, denoised, out_path: Path | str, title: str = "") -> None:
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, real, label="real (log-pageviews)", color="tab:blue", alpha=0.6)
    ax.plot(t, fitted, label="fitted sine", color="tab:red", linewidth=2)
    ax.plot(t, denoised, label="autoencoder reconstruction", color="tab:green", linewidth=2, alpha=0.85)
    ax.set_xlabel("week index")
    ax.set_ylabel("log(1 + pageviews)")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
