from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(default_config: str) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default_config)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(output_dir: Path | str) -> dict[str, Path]:
    name = Path(output_dir).name
    paths = {
        "tables": REPO_ROOT / "results" / "tables",
        "figures": REPO_ROOT / "results" / "figures" / name,
        "checkpoints": REPO_ROOT / "results" / "checkpoints" / name,
        "fits": REPO_ROOT / "results" / "fits",
        "data_cache": REPO_ROOT / "data_cache",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_noise_kind(cfg: dict, artifacts) -> str:
    requested = cfg.get("noise_kind", "auto")
    if requested == "auto":
        return artifacts.best_noise_kind
    return requested


def resolve_best_arch_from_csv(tables_dir: Path) -> str:
    csv_path = tables_dir / "arch_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"need arch comparison results at {csv_path}; run experiment 3 first"
        )
    df = pd.read_csv(csv_path)
    return str(df.sort_values("test_mse").iloc[0]["model"])


def resolve_model_name(cfg: dict, tables_dir: Path) -> str:
    name = cfg.get("model", {}).get("name", "auto")
    if name == "auto":
        return resolve_best_arch_from_csv(tables_dir)
    return name


def resolve_best_latent_from_csv(tables_dir: Path, fallback: int = 16) -> int:
    csv_path = tables_dir / "latent_sweep.csv"
    if not csv_path.exists():
        return fallback
    df = pd.read_csv(csv_path)
    return int(df.sort_values("snr_improvement_db", ascending=False).iloc[0]["latent_dim"])


def resolve_latent(cfg: dict, tables_dir: Path) -> int:
    latent = cfg.get("model", {}).get("latent_dim", 16)
    if latent == "auto":
        return resolve_best_latent_from_csv(tables_dir)
    return int(latent)


def find_checkpoint(checkpoints_root: Path, exp_name: str, subdir: str | None = None) -> Path:
    base = checkpoints_root / exp_name
    if subdir is not None:
        candidate = base / subdir / "best.pt"
    else:
        candidate = base / "best.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"no checkpoint at {candidate}")
    return candidate
