from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

_EPS = 1e-12
_SNR_CLIP_DB = 100.0


def _to_2d(x: Tensor | np.ndarray) -> Tensor:
    t = x if isinstance(x, Tensor) else torch.as_tensor(x)
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 3:
        return t.squeeze(1)
    return t


def mse(clean: Tensor | np.ndarray, recon: Tensor | np.ndarray) -> float:
    c = _to_2d(clean).float()
    r = _to_2d(recon).float()
    return float(((c - r) ** 2).mean().item())


def snr_db(clean: Tensor | np.ndarray, recon: Tensor | np.ndarray) -> float:
    c = _to_2d(clean).float()
    r = _to_2d(recon).float()
    diff = c - r
    sig_pow = (c * c).sum(dim=1)
    err_pow = (diff * diff).sum(dim=1)
    err_pow = torch.clamp(err_pow, min=_EPS)
    snrs = 10.0 * torch.log10(torch.clamp(sig_pow, min=_EPS) / err_pow)
    snrs = torch.clamp(snrs, min=-_SNR_CLIP_DB, max=_SNR_CLIP_DB)
    return float(snrs.mean().item())


def snr_improvement_db(clean, noisy, recon) -> float:
    return snr_db(clean, recon) - snr_db(clean, noisy)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str | torch.device | None = None) -> dict:
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    model.to(device).eval()

    cleans, noisies, recons = [], [], []
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        recon = model(noisy)
        cleans.append(clean.detach().cpu())
        noisies.append(noisy.detach().cpu())
        recons.append(recon.detach().cpu())

    clean_t = torch.cat(cleans, dim=0)
    noisy_t = torch.cat(noisies, dim=0)
    recon_t = torch.cat(recons, dim=0)

    return {
        "mse": mse(clean_t, recon_t),
        "snr_input_db": snr_db(clean_t, noisy_t),
        "snr_output_db": snr_db(clean_t, recon_t),
        "snr_improvement_db": snr_improvement_db(clean_t, noisy_t, recon_t),
    }


@torch.no_grad()
def collect_examples(model: nn.Module, loader: DataLoader, n: int = 4,
                     device: str | torch.device | None = None) -> list[dict]:
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    model.to(device).eval()

    examples: list[dict] = []
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        recon = model(noisy)
        for i in range(noisy.size(0)):
            examples.append({
                "clean": clean[i, 0].detach().cpu().numpy(),
                "noisy": noisy[i, 0].detach().cpu().numpy(),
                "recon": recon[i, 0].detach().cpu().numpy(),
            })
            if len(examples) >= n:
                return examples
    return examples
