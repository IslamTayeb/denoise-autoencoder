from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _epoch_pass(model: nn.Module, loader: DataLoader, loss_fn,
                optimizer: torch.optim.Optimizer | None,
                device: torch.device) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n = 0
    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            recon = model(noisy)
            loss = loss_fn(recon, clean)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        bs = noisy.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    patience: int = 10,
    device: str | torch.device | None = None,
    log_path: Path | str | None = None,
    verbose: bool = False,
) -> dict:
    device = _resolve_device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    start = time.time()

    for epoch in range(epochs):
        train_loss = _epoch_pass(model, train_loader, loss_fn, optimizer, device)
        val_loss = _epoch_pass(model, val_loader, loss_fn, None, device)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val - 1e-6:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            print(f"epoch {epoch:03d}  train={train_loss:.6f}  val={val_loss:.6f}")

        if bad_epochs >= patience:
            break

    train_seconds = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path: Path | None = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = log_path / "best.pt"
        cls_name = type(model).__name__
        torch.save({
            "model_state": model.state_dict(),
            "model_class": cls_name,
            "model_kwargs": {
                "window_len": getattr(model, "window_len", None),
                "latent_dim": getattr(model, "latent_dim", None),
            },
            "best_val": best_val,
            "best_epoch": best_epoch,
            "history": history,
        }, checkpoint_path)

    return {
        "history": history,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "train_seconds": float(train_seconds),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": str(device),
    }


def load_checkpoint(model: nn.Module, checkpoint_path: Path | str,
                   device: str | torch.device | None = None) -> nn.Module:
    device = _resolve_device(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def read_checkpoint_meta(checkpoint_path: Path | str) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    cls = state.get("model_class")
    kwargs = state.get("model_kwargs") or {}
    return {
        "model_class": cls,
        "window_len": kwargs.get("window_len"),
        "latent_dim": kwargs.get("latent_dim"),
        "best_val": state.get("best_val"),
        "best_epoch": state.get("best_epoch"),
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
