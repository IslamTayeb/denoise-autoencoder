from __future__ import annotations

import torch

from src.models import build_model
from src.models.cnn1d import CNN1DAE
from src.models.mlp import MLPAE
from src.models.rnn import RNNAE


def _check_io_and_grad(model_cls, latent_dim: int = 16, window_len: int = 128):
    model = model_cls(window_len=window_len, latent_dim=latent_dim)
    x = torch.randn(4, 1, window_len, requires_grad=False)
    y = model(x)
    assert y.shape == (4, 1, window_len)
    loss = ((y - x) ** 2).mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grad_norms and all(n >= 0 for n in grad_norms)
    assert any(n > 0 for n in grad_norms)


def test_mlp_io_and_grad():
    _check_io_and_grad(MLPAE)


def test_cnn_io_and_grad():
    _check_io_and_grad(CNN1DAE)


def test_rnn_io_and_grad():
    _check_io_and_grad(RNNAE)


def test_factory():
    for name in ("mlp", "cnn1d", "rnn"):
        m = build_model(name, latent_dim=8, window_len=64)
        x = torch.randn(2, 1, 64)
        assert m(x).shape == (2, 1, 64)
