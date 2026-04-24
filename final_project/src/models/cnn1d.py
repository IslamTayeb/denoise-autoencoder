from __future__ import annotations

import torch
from torch import Tensor, nn


class CNN1DAE(nn.Module):
    def __init__(self, window_len: int = 128, latent_dim: int = 16):
        super().__init__()
        if window_len % 8 != 0:
            raise ValueError(f"window_len must be divisible by 8; got {window_len}")
        self.window_len = window_len
        self.latent_dim = latent_dim
        self.bottleneck_len = window_len // 8

        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_proj = nn.Linear(64 * self.bottleneck_len, latent_dim)
        self.dec_proj = nn.Linear(latent_dim, 64 * self.bottleneck_len)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def encode(self, x: Tensor) -> Tensor:
        h = self.enc_conv(x)
        h = h.flatten(start_dim=1)
        return self.enc_proj(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.dec_proj(z).view(-1, 64, self.bottleneck_len)
        return self.dec_conv(h)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
