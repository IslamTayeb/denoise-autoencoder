from __future__ import annotations

import torch
from torch import Tensor, nn


class RNNAE(nn.Module):
    def __init__(self, window_len: int = 128, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.window_len = window_len
        self.latent_dim = latent_dim
        self.hidden = hidden

        self.encoder_gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.enc_proj = nn.Linear(hidden, latent_dim)

        self.dec_proj = nn.Linear(latent_dim, hidden)
        self.decoder_gru = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=1, batch_first=True)
        self.out_proj = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for gru in (self.encoder_gru, self.decoder_gru):
            for name, p in gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def encode(self, x: Tensor) -> Tensor:
        seq = x.transpose(1, 2)                       # (B, L, 1)
        _, h_n = self.encoder_gru(seq)
        h_last = h_n[-1]                              # (B, hidden)
        return self.enc_proj(h_last)

    def decode(self, z: Tensor) -> Tensor:
        h0 = self.dec_proj(z)                         # (B, hidden)
        seq = h0.unsqueeze(1).expand(-1, self.window_len, -1).contiguous()
        out, _ = self.decoder_gru(seq)
        out = self.out_proj(out)                      # (B, L, 1)
        return out.transpose(1, 2)                    # (B, 1, L)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
