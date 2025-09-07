from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class VAE1D(nn.Module):
    def __init__(self, in_channels: int, patch_len: int = 64, hidden_dim: int = 256, z_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            ConvBlock1D(in_channels, 64, 3, 2, 1),
            ConvBlock1D(64, 128, 3, 2, 1),
            ConvBlock1D(128, 256, 3, 2, 1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

        self.dec_fc = nn.Linear(z_dim, hidden_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, L]
        h = self.enc(x).squeeze(-1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, out_len: int) -> torch.Tensor:
        h = self.dec_fc(z)  # [B, H]
        h = h.unsqueeze(-1)  # [B, H, 1]
        x_hat = self.dec(h)
        # Ensure length
        if x_hat.shape[-1] != out_len:
            x_hat = F.interpolate(x_hat, size=out_len, mode="linear", align_corners=False)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, x.shape[-1])
        return x_hat, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 0.5 * sum(mu^2 + sigma^2 - log sigma^2 - 1)
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1).mean()


