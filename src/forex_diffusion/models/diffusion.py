from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    # t in [0,1]
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    # Sinusoidal embedding
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SimpleTemporalUNet(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, hidden: int = 256):
        super().__init__()
        self.fc_in = nn.Linear(z_dim, hidden)
        self.fc_t = nn.Linear(128, hidden)
        self.fc_cond = nn.Linear(cond_dim, hidden)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z_t: torch.Tensor, t_embed: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(z_t) + self.fc_t(t_embed) + self.fc_cond(cond)
        return self.mlp(h)


class LatentDiffusion(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, num_steps: int = 20, s: float = 0.008):
        super().__init__()
        self.num_steps = num_steps
        self.s = s
        self.model = SimpleTemporalUNet(z_dim=z_dim, cond_dim=cond_dim)

    def v_to_eps_and_z0(self, v: torch.Tensor, z_t: torch.Tensor, alpha_bar_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = alpha_bar_t
        sqrt_ab = torch.sqrt(ab)
        sqrt_om = torch.sqrt(1 - ab)
        # v = sqrt(ab)*eps - sqrt(1-ab)*z0
        eps = (v + sqrt_om[:, None] * z_t) / sqrt_ab[:, None]
        z0 = (sqrt_ab[:, None] * z_t - sqrt_om[:, None] * v)
        return eps, z0

    def forward(self, z0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Training loss on v-prediction
        B, D = z0.shape
        device = z0.device
        t = torch.randint(0, self.num_steps, (B,), device=device)
        u = (t + torch.rand_like(t.float())) / self.num_steps
        ab = cosine_alpha_bar(u, s=self.s)
        eps = torch.randn_like(z0)
        z_t = torch.sqrt(ab)[:, None] * z0 + torch.sqrt(1 - ab)[:, None] * eps
        t_emb = timestep_embedding(u, 128)
        v_true = torch.sqrt(ab)[:, None] * eps - torch.sqrt(1 - ab)[:, None] * z0
        v_pred = self.model(z_t, t_emb, cond)
        loss = F.mse_loss(v_pred, v_true)
        return loss

    @torch.no_grad()
    def sample_ddim(self, cond: torch.Tensor, steps: int) -> torch.Tensor:
        # Deterministic DDIM-like with v-prediction
        device = cond.device
        B = cond.shape[0]
        D = self.model.mlp[-1].out_features
        z_t = torch.randn(B, D, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps, 0, -1):
            t1 = ts[i]
            t0 = ts[i - 1]
            ab_t1 = cosine_alpha_bar(t1, s=self.s)
            ab_t0 = cosine_alpha_bar(t0, s=self.s)
            t_emb = timestep_embedding(t1.repeat(B), 128)
            v = self.model(z_t, t_emb, cond)
            eps, z0 = self.v_to_eps_and_z0(v, z_t, ab_t1)
            z_t = torch.sqrt(ab_t0)[:, None] * z0 + torch.sqrt(1 - ab_t0)[:, None] * eps
        return z_t


