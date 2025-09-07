from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..models.vae import VAE1D
from ..models.diffusion import LatentDiffusion


class ForecastModule(pl.LightningModule):
    def __init__(self, in_channels: int, patch_len: int, z_dim: int, hidden_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.vae = VAE1D(in_channels=in_channels, patch_len=patch_len, hidden_dim=hidden_dim, z_dim=z_dim)
        # Minimal conditional dim placeholder (to be extended with multi-scale embeddings)
        self.diff = LatentDiffusion(z_dim=z_dim, cond_dim=32)
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # batch: dict with 'x' [B,C,L], 'cond' [B,32]
        x = batch['x']
        cond = batch['cond']
        x_hat, mu, logvar = self.vae(x)
        rec_loss = nn.functional.mse_loss(x_hat, x)
        kl = self.vae.kl_divergence(mu, logvar)
        # Use posterior mean as z0
        z0 = mu
        diff_loss = self.diff(z0, cond)
        loss = rec_loss + 1e-3 * kl + diff_loss
        self.log_dict({"train_rec": rec_loss, "train_kl": kl, "train_diff": diff_loss, "train_loss": loss})
        return loss


