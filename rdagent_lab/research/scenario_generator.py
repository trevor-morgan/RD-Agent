"""Market Scenario Generator using diffusion models (experimental).

This is carried over from qlib-quant-lab to keep the research tooling in-tree.
It is intentionally self-contained; dependencies are behind the `quant-lab` and
`rl` extras.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        t = self.time_proj(t_emb)[:, :, None]
        h = h + t
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.residual(x)


class TimeSeriesUNet(nn.Module):
    """U-Net architecture for 1D time series used in DDPM."""

    def __init__(self, in_channels: int, model_dim: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.enc1 = ConvBlock(in_channels, model_dim, time_dim)
        self.down1 = nn.Conv1d(model_dim, model_dim, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(model_dim, model_dim * 2, time_dim)
        self.down2 = nn.Conv1d(model_dim * 2, model_dim * 2, 3, stride=2, padding=1)
        self.mid1 = ConvBlock(model_dim * 2, model_dim * 4, time_dim)
        self.mid2 = ConvBlock(model_dim * 4, model_dim * 4, time_dim)
        self.up2 = nn.ConvTranspose1d(model_dim * 4, model_dim * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(model_dim * 4, model_dim * 2, time_dim)
        self.up1 = nn.ConvTranspose1d(model_dim * 2, model_dim, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(model_dim * 2, model_dim, time_dim)
        self.out = nn.Conv1d(model_dim, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)
        m = self.mid1(d2, t_emb)
        m = self.mid2(m, t_emb)
        u2 = self.up2(m)
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.interpolate(u2, size=e2.shape[-1], mode="linear", align_corners=False)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)
        u1 = self.up1(u2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.interpolate(u1, size=e1.shape[-1], mode="linear", align_corners=False)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)
        return self.out(u1)


class DiffusionScheduler:
    """Noise schedule for DDPM."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: Literal["linear", "cosine"] = "linear",
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif schedule == "cosine":
            steps = torch.arange(num_timesteps, device=device)
            betas = torch.cos((steps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = 1 - betas / betas[0]
            betas = betas * (beta_end - beta_start) + beta_start
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.register_buffers(betas)

    def register_buffers(self, betas: torch.Tensor) -> None:
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        sqrt_ab = torch.sqrt(self.alpha_bars[t])[:, None, None]
        sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_bars[t])[:, None, None]
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise


class DiffusionModel(nn.Module):
    """DDPM for time series generation."""

    def __init__(self, in_channels: int, model_dim: int = 64, time_dim: int = 128, num_timesteps: int = 1000):
        super().__init__()
        self.unet = TimeSeriesUNet(in_channels, model_dim=model_dim, time_dim=time_dim)
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noisy = self.scheduler.add_noise(x0, t, noise=noise)
        pred_noise = self.unet(noisy, t)
        return pred_noise

    def sample(self, shape: tuple[int, int, int], device: str = "cpu") -> torch.Tensor:
        self.to(device)
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            with torch.no_grad():
                pred_noise = self.unet(x, t_tensor)
            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alpha_bars[t]
            beta = self.scheduler.betas[t]
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) + torch.sqrt(
                beta
            ) * noise
        return x.cpu()
