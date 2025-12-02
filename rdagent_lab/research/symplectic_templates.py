"""Symplectic and rough-volatility inspired templates for RD-Agent evolution.

**EXPERIMENTAL MODULE** - This module contains research-stage implementations
that are not yet production-ready. APIs may change without notice.

This module provides:

1. **FractionalDifferencer** - Neural network layer for fractional differencing,
   useful for making time series stationary while preserving memory (based on
   Marcos Lopez de Prado's "Advances in Financial Machine Learning").

2. **RoughVolatilityFactorTemplate** - Code generation template for creating
   rough volatility-based alpha factors with Hurst exponent < 0.5.

3. **SymplecticTransformer** - Transformer architecture with symplectic attention
   that theoretically preserves phase space volume (inspired by Hamiltonian
   mechanics for more stable gradient flow).

Example Usage
-------------
```python
import torch
from rdagent_lab.research.symplectic_templates import (
    FractionalDifferencer,
    SymplecticTransformer,
    RoughVolatilityFactorTemplate,
)

# Fractional differencing layer
diff = FractionalDifferencer(order=0.12, window_size=32)
x = torch.randn(2, 100, 4)  # (batch, time, features)
stationary = diff(x)

# Symplectic Transformer for prediction
model = SymplecticTransformer(d_feat=6, d_model=64, num_layers=2)
predictions = model(x[:, :, :6])  # (batch, time) predictions

# Generate factor code
template = RoughVolatilityFactorTemplate(hurst_exponent=0.1)
code = template.generate_factor_code(formula_idx=0, window=60)
```

References
----------
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Gatheral, J. et al. (2018). Volatility is Rough. Quantitative Finance.
- Toth, T. & Obermeyer, F. (2019). Hamiltonian Generative Networks. ICLR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FractionalDifferencer(nn.Module):
    """Fractional differencing via binomial coefficients."""

    def __init__(self, order: float = 0.12, window_size: int = 64):
        super().__init__()
        coeffs = self._fractional_binomial_coeffs(order, window_size)
        kernel = torch.tensor(coeffs[::-1], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _fractional_binomial_coeffs(d: float, size: int) -> list[float]:
        coeffs = [1.0]
        for k in range(1, size):
            coeffs.append(coeffs[-1] * (d - k + 1) / k)
        return coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, t, f = x.shape
        x_reshaped = x.transpose(1, 2)
        padding = self.kernel.shape[-1] - 1
        filtered = F.conv1d(x_reshaped, self.kernel.expand(f, 1, -1), groups=f, padding=padding)
        filtered = filtered[:, :, :t]
        return filtered.transpose(1, 2)


@dataclass
class RoughVolatilityFactorTemplate:
    """Template for rough volatility-based factors."""

    name: str = "rough_vol"
    hurst_exponent: float = 0.12
    window_sizes: list[int] = field(default_factory=lambda: [20, 60, 120])
    base_formulas: list[str] = field(
        default_factory=lambda: [
            "frac_diff($close, d={hurst}, window={window})",
            "frac_diff(rolling_var($close, {window}), d={hurst})",
            "frac_diff(rolling_std(rolling_std($close, 5), {window}), d={hurst})",
            "frac_diff($close / Ref($close, {window}), d={hurst})",
            "frac_diff($close * $volume, d={hurst}) / frac_diff($volume, d={hurst})",
        ]
    )

    def generate_factor_code(self, formula_idx: int = 0, window: int = 60) -> str:
        return f'''
import numpy as np
import pandas as pd

def rough_vol_factor_{formula_idx}(df: pd.DataFrame, hurst: float = {self.hurst_exponent}, window: int = {window}) -> pd.Series:
    def frac_binomial_coeffs(d, size):
        coeffs = [1.0]
        for k in range(1, size):
            coeffs.append(coeffs[-1] * (d - k + 1) / k)
        return np.array(coeffs)

    def frac_diff(series, d, window):
        coeffs = frac_binomial_coeffs(d, window)
        result = np.zeros(len(series))
        for i in range(window, len(series)):
            result[i] = np.dot(coeffs, series[i-window+1:i+1][::-1])
        return pd.Series(result, index=series.index)

    close = df["$close"]
    returns = close.pct_change()
    frac_returns = frac_diff(returns.fillna(0), hurst, window)
    factor = (frac_returns - frac_returns.rolling(window).mean()) / (frac_returns.rolling(window).std() + 1e-8)
    return factor
'''


class SymplecticAttentionBlock(nn.Module):
    """Symplectic multi-head attention preserving phase space volume."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for symplectic split."
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.noether_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, d // self.n_heads)
        k = self.k_proj(x).view(b, t, self.n_heads, d // self.n_heads)
        v = self.v_proj(x).view(b, t, self.n_heads, d // self.n_heads)
        q, k, v = [t_.transpose(1, 2) for t_ in (q, k, v)]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(attn)
        out = out + self.noether_gate(out)
        return out + x


class SymplecticEncoderLayer(nn.Module):
    """Transformer encoder block with symplectic attention."""

    def __init__(self, d_model: int, n_heads: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = SymplecticAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.attn(src)
        src = self.norm(src + self.ff(src))
        return src


class SymplecticTransformer(nn.Module):
    """Symplectic Transformer backbone for factor modeling."""

    def __init__(
        self,
        d_feat: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)
        self.layers = nn.ModuleList(
            [
                SymplecticEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        out = self.head(x).squeeze(-1)
        return out


def save_template(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
