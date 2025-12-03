"""Symplectic physics-informed neural network modules.

This module implements physics-informed neural network components based on
concepts from symplectic geometry and rough volatility theory:

1. **FractionalDifferencer**: Fractional differencing for rough volatility (H ≈ 0.12)
2. **SymplecticAttention**: Phase-space preserving attention mechanism
3. **HamiltonianBlock**: Energy-conserving latent evolution via Hamiltonian dynamics
4. **HolographicMemory**: Complex-valued associative memory for regime detection
5. **SymplecticNet**: Full network combining all components

These are pure PyTorch nn.Module implementations with no external dependencies
beyond torch. They can be used as building blocks for RD-Agent model evolution.

References:
    - Gatheral et al. (2018) "Volatility is Rough"
    - Hairer et al. (2006) "Geometric Numerical Integration"

Example:
    >>> import torch
    >>> from rdagent_lab.models.novel.symplectic import SymplecticNet
    >>> model = SymplecticNet(d_feat=158, d_model=64, n_layers=2)
    >>> x = torch.randn(32, 158)  # batch of 32, 158 features
    >>> out = model(x)  # shape: (32, 1)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FractionalDifferencer(nn.Module):
    """Fractional differencing for rough volatility modeling.

    Implements the fractional differencing operator using Hosking's method
    with binomial coefficients. This captures the long-memory properties
    of financial volatility with Hurst exponent H ≈ 0.12.

    Args:
        order: Fractional differencing order (typically 0.12 for rough vol)
        window_size: Size of the convolution kernel

    Shape:
        - Input: (batch, seq_len, features)
        - Output: (batch, seq_len, features)
    """

    def __init__(self, order: float = 0.12, window_size: int = 64):
        super().__init__()
        coeffs = self._fractional_binomial_coeffs(order, window_size)
        kernel = torch.tensor(coeffs[::-1], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _fractional_binomial_coeffs(d: float, size: int) -> list[float]:
        """Compute fractional binomial coefficients for order d."""
        coeffs = [1.0]
        for k in range(1, size):
            coeffs.append(coeffs[-1] * (d - k + 1) / k)
        return coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fractional differencing along time dimension."""
        b, t, f = x.shape
        x_reshaped = x.transpose(1, 2)  # (b, f, t)
        padding = self.kernel.shape[-1] - 1

        filtered = F.conv1d(
            x_reshaped,
            self.kernel.expand(f, 1, -1),
            groups=f,
            padding=padding,
        )
        filtered = filtered[:, :, :t]
        return filtered.transpose(1, 2)


class SymplecticAttention(nn.Module):
    """Symplectic multi-head attention preserving phase space volume.

    Implements attention with a symplectic transformation that preserves
    the phase space structure (q, p) → (p, -q), inspired by Hamiltonian
    mechanics and Liouville's theorem.

    Args:
        d_model: Model dimension (must be even for symplectic split)
        n_heads: Number of attention heads
        dropout: Dropout probability

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for symplectic split")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Noether gate for conservation constraint
        self.noether_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, t, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn = torch.matmul(weights, v)

        attn = attn.transpose(1, 2).contiguous().view(b, t, d)

        # Symplectic transformation: (q, p) → (p, -q)
        q_part, p_part = torch.chunk(attn, 2, dim=-1)
        symplectic = torch.cat([p_part, -q_part], dim=-1)
        out = self.out_proj(symplectic)

        # Conservation constraint via Noether gate
        conserved = self.noether_gate(out)
        out = out - conserved.mean(dim=1, keepdim=True)

        return out


class HamiltonianBlock(nn.Module):
    """Hamiltonian ODE block with learned energy-based dynamics.

    Evolves latent states using learned Hamiltonian dynamics that
    approximately preserve energy (symplectic structure).

    Args:
        latent_dim: Dimension of latent space (must be even)
        hidden_dim: Hidden dimension for energy network
        steps: Number of integration steps

    Shape:
        - Input: (batch, latent_dim)
        - Output: (batch, latent_dim)
    """

    def __init__(self, latent_dim: int, hidden_dim: int, steps: int = 5):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even for Hamiltonian split")

        self.dim = latent_dim // 2
        self.steps = steps

        # Hamiltonian energy network
        self.H = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """Evolve through Hamiltonian dynamics using learned drift."""
        q, p = torch.chunk(x, 2, dim=-1)
        batch_size = x.shape[0]
        t = torch.zeros(batch_size, 1, device=x.device)

        for _ in range(self.steps):
            # Compute energy and use as gating mechanism
            state = torch.cat([q, p, t], dim=-1)
            H = self.H(state)  # (batch, 1)
            gate = torch.sigmoid(H)

            # Learned drift approximating symplectic dynamics
            p_update = -torch.tanh(q) * gate
            q_update = torch.tanh(p) * gate

            p = p + dt * p_update
            q = q + dt * q_update
            t = t + dt

        return torch.cat([q, p], dim=-1)


class HolographicMemory(nn.Module):
    """Complex-valued associative memory for regime detection.

    Uses a Hermitian inner product over a learnable memory bank to
    identify and retrieve relevant market regime patterns.

    Args:
        dim: Input dimension (must be even for complex split)
        memory_size: Number of memory slots

    Shape:
        - Input: (batch, ..., dim)
        - Output: (batch, ..., dim)
    """

    def __init__(self, dim: int, memory_size: int = 512):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even for complex representation")

        self.dim = dim
        self.memory_size = memory_size
        self.head_dim = dim // 2

        # Learnable memory bank (real and imaginary parts)
        self.keys_real = nn.Parameter(torch.randn(memory_size, self.head_dim) * 0.02)
        self.keys_imag = nn.Parameter(torch.randn(memory_size, self.head_dim) * 0.02)
        self.values_real = nn.Parameter(torch.randn(memory_size, self.head_dim) * 0.02)
        self.values_imag = nn.Parameter(torch.randn(memory_size, self.head_dim) * 0.02)

        self.temperature = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_flat = x.view(-1, self.dim)

        q, p = torch.chunk(x_flat, 2, dim=-1)

        # Hermitian inner product
        score_real = torch.matmul(q, self.keys_real.t())
        score_imag = torch.matmul(p, self.keys_imag.t())
        energy = score_real + score_imag

        attn = F.softmax(energy * self.temperature, dim=-1)

        out_real = torch.matmul(attn, self.values_real)
        out_imag = torch.matmul(attn, self.values_imag)

        return torch.cat([out_real, out_imag], dim=-1).view(input_shape)


class SymplecticNet(nn.Module):
    """Full symplectic neural network combining all physics-inspired components.

    This network combines:
    1. Fractional differencing for rough volatility
    2. Symplectic attention layers
    3. Hamiltonian dynamics evolution
    4. Holographic memory for regime detection

    Args:
        d_feat: Number of input features
        d_model: Internal model dimension (will be rounded to even)
        n_heads: Number of attention heads
        n_layers: Number of symplectic attention layers
        hamiltonian_steps: Integration steps for Hamiltonian block
        dropout: Dropout probability
        use_fractional_diff: Whether to apply fractional differencing
        hurst_exponent: Hurst exponent for fractional differencing
        use_holographic_memory: Whether to use holographic memory
        memory_size: Size of holographic memory bank

    Shape:
        - Input: (batch, d_feat) or (batch, seq_len, d_feat)
        - Output: (batch, 1)

    Example:
        >>> model = SymplecticNet(d_feat=158, d_model=64)
        >>> x = torch.randn(32, 158)
        >>> pred = model(x)  # (32, 1)
    """

    def __init__(
        self,
        d_feat: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        hamiltonian_steps: int = 5,
        dropout: float = 0.1,
        use_fractional_diff: bool = True,
        hurst_exponent: float = 0.12,
        use_holographic_memory: bool = True,
        memory_size: int = 512,
    ):
        super().__init__()

        self.d_feat = d_feat
        # Ensure even d_model for symplectic split
        self.d_model = d_model if d_model % 2 == 0 else d_model + 1
        self.use_fractional_diff = use_fractional_diff
        self.use_holographic_memory = use_holographic_memory

        # Fractional differencing layer
        if use_fractional_diff:
            self.frac_diff = FractionalDifferencer(order=hurst_exponent)

        # Input projection
        self.input_proj = nn.Linear(d_feat, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)

        # Symplectic attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": SymplecticAttention(self.d_model, n_heads, dropout),
                "ff": nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.d_model * 4, self.d_model),
                    nn.Dropout(dropout),
                ),
                "norm1": nn.LayerNorm(self.d_model),
                "norm2": nn.LayerNorm(self.d_model),
            })
            for _ in range(n_layers)
        ])

        # Hamiltonian dynamics
        self.hamiltonian = HamiltonianBlock(
            self.d_model, self.d_model * 2, steps=hamiltonian_steps
        )

        # Holographic memory
        if use_holographic_memory:
            self.memory = HolographicMemory(self.d_model, memory_size)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, d_feat) or (batch, seq_len, d_feat)

        Returns:
            Predictions of shape (batch, 1)
        """
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_feat)

        # Fractional differencing
        if self.use_fractional_diff and x.shape[1] > 1:
            x = self.frac_diff(x)

        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Symplectic attention layers
        for layer in self.layers:
            h = h + layer["attn"](layer["norm1"](h))
            h = h + layer["ff"](layer["norm2"](h))

        # Pool over sequence
        h = h.mean(dim=1)

        # Holographic memory
        if self.use_holographic_memory:
            memory_context = self.memory(h)
            h = h + memory_context

        # Hamiltonian evolution
        h = self.hamiltonian(h)

        # Output
        return self.head(h)


__all__ = [
    "FractionalDifferencer",
    "SymplecticAttention",
    "HamiltonianBlock",
    "HolographicMemory",
    "SymplecticNet",
]
