"""Simple GRU Seed Model for RD-Agent Evolution.

A simpler baseline model to test the seed model functionality.
Use this if the SymplecticNet is too complex for initial testing.

Usage:
    rdagent fin_model \
        --seed-model ./seed_models/simple_gru.py \
        --seed-hypothesis "Simple 2-layer GRU with attention pooling" \
        --data-region alpaca_us \
        --loop-n 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Simple GRU model for RD-Agent.

    A straightforward baseline with:
    1. 2-layer bidirectional GRU
    2. Attention-based pooling
    3. MLP output head

    Args:
        num_features: Number of input features
        num_timesteps: Number of time steps (optional, for time series)
    """

    def __init__(
        self,
        num_features: int,
        num_timesteps: int = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_size)

        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, features] or [batch, timesteps, features]

        Returns:
            [batch, 1] predictions
        """
        # Handle 2D input (tabular)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Input projection
        h = self.input_proj(x)

        # GRU encoding
        gru_out, _ = self.gru(h)

        # Attention pooling
        attn_weights = self.attention(gru_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)

        # Output
        return self.head(context)


# RD-Agent expects this variable
model_cls = Net


if __name__ == "__main__":
    model = Net(num_features=20, num_timesteps=20)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(32, 20, 20)
    out = model(x)
    print(f"Input {x.shape} -> output {out.shape}")
