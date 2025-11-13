"""
SEMANTIC SPACE TRADING NEURAL NETWORK

The Universe is Semantic Space:
- Market states are embeddings in continuous semantic space
- Similar market conditions cluster together
- Transformers learn the language of markets
- Attention captures relationships across assets and time

Architecture:
1. Market State Embeddings: Map raw features to semantic space
2. Multi-Head Self-Attention: Capture temporal dependencies
3. Cross-Asset Attention: Learn relationships between assets
4. Positional Encoding: Encode time information
5. Prediction Head: Map semantic state to future returns

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from datetime import datetime


class PositionalEncoding(nn.Module):
    """
    Inject temporal information into embeddings.
    Markets have time structure - this captures it.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, d_model]
        """
        return x + self.pe[:x.size(0)]


class MarketStateEmbedding(nn.Module):
    """
    Map raw market features to semantic space.

    Input: [returns, volumes, correlations]
    Output: Dense embedding in semantic space
    """

    def __init__(
        self,
        n_tickers: int,
        n_correlations: int,
        embed_dim: int = 256
    ):
        super().__init__()

        self.n_tickers = n_tickers
        self.n_correlations = n_correlations

        # Feature dimensions
        input_dim = n_tickers * 2 + n_correlations  # returns, volumes, correlations

        # Embedding layers
        self.fc1 = nn.Linear(input_dim, embed_dim * 2)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, returns, volumes, correlations):
        """
        Args:
            returns: [batch, n_tickers]
            volumes: [batch, n_tickers]
            correlations: [batch, n_correlations]

        Returns:
            embedding: [batch, embed_dim]
        """
        # Concatenate all features
        x = torch.cat([returns, volumes, correlations], dim=1)

        # Embed into semantic space
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln(x)

        return x


class CrossAssetAttention(nn.Module):
    """
    Learn relationships between assets.

    Example: When tech stocks fall, financials may rise.
    This captures those semantic relationships.
    """

    def __init__(self, embed_dim: int, n_heads: int = 8):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=0.1,
            batch_first=True
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, embed_dim]

        Returns:
            attended: [batch, seq_len, embed_dim]
        """
        # Self-attention over sequence
        attn_out, _ = self.attention(x, x, x)

        # Residual connection
        x = x + self.dropout(attn_out)
        x = self.ln(x)

        return x


class SemanticSpaceNetwork(nn.Module):
    """
    Complete Semantic Space Trading Network.

    The network learns:
    1. How to embed market states into semantic space
    2. How market states evolve over time (attention)
    3. How assets relate to each other (cross-asset attention)
    4. How to predict future returns from current semantic state
    """

    def __init__(
        self,
        n_tickers: int,
        n_correlations: int,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        sequence_length: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_tickers = n_tickers
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length

        # 1. Embedding: Map raw features to semantic space
        self.embedding = MarketStateEmbedding(
            n_tickers=n_tickers,
            n_correlations=n_correlations,
            embed_dim=embed_dim
        )

        # 2. Positional encoding: Add time information
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=5000)

        # 3. Transformer layers: Learn temporal and cross-asset patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Cross-asset attention
        self.cross_asset_attention = CrossAssetAttention(embed_dim, n_heads)

        # 5. Prediction head: Map semantic state to future returns
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, n_tickers)  # Predict return for each ticker

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, returns, volumes, correlations):
        """
        Args:
            returns: [batch, seq_len, n_tickers]
            volumes: [batch, seq_len, n_tickers]
            correlations: [batch, seq_len, n_correlations]

        Returns:
            predictions: [batch, n_tickers]
        """
        batch_size, seq_len, _ = returns.shape

        # 1. Embed each timestep into semantic space
        embeddings = []
        for t in range(seq_len):
            emb = self.embedding(
                returns[:, t, :],
                volumes[:, t, :],
                correlations[:, t, :]
            )
            embeddings.append(emb)

        # Stack: [batch, seq_len, embed_dim]
        x = torch.stack(embeddings, dim=1)

        # 2. Add positional encoding
        # Transpose for positional encoder: [seq_len, batch, embed_dim]
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Back to [batch, seq_len, embed_dim]

        # 3. Transformer: Learn temporal patterns
        x = self.transformer(x)

        # 4. Cross-asset attention: Learn asset relationships
        x = self.cross_asset_attention(x)

        # 5. Take final state
        x = x[:, -1, :]  # [batch, embed_dim]

        # 6. Predict future returns
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        predictions = self.fc2(x)  # [batch, n_tickers]

        return predictions

    def get_semantic_embedding(self, returns, volumes, correlations):
        """
        Get the semantic space embedding for a market state.
        Useful for visualization and analysis.
        """
        with torch.no_grad():
            batch_size, seq_len, _ = returns.shape

            # Embed each timestep
            embeddings = []
            for t in range(seq_len):
                emb = self.embedding(
                    returns[:, t, :],
                    volumes[:, t, :],
                    correlations[:, t, :]
                )
                embeddings.append(emb)

            x = torch.stack(embeddings, dim=1)

            # Process through transformer
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)
            x = self.transformer(x)
            x = self.cross_asset_attention(x)

            # Final semantic state
            semantic_state = x[:, -1, :]

            return semantic_state


def create_network(dataset: dict) -> SemanticSpaceNetwork:
    """Create network from dataset."""

    n_tickers = dataset['n_tickers']
    n_correlations = dataset['correlations'].shape[1]

    print("=" * 80)
    print("CREATING SEMANTIC SPACE NETWORK")
    print("=" * 80)
    print()
    print("Architecture:")
    print(f"  Input tickers: {n_tickers}")
    print(f"  Correlation features: {n_correlations}")
    print(f"  Embedding dimension: 256")
    print(f"  Attention heads: 8")
    print(f"  Transformer layers: 4")
    print(f"  Sequence length: 20")
    print()

    network = SemanticSpaceNetwork(
        n_tickers=n_tickers,
        n_correlations=n_correlations,
        embed_dim=256,
        n_heads=8,
        n_layers=4,
        sequence_length=20,
        dropout=0.1
    )

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    print("✓ Network created")

    return network


if __name__ == '__main__':
    # Test network creation
    print("Testing Semantic Space Network...")
    print()

    # Create dummy dataset
    dummy_dataset = {
        'n_tickers': 23,
        'correlations': np.zeros((100, 253)),
    }

    network = create_network(dummy_dataset)

    # Test forward pass
    batch_size = 32
    seq_len = 20
    n_tickers = 23
    n_corr = 253

    returns = torch.randn(batch_size, seq_len, n_tickers)
    volumes = torch.randn(batch_size, seq_len, n_tickers)
    correlations = torch.randn(batch_size, seq_len, n_corr)

    print("Testing forward pass...")
    predictions = network(returns, volumes, correlations)

    print(f"Input shape: [{batch_size}, {seq_len}, {n_tickers}]")
    print(f"Output shape: {predictions.shape}")
    print()
    print("✓ Forward pass successful")
    print()

    # Test semantic embedding extraction
    print("Testing semantic embedding extraction...")
    semantic_state = network.get_semantic_embedding(returns, volumes, correlations)
    print(f"Semantic state shape: {semantic_state.shape}")
    print()
    print("✓ Semantic embedding extraction successful")
    print()
    print("=" * 80)
    print("READY FOR TRAINING")
    print("=" * 80)
