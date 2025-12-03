"""Market State neural network modules with topology and complexity features.

This module implements neural network components for market state modeling
using concepts from:

1. **TopologyFeatures**: Betti numbers from correlation networks (legitimate TDA)
2. **CompressionComplexity**: Compression-based predictability measure
3. **ActivityMetrics**: Aggregate momentum and volatility metrics
4. **QuantilePredictor**: Uncertainty quantification via quantile regression
5. **AnomalyDetector**: General anomaly detection head
6. **MarketStateNet**: Full network combining all components

These are pure PyTorch nn.Module implementations with no external dependencies
beyond torch, numpy, and zlib (stdlib).

References:
    - Gidea & Katz (2018) "Topological Data Analysis of Financial Time Series"
    - Kolmogorov Complexity and compression-based measures

Example:
    >>> import torch
    >>> from rdagent_lab.models.novel.market_state import MarketStateNet
    >>> model = MarketStateNet(d_feat=158, d_model=64)
    >>> x = torch.randn(32, 158)  # batch of 32, 158 features
    >>> outputs = model(x)  # dict with prediction, quantiles, anomaly_score
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyFeatures(nn.Module):
    """Extract topological features from correlation networks.

    Computes Betti numbers (connected components, cycles) at multiple
    correlation thresholds. These features have been shown to correlate
    with market fragility and crash risk.

    Args:
        n_assets: Number of assets in the correlation network
        thresholds: Correlation thresholds for graph construction

    Shape:
        - Input: (batch, window, n_assets) return series
        - Output: (batch, n_thresholds * 2) Betti numbers
    """

    def __init__(
        self,
        n_assets: int,
        thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    ):
        super().__init__()
        self.n_assets = n_assets
        self.thresholds = thresholds
        self.output_dim = len(thresholds) * 2

    def _compute_betti_numbers(self, adj: torch.Tensor) -> tuple[int, int]:
        """Compute Betti-0 (components) and Betti-1 (cycles) from adjacency matrix."""
        n = adj.shape[0]
        adj_np = (adj > 0).cpu().numpy().astype(int)
        np.fill_diagonal(adj_np, 0)

        # Find connected components via BFS
        visited = [False] * n
        components = 0

        for start in range(n):
            if not visited[start]:
                components += 1
                stack = [start]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        neighbors = np.where(adj_np[node] > 0)[0]
                        stack.extend(neighbors)

        # Count edges (undirected)
        edges = adj_np.sum() // 2

        # Betti-1 = E - V + C (Euler characteristic)
        betti_0 = components
        betti_1 = max(0, edges - n + components)

        return betti_0, betti_1

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute topology features from return correlations."""
        batch_size = returns.shape[0]
        device = returns.device
        features = torch.zeros(batch_size, self.output_dim, device=device)

        for b in range(batch_size):
            ret = returns[b]  # (window, n_assets)

            # Compute correlation matrix
            ret_centered = ret - ret.mean(dim=0, keepdim=True)
            std = ret_centered.std(dim=0, keepdim=True) + 1e-8
            ret_normalized = ret_centered / std
            corr = torch.mm(ret_normalized.t(), ret_normalized) / ret.shape[0]

            # Compute Betti numbers at each threshold
            for i, thresh in enumerate(self.thresholds):
                adj = (corr.abs() > thresh).float()
                betti_0, betti_1 = self._compute_betti_numbers(adj)
                features[b, i * 2] = betti_0 / self.n_assets
                features[b, i * 2 + 1] = betti_1 / self.n_assets

        return features


class CompressionComplexity(nn.Module):
    """Measure predictability via compression ratio.

    Uses zlib compression as a proxy for Kolmogorov complexity.
    Highly compressible series are more predictable.

    Args:
        n_assets: Number of assets
        precision: Decimal precision for discretization

    Shape:
        - Input: (batch, window, n_assets)
        - Output: (batch, n_assets) complexity scores
    """

    def __init__(self, n_assets: int, precision: int = 4):
        super().__init__()
        self.n_assets = n_assets
        self.precision = precision
        self.output_dim = n_assets

    def _compression_ratio(self, series: np.ndarray) -> float:
        """Compute compression ratio for a series."""
        discretized = (series * (10 ** self.precision)).astype(np.int32)
        byte_data = discretized.tobytes()
        compressed = zlib.compress(byte_data, level=9)
        return len(compressed) / (len(byte_data) + 1e-8)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute complexity features per asset."""
        batch_size = returns.shape[0]
        device = returns.device
        features = torch.zeros(batch_size, self.n_assets, device=device)
        returns_np = returns.cpu().numpy()

        for b in range(batch_size):
            for a in range(self.n_assets):
                series = returns_np[b, :, a]
                ratio = self._compression_ratio(series)
                features[b, a] = min(ratio, 1.0)

        return features


class ActivityMetrics(nn.Module):
    """Aggregate activity metrics across assets.

    Computes momentum and volatility-based metrics that indicate
    market-wide activity levels.

    Args:
        n_assets: Number of assets

    Shape:
        - Input: (batch, window, n_assets)
        - Output: (batch, 4) [mean_momentum, momentum_dispersion, mean_vol, vol_of_vol]
    """

    def __init__(self, n_assets: int):
        super().__init__()
        self.n_assets = n_assets
        self.output_dim = 4

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute activity metrics."""
        # Per-asset metrics
        momentum = returns.sum(dim=1)  # (batch, n_assets)
        volatility = returns.std(dim=1)  # (batch, n_assets)

        # Aggregate metrics
        mean_momentum = momentum.mean(dim=1, keepdim=True)
        momentum_dispersion = momentum.std(dim=1, keepdim=True)
        mean_volatility = volatility.mean(dim=1, keepdim=True)
        vol_of_vol = volatility.std(dim=1, keepdim=True)

        return torch.cat([
            mean_momentum,
            momentum_dispersion,
            mean_volatility,
            vol_of_vol,
        ], dim=1)


class QuantilePredictor(nn.Module):
    """Predict return quantiles for uncertainty estimation.

    Outputs multiple quantiles using pinball loss for training.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        quantiles: Quantile levels to predict

    Shape:
        - Input: (batch, input_dim)
        - Output: (batch, n_quantiles)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict quantiles."""
        return self.network(x)

    def pinball_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pinball (quantile) loss."""
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - pred[:, i]
            loss = torch.where(
                error >= 0,
                q * error,
                (q - 1) * error
            )
            losses.append(loss.mean())
        return sum(losses) / len(losses)


class AnomalyDetector(nn.Module):
    """Detect unusual market patterns.

    General anomaly detection head outputting probability score.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension

    Shape:
        - Input: (batch, input_dim)
        - Output: (batch, 1) anomaly probability
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly probability."""
        return self.network(x)


class MarketStateNet(nn.Module):
    """Full market state network combining topology, complexity, and activity features.

    This network combines multiple feature extraction approaches:
    1. Transformer encoder for main features
    2. Optional topology features from correlation graphs
    3. Optional compression-based complexity features
    4. Optional activity metrics (momentum, volatility)
    5. Quantile prediction for uncertainty
    6. Anomaly detection head

    Args:
        d_feat: Number of input features
        n_assets: Number of assets (for topology/complexity)
        d_model: Internal model dimension
        n_layers: Number of transformer layers
        dropout: Dropout probability
        use_topology: Whether to use topology features
        use_complexity: Whether to use compression complexity
        use_activity: Whether to use activity metrics
        topo_thresholds: Correlation thresholds for topology
        quantiles: Quantile levels for prediction

    Shape:
        - Input x: (batch, d_feat) main features
        - Input returns_window: (batch, window, n_assets) optional
        - Output: dict with "prediction", "quantiles", "anomaly_score", "latent"

    Example:
        >>> model = MarketStateNet(d_feat=158, d_model=64)
        >>> x = torch.randn(32, 158)
        >>> outputs = model(x)
        >>> pred = outputs["prediction"]  # (32, 1)
    """

    def __init__(
        self,
        d_feat: int,
        n_assets: int = 500,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_topology: bool = True,
        use_complexity: bool = True,
        use_activity: bool = True,
        topo_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()

        self.d_feat = d_feat
        self.n_assets = n_assets
        self.use_topology = use_topology
        self.use_complexity = use_complexity
        self.use_activity = use_activity

        # Feature extractors
        if use_topology:
            self.topology = TopologyFeatures(n_assets, topo_thresholds)
        if use_complexity:
            self.complexity = CompressionComplexity(n_assets)
        if use_activity:
            self.activity = ActivityMetrics(n_assets)

        # Calculate extra feature dimension
        extra_features = 0
        if use_topology:
            extra_features += len(topo_thresholds) * 2
        if use_complexity:
            extra_features += n_assets
        if use_activity:
            extra_features += 4

        # Main encoder
        self.input_proj = nn.Linear(d_feat, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model + extra_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        self.point_head = nn.Linear(d_model, 1)
        self.quantile_head = QuantilePredictor(d_model, d_model, quantiles)
        self.anomaly_head = AnomalyDetector(d_model, d_model // 2)

    def forward(
        self,
        x: torch.Tensor,
        returns_window: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, d_feat) main features
            returns_window: (batch, window, n_assets) optional for topology/complexity

        Returns:
            Dict with predictions, quantiles, anomaly scores, and latent features
        """
        # Encode main features
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = h.unsqueeze(1)  # (batch, 1, d_model)
        h = self.encoder(h)
        h = h.squeeze(1)  # (batch, d_model)

        # Extract additional features
        extra = []

        if returns_window is not None:
            if self.use_topology:
                topo_feat = self.topology(returns_window)
                extra.append(topo_feat)
            if self.use_complexity:
                comp_feat = self.complexity(returns_window)
                extra.append(comp_feat)
            if self.use_activity:
                act_feat = self.activity(returns_window)
                extra.append(act_feat)

        # Fuse features
        if extra:
            extra_tensor = torch.cat(extra, dim=1)
            h = self.fusion(torch.cat([h, extra_tensor], dim=1))

        # Predictions
        point_pred = self.point_head(h)
        quantile_pred = self.quantile_head(h)
        anomaly_pred = self.anomaly_head(h)

        return {
            "prediction": point_pred,
            "quantiles": quantile_pred,
            "anomaly_score": anomaly_pred,
            "latent": h,
        }


__all__ = [
    "TopologyFeatures",
    "CompressionComplexity",
    "ActivityMetrics",
    "QuantilePredictor",
    "AnomalyDetector",
    "MarketStateNet",
]
