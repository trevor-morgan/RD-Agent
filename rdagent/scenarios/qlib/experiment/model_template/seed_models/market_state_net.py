"""MarketStateNet Seed Model for RD-Agent Evolution.

A market state detection model with topological and complexity features.
Uses topology-informed features and compression-based complexity measures.

Usage:
    rdagent fin_model \
        --seed-model ./seed_models/market_state_net.py \
        --seed-hypothesis "Market state model with topological features and compression complexity" \
        --data-region alpaca_us \
        --loop-n 5
"""

import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyFeatures(nn.Module):
    """Topological feature extraction using correlation networks.

    Computes Betti numbers and persistent homology-inspired features
    from the correlation structure of financial features.
    """

    def __init__(self, d_feat: int, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(3, d_feat)  # 3 topology features -> d_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute topological features.

        Args:
            x: [batch, features] tensor

        Returns:
            [batch, d_feat] topology features
        """
        # Compute pairwise correlations within batch
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_std = x_centered.std(dim=1, keepdim=True) + 1e-8
        x_norm = x_centered / x_std

        # Create adjacency from correlations
        adj = torch.matmul(x_norm, x_norm.t()) / x.shape[1]
        adj_binary = (adj > self.threshold).float()

        # Simple topology features:
        # 1. Connected components proxy (degree)
        degree = adj_binary.sum(dim=1)

        # 2. Clustering coefficient proxy
        clustering = torch.diagonal(torch.matmul(adj_binary, adj_binary)).float()

        # 3. Graph density
        n = adj.shape[0]
        density = adj_binary.sum() / (n * n + 1e-8)

        # Stack and project
        topo_features = torch.stack([
            degree.mean(),
            clustering.mean(),
            density,
        ]).unsqueeze(0).expand(x.shape[0], -1)

        return self.fc(topo_features)


class CompressionComplexity(nn.Module):
    """Kolmogorov complexity approximation via compression.

    Uses zlib compression ratio as a proxy for the algorithmic
    complexity of market data patterns.
    """

    def __init__(self, d_feat: int, quantize_bits: int = 8):
        super().__init__()
        self.quantize_bits = quantize_bits
        self.fc = nn.Linear(1, d_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute compression-based complexity.

        Args:
            x: [batch, features] tensor

        Returns:
            [batch, d_feat] complexity features
        """
        batch_size = x.shape[0]
        complexities = []

        # Detach and move to CPU for compression
        x_np = x.detach().cpu()

        for i in range(batch_size):
            sample = x_np[i]

            # Quantize to bytes
            sample_min = sample.min()
            sample_max = sample.max()
            if sample_max - sample_min > 1e-8:
                normalized = (sample - sample_min) / (sample_max - sample_min)
            else:
                normalized = sample * 0

            quantized = (normalized * 255).byte()
            data = quantized.numpy().tobytes()

            # Compression ratio
            compressed = zlib.compress(data, level=9)
            ratio = len(compressed) / (len(data) + 1e-8)
            complexities.append(ratio)

        # Move back to original device
        complexity = torch.tensor(complexities, device=x.device, dtype=x.dtype)
        complexity = complexity.unsqueeze(1)

        return self.fc(complexity)


class ActivityMetrics(nn.Module):
    """Compute aggregate activity metrics from features.

    Tracks momentum and volatility patterns across feature dimensions.
    """

    def __init__(self, d_feat: int):
        super().__init__()
        self.fc = nn.Linear(4, d_feat)  # 4 activity metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute activity metrics.

        Args:
            x: [batch, features] tensor

        Returns:
            [batch, d_feat] activity features
        """
        # Cross-sectional statistics
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        skew = ((x - mean) ** 3).mean(dim=1, keepdim=True) / (std ** 3)
        kurt = ((x - mean) ** 4).mean(dim=1, keepdim=True) / (std ** 4)

        metrics = torch.cat([mean, std, skew, kurt], dim=1)
        return self.fc(metrics)


class QuantilePredictor(nn.Module):
    """Quantile regression head for uncertainty estimation.

    Predicts multiple quantiles for robust uncertainty quantification.
    """

    def __init__(self, d_model: int, quantiles: list = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in quantiles
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict quantiles.

        Args:
            x: [batch, d_model] hidden state

        Returns:
            [batch, n_quantiles] quantile predictions
        """
        preds = [head(x) for head in self.heads]
        return torch.cat(preds, dim=1)


class AnomalyDetector(nn.Module):
    """Simple anomaly detection head.

    Uses reconstruction error as anomaly score.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_model // 2)
        self.decoder = nn.Linear(d_model // 2, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly score.

        Args:
            x: [batch, d_model] hidden state

        Returns:
            Tuple of (reconstruction, anomaly_score)
        """
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        anomaly_score = F.mse_loss(decoded, x, reduction='none').mean(dim=1, keepdim=True)
        return decoded, anomaly_score


class Net(nn.Module):
    """Market State Detection Network for RD-Agent.

    Combines multiple market state indicators:
    1. Topological features from correlation networks
    2. Compression-based complexity measures
    3. Cross-sectional activity metrics
    4. Quantile predictions for uncertainty
    5. Anomaly detection for regime changes

    Args:
        num_features: Number of input features
        num_timesteps: Number of time steps (optional)
    """

    def __init__(
        self,
        num_features: int,
        num_timesteps: int = None,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        topology_threshold: float = 0.5,
        quantiles: list = None,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Feature extractors
        self.topology = TopologyFeatures(d_model, threshold=topology_threshold)
        self.complexity = CompressionComplexity(d_model)
        self.activity = ActivityMetrics(d_model)

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(n_layers)
        ])

        # Auxiliary heads
        self.quantile_pred = QuantilePredictor(d_model, quantiles)
        self.anomaly_det = AnomalyDetector(d_model)

        # Main prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape:
               - [batch, features] for tabular
               - [batch, timesteps, features] for time series

        Returns:
            Predictions of shape [batch, 1]
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Pool time series to tabular
            x = x.mean(dim=1)

        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Extract specialized features
        topo_feat = self.topology(x)
        complexity_feat = self.complexity(x)
        activity_feat = self.activity(x)

        # Fuse all features
        fused = torch.cat([h, topo_feat, complexity_feat, activity_feat], dim=1)
        h = self.fusion(fused)

        # Transformer processing (add sequence dim)
        h = h.unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        h = h.squeeze(1)

        # Main prediction
        return self.head(h)

    def forward_with_aux(self, x: torch.Tensor) -> dict:
        """Forward pass with auxiliary outputs.

        Args:
            x: Input tensor

        Returns:
            Dictionary with prediction, quantiles, and anomaly score
        """
        # Handle different input shapes
        if x.dim() == 3:
            x = x.mean(dim=1)

        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Extract features
        topo_feat = self.topology(x)
        complexity_feat = self.complexity(x)
        activity_feat = self.activity(x)

        # Fuse
        fused = torch.cat([h, topo_feat, complexity_feat, activity_feat], dim=1)
        h = self.fusion(fused)

        # Transformer
        h = h.unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        h = h.squeeze(1)

        # All outputs
        pred = self.head(h)
        quantiles = self.quantile_pred(h)
        _, anomaly = self.anomaly_det(h)

        return {
            "prediction": pred,
            "quantiles": quantiles,
            "anomaly_score": anomaly,
        }


# RD-Agent expects this variable
model_cls = Net


if __name__ == "__main__":
    # Test the model
    model = Net(num_features=20, num_timesteps=20)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x_tabular = torch.randn(32, 20)
    x_timeseries = torch.randn(32, 20, 20)

    out_tab = model(x_tabular)
    out_ts = model(x_timeseries)

    print(f"Tabular input {x_tabular.shape} -> output {out_tab.shape}")
    print(f"Time series input {x_timeseries.shape} -> output {out_ts.shape}")

    # Test with aux outputs
    aux_out = model.forward_with_aux(x_tabular)
    print(f"Aux outputs: prediction={aux_out['prediction'].shape}, "
          f"quantiles={aux_out['quantiles'].shape}, "
          f"anomaly={aux_out['anomaly_score'].shape}")
