"""
HCAN + Analog: Integrated Continuous Dynamics Architecture
===========================================================

Combines:
1. Original HCAN (Hybrid Chaos-Aware Network) - Digital path
2. Analog Derivative Extractors - Continuous path
3. Cross-modal fusion - Digital ↔ Analog interaction

This represents Level 4 architecture - modeling the evolution of dynamics.

Architecture:
                    ┌─────────────────┐
                    │   Input Data    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         Digital Path                  Analog Path
              │                             │
    ┌─────────▼─────────┐       ┌──────────▼──────────┐
    │   Reservoir       │       │ Wavelet Transform   │
    │   Computing       │       │ SDE Evolution       │
    │   (HCAN)          │       │ Curvature Calc      │
    └─────────┬─────────┘       │ Order Flow Hawkes   │
              │                 └──────────┬──────────┘
              │                            │
    ┌─────────▼─────────┐       ┌──────────▼──────────┐
    │  Phase Space      │       │ Continuous Features │
    │  Reconstructor    │       │ (multi-scale)       │
    └─────────┬─────────┘       └──────────┬──────────┘
              │                            │
              └──────────────┬─────────────┘
                             │
                    ┌────────▼────────┐
                    │  Cross-Modal    │
                    │     Fusion      │
                    │  (Attention)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Transformer    │
                    │  (Phase-aware)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Multi-Task     │
                    │    Heads        │
                    └─────────────────┘

                Return | Lyap | Hurst | Bifurc | dλ/dt | dH/dt

Author: RD-Agent Research Team
Date: 2025-11-13
Level: 4 (Meta-dynamics)
Status: RESEARCH PROTOTYPE
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

# Import HCAN components
from hcan_chaos_neural_network import (
    ReservoirLayer,
    PhaseSpaceReconstructor,
    TransformerBlock,
)

# Import analog extractors
from hcan_analog_extractors import (
    WaveletDerivatives,
    LyapunovSDE,
    HurstSDE,
    LiquidityCurvature,
    OrderFlowHawkes,
    ContinuousWaveletLayer,
    LyapunovSDELayer,
)


# ============================================================================
# 1. ANALOG FEATURE AGGREGATOR
# ============================================================================

class AnalogFeatureAggregator(nn.Module):
    """
    Aggregates analog features into fixed-size representation.

    Handles variable-length continuous signals and multi-scale features.
    """

    def __init__(self,
                 n_wavelet_scales: int = 32,
                 chaos_horizon: int = 10,
                 output_dim: int = 128):
        super().__init__()

        self.n_wavelet_scales = n_wavelet_scales
        self.chaos_horizon = chaos_horizon

        # Wavelet feature processor
        self.wavelet_layer = ContinuousWaveletLayer(
            in_features=100,  # Will be adaptive
            n_scales=n_wavelet_scales
        )

        # Lyapunov evolution predictor
        self.lyap_sde_layer = LyapunovSDELayer(hidden_dim=32)

        # Microstructure encoder
        self.micro_encoder = nn.Sequential(
            nn.Linear(5, 32),  # curvature, imbalance, spread, bid_depth, ask_depth
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Order flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Linear(4, 32),  # intensity, duration, activity, acceleration
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Fusion layer
        total_analog_dim = (
            n_wavelet_scales +  # Wavelet features
            chaos_horizon +      # Lyapunov evolution
            chaos_horizon +      # Hurst evolution (placeholder)
            32 +                 # Microstructure
            32                   # Order flow
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_analog_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, analog_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            analog_dict: Dictionary with keys:
                - 'returns': [B, T] - for wavelet
                - 'current_lyapunov': [B, 1] - for SDE
                - 'current_hurst': [B, 1] - for SDE
                - 'microstructure': [B, 5] - liquidity features
                - 'order_flow': [B, 4] - flow features

        Returns:
            analog_features: [B, output_dim]
        """
        features = []

        # 1. Wavelet features
        if 'returns' in analog_dict:
            wavelet_feat = self.wavelet_layer(analog_dict['returns'])
            features.append(wavelet_feat)

        # 2. Chaos evolution
        if 'current_lyapunov' in analog_dict:
            lyap_evol = self.lyap_sde_layer(
                analog_dict['current_lyapunov'],
                horizon=self.chaos_horizon
            )
            features.append(lyap_evol)

        if 'current_hurst' in analog_dict:
            # Simplified: Use direct values, expand to horizon
            hurst_val = analog_dict['current_hurst']  # Could be [B], [B, 1], or [B, 1, 1]
            # Flatten to [B, 1]
            while len(hurst_val.shape) > 2:
                hurst_val = hurst_val.squeeze(-1)
            if len(hurst_val.shape) == 1:
                hurst_val = hurst_val.unsqueeze(1)
            # Now hurst_val is [B, 1], expand along dim 1 to get [B, horizon]
            hurst_evol = hurst_val.expand(-1, self.chaos_horizon)  # [B, horizon]
            features.append(hurst_evol)

        # 3. Microstructure
        if 'microstructure' in analog_dict:
            micro_feat = self.micro_encoder(analog_dict['microstructure'])
            features.append(micro_feat)

        # 4. Order flow
        if 'order_flow' in analog_dict:
            flow_feat = self.flow_encoder(analog_dict['order_flow'])
            features.append(flow_feat)

        # Concatenate all analog features
        combined = torch.cat(features, dim=-1)

        # Fuse into fixed representation
        analog_features = self.fusion(combined)

        return analog_features


# ============================================================================
# 2. CROSS-MODAL FUSION MODULE
# ============================================================================

class CrossModalFusion(nn.Module):
    """
    Fuses digital (discrete) and analog (continuous) features.

    Uses cross-attention to allow each modality to attend to the other.
    """

    def __init__(self, digital_dim: int, analog_dim: int, num_heads: int = 8):
        super().__init__()

        self.digital_dim = digital_dim
        self.analog_dim = analog_dim

        # Ensure dimensions are compatible
        assert digital_dim == analog_dim, "Digital and analog dims must match for fusion"

        # Digital → Analog attention
        self.digital_to_analog = nn.MultiheadAttention(
            embed_dim=digital_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Analog → Digital attention
        self.analog_to_digital = nn.MultiheadAttention(
            embed_dim=analog_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Fusion gates
        self.digital_gate = nn.Sequential(
            nn.Linear(digital_dim * 2, digital_dim),
            nn.Sigmoid()
        )

        self.analog_gate = nn.Sequential(
            nn.Linear(analog_dim * 2, analog_dim),
            nn.Sigmoid()
        )

        # Layer norms
        self.norm_digital = nn.LayerNorm(digital_dim)
        self.norm_analog = nn.LayerNorm(analog_dim)

    def forward(self,
                digital_features: torch.Tensor,
                analog_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            digital_features: [B, T, digital_dim] - from HCAN
            analog_features: [B, analog_dim] - from analog extractors

        Returns:
            fused_digital: [B, T, digital_dim]
            fused_analog: [B, analog_dim]
        """
        # Expand analog to match digital time dimension
        B, T, D = digital_features.shape
        analog_expanded = analog_features.unsqueeze(1).expand(B, T, -1)

        # Digital attends to analog
        digital_attended, _ = self.digital_to_analog(
            query=digital_features,
            key=analog_expanded,
            value=analog_expanded
        )

        # Gated fusion for digital
        gate_d = self.digital_gate(torch.cat([digital_features, digital_attended], dim=-1))
        fused_digital = gate_d * digital_features + (1 - gate_d) * digital_attended
        fused_digital = self.norm_digital(fused_digital + digital_features)  # Residual

        # Analog attends to digital (use mean pooling over time)
        digital_pooled = digital_features.mean(dim=1, keepdim=True)  # [B, 1, D]
        analog_attended, _ = self.analog_to_digital(
            query=analog_expanded.mean(dim=1, keepdim=True),
            key=digital_pooled,
            value=digital_pooled
        )
        analog_attended = analog_attended.squeeze(1)

        # Gated fusion for analog
        gate_a = self.analog_gate(torch.cat([analog_features, analog_attended], dim=-1))
        fused_analog = gate_a * analog_features + (1 - gate_a) * analog_attended
        fused_analog = self.norm_analog(fused_analog + analog_features)  # Residual

        return fused_digital, fused_analog


# ============================================================================
# 3. HCAN + ANALOG INTEGRATED MODEL
# ============================================================================

class HCANAnalog(nn.Module):
    """
    Integrated HCAN + Analog Derivatives Model.

    Combines discrete (digital) HCAN with continuous (analog) extractors.
    """

    def __init__(
        self,
        # HCAN parameters
        input_dim: int = 20,
        reservoir_size: int = 500,
        embed_dim: int = 128,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        phase_dim: int = 3,
        dropout: float = 0.1,
        # Analog parameters
        n_wavelet_scales: int = 32,
        chaos_horizon: int = 10,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # ===== DIGITAL PATH (Original HCAN) =====

        # 1. Reservoir layer
        self.reservoir = ReservoirLayer(
            input_dim=input_dim,
            reservoir_size=reservoir_size,
            spectral_radius=1.2,
            sparsity=0.1,
            leak_rate=0.3,
        )

        # 2. Embedding
        self.embedding = nn.Linear(reservoir_size, embed_dim)

        # 3. Phase space reconstructor
        self.phase_reconstructor = PhaseSpaceReconstructor(
            input_dim=embed_dim,
            phase_dim=phase_dim,
        )

        # ===== ANALOG PATH =====

        # 4. Analog feature aggregator
        self.analog_aggregator = AnalogFeatureAggregator(
            n_wavelet_scales=n_wavelet_scales,
            chaos_horizon=chaos_horizon,
            output_dim=embed_dim  # Match digital embedding
        )

        # ===== FUSION =====

        # 5. Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(
            digital_dim=embed_dim,
            analog_dim=embed_dim,
            num_heads=num_heads
        )

        # ===== TRANSFORMER =====

        # 6. Transformer blocks with phase attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        # ===== OUTPUT HEADS =====

        # 7. Multi-task prediction heads
        self.return_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.lyapunov_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.hurst_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.bifurcation_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # NEW: Analog dynamics heads
        self.lyap_derivative_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # dλ/dt can be positive or negative
        )

        self.hurst_derivative_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # dH/dt can be positive or negative
        )

    def forward(self,
                digital_features: torch.Tensor,
                analog_dict: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through integrated model.

        Args:
            digital_features: [B, T, input_dim] - traditional features
            analog_dict: Optional dictionary of analog features:
                - 'returns': [B, T_analog]
                - 'current_lyapunov': [B, 1]
                - 'current_hurst': [B, 1]
                - 'microstructure': [B, 5]
                - 'order_flow': [B, 4]

        Returns:
            Tuple of predictions:
                - return_pred: [B, 1]
                - lyapunov_pred: [B, 1]
                - hurst_pred: [B, 1]
                - bifurcation_pred: [B, 1]
                - lyap_derivative_pred: [B, 1] (NEW)
                - hurst_derivative_pred: [B, 1] (NEW)
                - phase_coords: [B, T, phase_dim]
        """
        # ===== DIGITAL PATH =====

        # 1. Reservoir dynamics
        reservoir_states = self.reservoir(digital_features)  # [B, T, reservoir_size]

        # 2. Embedding
        digital_embeddings = self.embedding(reservoir_states)  # [B, T, embed_dim]

        # 3. Phase space reconstruction
        phase_coords = self.phase_reconstructor(digital_embeddings)  # [B, T, phase_dim]

        # ===== ANALOG PATH =====

        if analog_dict is not None:
            # 4. Extract analog features
            analog_embeddings = self.analog_aggregator(analog_dict)  # [B, embed_dim]

            # 5. Cross-modal fusion
            fused_digital, fused_analog = self.cross_modal_fusion(
                digital_embeddings,
                analog_embeddings
            )
        else:
            # No analog features - use digital only
            fused_digital = digital_embeddings
            fused_analog = None

        # ===== TRANSFORMER =====

        # 6. Transformer blocks with phase attention
        x = fused_digital
        for block in self.transformer_blocks:
            x = block(x, phase_coords)

        # Pool over time for prediction
        x_pooled = x.mean(dim=1)  # [B, embed_dim]

        # If analog features exist, incorporate them
        if fused_analog is not None:
            # Combine digital and analog for final prediction
            x_pooled = x_pooled + 0.3 * fused_analog  # Weighted combination

        # ===== PREDICTIONS =====

        # 7. Multi-task heads
        pred_return = self.return_head(x_pooled)
        pred_lyapunov = self.lyapunov_head(x_pooled)
        pred_hurst = self.hurst_head(x_pooled)
        pred_bifurcation = self.bifurcation_head(x_pooled)

        # NEW: Analog dynamics predictions
        if fused_analog is not None:
            pred_lyap_derivative = self.lyap_derivative_head(fused_analog)
            pred_hurst_derivative = self.hurst_derivative_head(fused_analog)
        else:
            # Return zeros if no analog features
            batch_size = digital_features.shape[0]
            pred_lyap_derivative = torch.zeros(batch_size, 1, device=digital_features.device)
            pred_hurst_derivative = torch.zeros(batch_size, 1, device=digital_features.device)

        return (
            pred_return,
            pred_lyapunov,
            pred_hurst,
            pred_bifurcation,
            pred_lyap_derivative,
            pred_hurst_derivative,
            phase_coords
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 4. LOSS FUNCTION FOR ANALOG DYNAMICS
# ============================================================================

class AnalogChaosLoss(nn.Module):
    """
    Loss function for HCAN + Analog model.

    Extends original multi-task loss with analog dynamics.
    """

    def __init__(self,
                 return_weight: float = 1.0,
                 lyapunov_weight: float = 0.5,
                 hurst_weight: float = 0.5,
                 bifurcation_weight: float = 0.3,
                 lyap_deriv_weight: float = 0.4,  # NEW
                 hurst_deriv_weight: float = 0.4,  # NEW
                 consistency_weight: float = 0.2,
                 phase_smooth_weight: float = 0.1):
        super().__init__()

        self.return_weight = return_weight
        self.lyapunov_weight = lyapunov_weight
        self.hurst_weight = hurst_weight
        self.bifurcation_weight = bifurcation_weight
        self.lyap_deriv_weight = lyap_deriv_weight
        self.hurst_deriv_weight = hurst_deriv_weight
        self.consistency_weight = consistency_weight
        self.phase_smooth_weight = phase_smooth_weight

    def forward(self,
                predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...],
                phase_coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            predictions: (return, lyap, hurst, bifurc, dλ/dt, dH/dt)
            targets: (return_target, lyap_target, hurst_target, bifurc_target, dλ/dt_target, dH/dt_target)
            phase_coords: [B, T, phase_dim]

        Returns:
            total_loss, loss_dict
        """
        pred_return, pred_lyap, pred_hurst, pred_bifurc, pred_lyap_deriv, pred_hurst_deriv = predictions

        # Standard targets
        target_return = targets[0]
        target_lyap = targets[1] if len(targets) > 1 else None
        target_hurst = targets[2] if len(targets) > 2 else None
        target_bifurc = targets[3] if len(targets) > 3 else None

        # Analog derivative targets
        target_lyap_deriv = targets[4] if len(targets) > 4 else None
        target_hurst_deriv = targets[5] if len(targets) > 5 else None

        # 1. Return prediction loss
        loss_return = F.mse_loss(pred_return, target_return)

        # 2. Chaos metric losses
        loss_lyap = F.mse_loss(pred_lyap, target_lyap) if target_lyap is not None else 0.0
        loss_hurst = F.mse_loss(pred_hurst, target_hurst) if target_hurst is not None else 0.0
        # Clip bifurcation predictions to valid range with epsilon for numerical stability
        eps = 1e-7
        pred_bifurc_clipped = torch.clamp(pred_bifurc, eps, 1.0 - eps)
        target_bifurc_clipped = torch.clamp(target_bifurc, eps, 1.0 - eps)
        loss_bifurc = F.binary_cross_entropy(pred_bifurc_clipped, target_bifurc_clipped) if target_bifurc is not None else 0.0

        # 3. NEW: Analog derivative losses
        loss_lyap_deriv = F.mse_loss(pred_lyap_deriv, target_lyap_deriv) if target_lyap_deriv is not None else 0.0
        loss_hurst_deriv = F.mse_loss(pred_hurst_deriv, target_hurst_deriv) if target_hurst_deriv is not None else 0.0

        # 4. Consistency loss (chaos metrics should align)
        if target_lyap is not None and target_hurst is not None:
            # High Lyapunov + High Hurst → both indicate structure
            consistency = F.mse_loss(
                pred_lyap * pred_hurst,
                target_lyap * target_hurst
            )
        else:
            consistency = 0.0

        # 5. Phase space smoothness
        if phase_coords.shape[1] > 1:
            phase_diff = phase_coords[:, 1:] - phase_coords[:, :-1]
            phase_smooth = torch.mean(phase_diff ** 2)
        else:
            phase_smooth = 0.0

        # Total loss
        total_loss = (
            self.return_weight * loss_return +
            self.lyapunov_weight * loss_lyap +
            self.hurst_weight * loss_hurst +
            self.bifurcation_weight * loss_bifurc +
            self.lyap_deriv_weight * loss_lyap_deriv +
            self.hurst_deriv_weight * loss_hurst_deriv +
            self.consistency_weight * consistency +
            self.phase_smooth_weight * phase_smooth
        )

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'return': loss_return.item(),
            'lyapunov': loss_lyap.item() if isinstance(loss_lyap, torch.Tensor) else loss_lyap,
            'hurst': loss_hurst.item() if isinstance(loss_hurst, torch.Tensor) else loss_hurst,
            'bifurcation': loss_bifurc.item() if isinstance(loss_bifurc, torch.Tensor) else loss_bifurc,
            'lyap_derivative': loss_lyap_deriv.item() if isinstance(loss_lyap_deriv, torch.Tensor) else loss_lyap_deriv,
            'hurst_derivative': loss_hurst_deriv.item() if isinstance(loss_hurst_deriv, torch.Tensor) else loss_hurst_deriv,
            'consistency': consistency.item() if isinstance(consistency, torch.Tensor) else consistency,
            'phase_smooth': phase_smooth.item() if isinstance(phase_smooth, torch.Tensor) else phase_smooth,
        }

        return total_loss, loss_dict


# ============================================================================
# 5. VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HCAN + ANALOG INTEGRATED MODEL - VALIDATION")
    print("=" * 80)

    # Model parameters
    batch_size = 8
    seq_len = 20
    input_dim = 20

    # Create model
    model = HCANAnalog(
        input_dim=input_dim,
        reservoir_size=300,  # Smaller for testing
        embed_dim=128,
        num_transformer_layers=2,
        num_heads=4,
        phase_dim=3,
        n_wavelet_scales=16,
        chaos_horizon=5,
    )

    print(f"\nModel Parameters: {model.count_parameters():,}")

    # Test data
    digital_features = torch.randn(batch_size, seq_len, input_dim)

    # Analog features
    analog_dict = {
        'returns': torch.randn(batch_size, 100) * 0.01,
        'current_lyapunov': torch.rand(batch_size, 1) * 0.5,
        'current_hurst': torch.rand(batch_size, 1) * 0.4 + 0.3,
        'microstructure': torch.randn(batch_size, 5),
        'order_flow': torch.randn(batch_size, 4),
    }

    print("\n" + "-" * 80)
    print("Testing forward pass...")

    # Forward pass
    outputs = model(digital_features, analog_dict)

    pred_return, pred_lyap, pred_hurst, pred_bifurc, pred_lyap_deriv, pred_hurst_deriv, phase_coords = outputs

    print(f"\nOutput shapes:")
    print(f"  Return prediction: {pred_return.shape}")
    print(f"  Lyapunov prediction: {pred_lyap.shape}")
    print(f"  Hurst prediction: {pred_hurst.shape}")
    print(f"  Bifurcation prediction: {pred_bifurc.shape}")
    print(f"  dλ/dt prediction: {pred_lyap_deriv.shape}")
    print(f"  dH/dt prediction: {pred_hurst_deriv.shape}")
    print(f"  Phase coordinates: {phase_coords.shape}")

    print("\n" + "-" * 80)
    print("Testing loss computation...")

    # Create targets
    targets = (
        torch.randn(batch_size, 1) * 0.01,  # returns
        torch.rand(batch_size, 1) * 0.5,     # lyapunov
        torch.rand(batch_size, 1) * 0.4 + 0.3,  # hurst
        torch.rand(batch_size, 1),           # bifurcation
        torch.randn(batch_size, 1) * 0.1,    # dλ/dt
        torch.randn(batch_size, 1) * 0.1,    # dH/dt
    )

    # Loss function
    loss_fn = AnalogChaosLoss()
    total_loss, loss_dict = loss_fn(outputs[:6], targets, phase_coords)

    print(f"\nLosses:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")

    print("\n" + "-" * 80)
    print("Testing backward pass...")

    total_loss.backward()
    print("  Gradients computed successfully!")

    # Count parameters
    total_params = model.count_parameters()
    digital_params = sum(p.numel() for p in model.reservoir.parameters() if p.requires_grad)
    analog_params = sum(p.numel() for p in model.analog_aggregator.parameters() if p.requires_grad)
    fusion_params = sum(p.numel() for p in model.cross_modal_fusion.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print("PARAMETER BREAKDOWN")
    print("=" * 80)
    print(f"Total parameters:         {total_params:,}")
    print(f"  Digital (Reservoir):    {digital_params:,}")
    print(f"  Analog (Extractors):    {analog_params:,}")
    print(f"  Fusion:                 {fusion_params:,}")
    print(f"  Transformer + Heads:    {total_params - digital_params - analog_params - fusion_params:,}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("HCAN + Analog model working correctly!")
    print("=" * 80)
    print("\n🚀 LEVEL 4 ARCHITECTURE READY FOR TRAINING")
