"""
QUANTUM CONSCIOUSNESS NETWORK TRAINING
Train quantum-enhanced network in parallel with standard and fractal networks

Author: RD-Agent Research Team
Date: 2025-11-14
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from semantic_space_data_loader import load_semantic_dataset
from quantum_consciousness_network import QuantumConsciousnessNetwork


class QuantumSemanticDataset(Dataset):
    """Dataset for quantum consciousness network training."""

    def __init__(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        correlations: np.ndarray,
        sequence_length: int = 20,
        prediction_horizon: int = 1
    ):
        """
        Args:
            returns: [n_times, n_tickers]
            volumes: [n_times, n_tickers]
            correlations: [n_times, n_corr_features]
            sequence_length: Past timesteps to use
            prediction_horizon: Steps ahead to predict
        """
        self.returns = returns
        self.volumes = volumes
        self.correlations = correlations
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        n_times, n_tickers = returns.shape

        # Normalize volumes
        self.volumes_norm = np.zeros_like(volumes)
        for i in range(volumes.shape[1]):
            mean_v = np.mean(volumes[:, i])
            std_v = np.std(volumes[:, i])
            if std_v > 0:
                self.volumes_norm[:, i] = (volumes[:, i] - mean_v) / std_v

        # Valid indices
        self.valid_indices = list(range(
            sequence_length,
            len(returns) - prediction_horizon
        ))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Get sequence
        seq_returns = self.returns[i - self.sequence_length:i, :]
        seq_volumes = self.volumes_norm[i - self.sequence_length:i, :]
        seq_correlations = self.correlations[i - self.sequence_length:i, :]

        # Get target
        target_returns = self.returns[i + self.prediction_horizon - 1, :]

        # Market state (current timestep features for consciousness field)
        market_state = np.concatenate([
            self.returns[i-1],
            self.volumes_norm[i-1],
            self.correlations[i-1]
        ])

        return (
            torch.FloatTensor(seq_returns),
            torch.FloatTensor(seq_volumes),
            torch.FloatTensor(seq_correlations),
            torch.FloatTensor(market_state),
            torch.FloatTensor(target_returns)
        )


def train_quantum_consciousness_network(
    epochs: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    seq_length: int = 20,
    device: str = 'cpu'
):
    """Train quantum consciousness network."""

    print("=" * 80)
    print("QUANTUM CONSCIOUSNESS NETWORK TRAINING")
    print("=" * 80)
    print()
    print("Training revolutionary quantum-enhanced network with:")
    print("- Quantum attention (superposition + entanglement)")
    print("- Consciousness field (coherence + presentiment)")
    print("- Retrocausal layers (future affects past)")
    print("- Remote viewing (8 non-local channels)")
    print()
    print("=" * 80)
    print()

    # Load data
    print("Loading dataset...")
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'GS', 'MS',
        'WMT', 'HD', 'MCD', 'NKE',
        'JNJ', 'UNH', 'PFE',
        'XOM', 'CVX',
        'SPY', 'QQQ', 'IWM',
    ]

    dataset_dict = load_semantic_dataset(
        tickers=TICKERS,
        interval='1d',
        days=3650  # 10 years
    )

    returns = dataset_dict['returns']
    volumes = dataset_dict['volumes']
    correlations = dataset_dict['correlations']
    n_tickers = returns.shape[1]
    n_corr = correlations.shape[1]

    print()
    print("=" * 80)
    print("QUANTUM CONSCIOUSNESS NETWORK TRAINING")
    print("=" * 80)
    print()
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Sequence length: {seq_length}")
    print()
    print(f"Device: {device}")
    print()

    # Create datasets
    print("Creating datasets...")

    # Split data into train/val
    n_total = len(returns)
    n_train = int(0.8 * n_total)

    train_returns = returns[:n_train]
    train_volumes = volumes[:n_train]
    train_correlations = correlations[:n_train]

    val_returns = returns[n_train:]
    val_volumes = volumes[n_train:]
    val_correlations = correlations[n_train:]

    train_dataset = QuantumSemanticDataset(
        train_returns,
        train_volumes,
        train_correlations,
        sequence_length=seq_length,
        prediction_horizon=1
    )

    val_dataset = QuantumSemanticDataset(
        val_returns,
        val_volumes,
        val_correlations,
        sequence_length=seq_length,
        prediction_horizon=1
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # Create network
    print("Creating Quantum Consciousness Network...")

    model = QuantumConsciousnessNetwork(
        n_tickers=n_tickers,
        embed_dim=256,
        n_quantum_heads=8,
        n_layers=4,
        field_dim=64,
        n_viewing_channels=8
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print("=" * 80)
    print("TRAINING QUANTUM CONSCIOUSNESS NETWORK")
    print("=" * 80)
    print()

    best_val_ic = -999.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            seq_returns, seq_volumes, seq_correlations, market_state, target = batch

            seq_returns = seq_returns.to(device)
            seq_volumes = seq_volumes.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            outputs = model(seq_returns, seq_volumes)

            # Main loss: return prediction
            loss = nn.MSELoss()(outputs['return_pred'], target)

            # Additional losses for quantum features (optional, small weight)
            # Encourage coherence and presentiment to be informative
            coherence_reg = 0.001 * torch.mean((outputs['coherence'] - 0.5) ** 2)
            presentiment_reg = 0.001 * torch.mean(outputs['presentiment'] ** 2)

            total_loss = loss + coherence_reg + presentiment_reg

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(outputs['return_pred'].detach().cpu().numpy())
            train_targets.append(target.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

        # Calculate train IC
        train_ic = np.mean([
            np.corrcoef(train_preds[:, i], train_targets[:, i])[0, 1]
            for i in range(n_tickers)
            if not np.isnan(np.corrcoef(train_preds[:, i], train_targets[:, i])[0, 1])
        ])

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_coherence = []
        val_presentiment = []

        with torch.no_grad():
            for batch in val_loader:
                seq_returns, seq_volumes, seq_correlations, market_state, target = batch

                seq_returns = seq_returns.to(device)
                seq_volumes = seq_volumes.to(device)
                target = target.to(device)

                outputs = model(seq_returns, seq_volumes)

                loss = nn.MSELoss()(outputs['return_pred'], target)

                val_loss += loss.item()
                val_preds.append(outputs['return_pred'].cpu().numpy())
                val_targets.append(target.cpu().numpy())
                val_coherence.append(outputs['coherence'].cpu().numpy())
                val_presentiment.append(outputs['presentiment'].cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_coherence = np.concatenate(val_coherence, axis=0)
        val_presentiment = np.concatenate(val_presentiment, axis=0)

        # Calculate val IC
        val_ic = np.mean([
            np.corrcoef(val_preds[:, i], val_targets[:, i])[0, 1]
            for i in range(n_tickers)
            if not np.isnan(np.corrcoef(val_preds[:, i], val_targets[:, i])[0, 1])
        ])

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        total_time = epoch_time * epoch / 60

        # Print progress every 10 epochs
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s, total {total_time:.1f}m)")
            print(f"  Train - Loss: {train_loss:.6f}, IC: {train_ic:+.4f}")
            print(f"  Val   - Loss: {val_loss:.6f}, IC: {val_ic:+.4f}")
            print(f"  Quantum - Coherence: {val_coherence.mean():.4f}, Presentiment: {val_presentiment.mean():+.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  Best IC: {best_val_ic:+.4f} (epoch {best_epoch})")
            print()

        # Save best model
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_epoch = epoch

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_ic': val_ic,
                'val_loss': val_loss,
                'train_ic': train_ic,
                'train_loss': train_loss,
            }

            torch.save(checkpoint, 'quantum_consciousness_network_best.pt')

    print()
    print("=" * 80)
    print("QUANTUM CONSCIOUSNESS TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Best validation IC: {best_val_ic:+.4f} (epoch {best_epoch})")
    print(f"Total training time: {total_time:.1f} minutes")
    print()
    print("Model saved: quantum_consciousness_network_best.pt")
    print()


if __name__ == "__main__":
    train_quantum_consciousness_network(
        epochs=1000,
        batch_size=64,
        learning_rate=1e-4,
        seq_length=20,
        device='cpu'
    )
