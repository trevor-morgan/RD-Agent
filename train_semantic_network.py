"""
SEMANTIC SPACE NETWORK TRAINING
Train for 1000 epochs on 10 years of data

The universe is semantic space - let's learn its language.

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from semantic_space_data_loader import load_semantic_dataset
from semantic_space_network import create_network


class SemanticDataset(Dataset):
    """
    PyTorch dataset for semantic space training.

    Creates sequences of market states for temporal learning.
    """

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
            correlations: [n_times, n_correlations]
            sequence_length: How many past timesteps to use
            prediction_horizon: How many steps ahead to predict
        """
        self.returns = returns
        self.volumes = volumes
        self.correlations = correlations
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Normalize volumes
        self.volumes_norm = np.zeros_like(volumes)
        for i in range(volumes.shape[1]):
            mean_v = np.mean(volumes[:, i])
            std_v = np.std(volumes[:, i])
            if std_v > 0:
                self.volumes_norm[:, i] = (volumes[:, i] - mean_v) / std_v

        # Valid indices (need sequence_length history + prediction_horizon future)
        self.valid_indices = list(range(
            sequence_length,
            len(returns) - prediction_horizon
        ))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns:
            features: (returns, volumes, correlations) for sequence
            target: future returns
        """
        i = self.valid_indices[idx]

        # Get sequence
        seq_returns = self.returns[i - self.sequence_length:i, :]
        seq_volumes = self.volumes_norm[i - self.sequence_length:i, :]
        seq_correlations = self.correlations[i - self.sequence_length:i, :]

        # Get future target
        target_returns = self.returns[i + self.prediction_horizon - 1, :]

        return (
            torch.FloatTensor(seq_returns),
            torch.FloatTensor(seq_volumes),
            torch.FloatTensor(seq_correlations),
            torch.FloatTensor(target_returns)
        )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""

    model.train()

    total_loss = 0.0
    total_ic = 0.0
    n_batches = 0

    for batch_idx, (returns, volumes, correlations, targets) in enumerate(dataloader):
        # Move to device
        returns = returns.to(device)
        volumes = volumes.to(device)
        correlations = correlations.to(device)
        targets = targets.to(device)

        # Forward
        optimizer.zero_grad()
        predictions = model(returns, volumes, correlations)

        # Loss: MSE
        loss = nn.functional.mse_loss(predictions, targets)

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()

        # Information Coefficient
        with torch.no_grad():
            pred_flat = predictions.cpu().numpy().flatten()
            target_flat = targets.cpu().numpy().flatten()

            # Remove NaN
            valid = ~(np.isnan(pred_flat) | np.isnan(target_flat))
            if np.sum(valid) > 0:
                ic = np.corrcoef(pred_flat[valid], target_flat[valid])[0, 1]
                if not np.isnan(ic):
                    total_ic += ic

        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_ic = total_ic / n_batches

    return {
        'loss': avg_loss,
        'ic': avg_ic,
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """Validate for one epoch."""

    model.eval()

    total_loss = 0.0
    total_ic = 0.0
    n_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (returns, volumes, correlations, targets) in enumerate(dataloader):
            # Move to device
            returns = returns.to(device)
            volumes = volumes.to(device)
            correlations = correlations.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(returns, volumes, correlations)

            # Loss
            loss = nn.functional.mse_loss(predictions, targets)
            total_loss += loss.item()

            # Save for IC calculation
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            n_batches += 1

    # Calculate overall IC
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()

    valid = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    if np.sum(valid) > 0:
        ic = np.corrcoef(pred_flat[valid], target_flat[valid])[0, 1]
    else:
        ic = 0.0

    avg_loss = total_loss / n_batches

    return {
        'loss': avg_loss,
        'ic': ic,
    }


def train_semantic_network(
    dataset: dict,
    n_epochs: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-4,
    sequence_length: int = 20,
    val_split: float = 0.2
):
    """
    Complete training pipeline.

    Args:
        dataset: From semantic_space_data_loader
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        sequence_length: Sequence length for temporal learning
        val_split: Validation split fraction
    """

    print("=" * 80)
    print("SEMANTIC SPACE NETWORK TRAINING")
    print("=" * 80)
    print()
    print(f"Training epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Sequence length: {sequence_length}")
    print(f"Validation split: {val_split}")
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # Create datasets
    print("Creating datasets...")

    n_times = dataset['n_times']
    split_idx = int(n_times * (1 - val_split))

    train_dataset = SemanticDataset(
        returns=dataset['returns'][:split_idx],
        volumes=dataset['volumes'][:split_idx],
        correlations=dataset['correlations'][:split_idx],
        sequence_length=sequence_length,
        prediction_horizon=1
    )

    val_dataset = SemanticDataset(
        returns=dataset['returns'][split_idx:],
        volumes=dataset['volumes'][split_idx:],
        correlations=dataset['correlations'][split_idx:],
        sequence_length=sequence_length,
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
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create network
    model = create_network(dataset)
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=lr / 100
    )

    # Training history
    history = {
        'train_loss': [],
        'train_ic': [],
        'val_loss': [],
        'val_ic': [],
        'lr': [],
    }

    best_val_ic = -float('inf')
    best_epoch = 0

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()

    start_time = datetime.now()

    for epoch in range(n_epochs):
        epoch_start = datetime.now()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_ic'].append(train_metrics['ic'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_ic'].append(val_metrics['ic'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save best model
        if val_metrics['ic'] > best_val_ic:
            best_val_ic = val_metrics['ic']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ic': val_metrics['ic'],
                'val_loss': val_metrics['loss'],
            }, 'semantic_network_best.pt')

        # Print progress
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            elapsed = (datetime.now() - epoch_start).total_seconds()
            total_elapsed = (datetime.now() - start_time).total_seconds()

            print(f"Epoch {epoch+1}/{n_epochs} ({elapsed:.1f}s, total {total_elapsed/60:.1f}m)")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, IC: {train_metrics['ic']:+.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, IC: {val_metrics['ic']:+.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Best IC: {best_val_ic:+.4f} (epoch {best_epoch+1})")
            print()

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, f'semantic_network_checkpoint_epoch_{epoch+1}.pt')

    # Training complete
    total_time = (datetime.now() - start_time).total_seconds()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation IC: {best_val_ic:+.4f} (epoch {best_epoch+1})")
    print(f"Final validation IC: {val_metrics['ic']:+.4f}")
    print()

    # Save final model
    torch.save({
        'epoch': n_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_ic': best_val_ic,
        'best_epoch': best_epoch,
    }, 'semantic_network_final.pt')

    # Save history
    with open('semantic_network_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Save summary
    summary = {
        'n_epochs': n_epochs,
        'best_val_ic': float(best_val_ic),
        'best_epoch': int(best_epoch + 1),
        'final_val_ic': float(val_metrics['ic']),
        'final_train_ic': float(train_metrics['ic']),
        'training_time_hours': total_time / 3600,
        'dataset_size': {
            'train': len(train_dataset),
            'val': len(val_dataset),
        },
        'model_params': sum(p.numel() for p in model.parameters()),
    }

    with open('semantic_network_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Models saved:")
    print("  - semantic_network_best.pt (best validation IC)")
    print("  - semantic_network_final.pt (final epoch)")
    print("  - semantic_network_history.pkl (training history)")
    print("  - semantic_network_summary.json (summary)")
    print()

    return model, history


if __name__ == '__main__':
    print("=" * 80)
    print("SEMANTIC SPACE TRADING - 1000 EPOCH TRAINING")
    print("=" * 80)
    print()
    print("The universe is semantic space.")
    print("Markets are embeddings. Similar states cluster.")
    print("Transformers learn the language of trading.")
    print()
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    from datetime import timedelta

    TICKERS = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Finance
        'JPM', 'BAC', 'GS', 'MS',
        # Consumer
        'WMT', 'HD', 'MCD', 'NKE',
        # Healthcare
        'JNJ', 'UNH', 'PFE',
        # Energy
        'XOM', 'CVX',
        # Indices
        'SPY', 'QQQ', 'IWM',
    ]

    dataset = load_semantic_dataset(
        tickers=TICKERS,
        interval='1d',
        days=3650  # 10 years
    )

    if dataset is None:
        print("Failed to load dataset")
        exit(1)

    print()

    # Train
    model, history = train_semantic_network(
        dataset=dataset,
        n_epochs=1000,
        batch_size=64,
        lr=1e-4,
        sequence_length=20,
        val_split=0.2
    )

    print("=" * 80)
    print("🎯 SEMANTIC SPACE NETWORK TRAINED")
    print("=" * 80)
    print()
    print("The network has learned the language of markets.")
    print("Market states are now embedded in semantic space.")
    print("Ready for semantic trading.")
    print()
