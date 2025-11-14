"""
TRAIN FRACTAL SEMANTIC NETWORK
Train fractal-enhanced semantic space network for 1000 epochs

Runs in parallel with standard semantic network training for comparison.

Author: RD-Agent Research Team
Date: 2025-11-14
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
from fractal_semantic_space import FractalSemanticNetwork, FractalFeatureExtractor


class FractalSemanticDataset(Dataset):
    """
    Dataset with fractal features included.
    """

    def __init__(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        sequence_length: int = 20,
        prediction_horizon: int = 1
    ):
        """
        Args:
            returns: [n_times, n_tickers]
            volumes: [n_times, n_tickers]
            sequence_length: Past timesteps to use
            prediction_horizon: Steps ahead to predict
        """
        self.returns = returns
        self.volumes = volumes
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Normalize volumes
        self.volumes_norm = np.zeros_like(volumes)
        for i in range(volumes.shape[1]):
            mean_v = np.mean(volumes[:, i])
            std_v = np.std(volumes[:, i])
            if std_v > 0:
                self.volumes_norm[:, i] = (volumes[:, i] - mean_v) / std_v

        # Extract fractal features
        print("Extracting fractal features for dataset...")
        self.fractal_features = self._extract_all_fractal_features()
        print(f"✓ Fractal features shape: {self.fractal_features.shape}")

        # Valid indices
        self.valid_indices = list(range(
            sequence_length,
            len(returns) - prediction_horizon
        ))

    def _extract_all_fractal_features(self) -> np.ndarray:
        """Extract fractal features for all timesteps and tickers."""

        extractor = FractalFeatureExtractor(scales=[5, 10, 20])

        n_times, n_tickers = self.returns.shape

        # Calculate number of features per ticker
        # For each scale: hurst, fractal_dim, dfa_alpha = 3 features
        # 3 scales × 3 features = 9 features per ticker
        n_features_per_ticker = 9
        n_total_features = n_tickers * n_features_per_ticker

        fractal_features = np.zeros((n_times, n_total_features))

        # Extract for each ticker
        for ticker_idx in range(n_tickers):
            ticker_returns = self.returns[:, ticker_idx]

            # For each timestep
            for t in range(self.sequence_length, n_times):
                # Use past sequence_length returns
                past_returns = ticker_returns[max(0, t - self.sequence_length):t]

                if len(past_returns) < 10:
                    continue

                feature_idx = ticker_idx * n_features_per_ticker

                # Extract features at each scale
                for scale_idx, scale in enumerate([5, 10, 20]):
                    if len(past_returns) >= scale * 2:
                        window = past_returns[-scale:]

                        # Hurst exponent
                        h = extractor.hurst_exponent(window, max_lag=min(10, scale//2))
                        fractal_features[t, feature_idx + scale_idx * 3] = h

                        # Fractal dimension
                        d = 2.0 - h
                        fractal_features[t, feature_idx + scale_idx * 3 + 1] = d

                        # DFA alpha
                        alpha = extractor.detrended_fluctuation_analysis(window)
                        fractal_features[t, feature_idx + scale_idx * 3 + 2] = alpha

            if (ticker_idx + 1) % 5 == 0:
                print(f"  Processed {ticker_idx + 1}/{n_tickers} tickers")

        return fractal_features

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Get sequence
        seq_returns = self.returns[i - self.sequence_length:i, :]
        seq_volumes = self.volumes_norm[i - self.sequence_length:i, :]
        seq_fractal = self.fractal_features[i - self.sequence_length:i, :]

        # Get target
        target_returns = self.returns[i + self.prediction_horizon - 1, :]

        return (
            torch.FloatTensor(seq_returns),
            torch.FloatTensor(seq_volumes),
            torch.FloatTensor(seq_fractal),
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

    for batch_idx, (returns, volumes, fractal_features, targets) in enumerate(dataloader):
        # Move to device
        returns = returns.to(device)
        volumes = volumes.to(device)
        fractal_features = fractal_features.to(device)
        targets = targets.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(returns, volumes, fractal_features)
        predictions = outputs['return_pred']

        # Loss: MSE on returns
        loss = nn.functional.mse_loss(predictions, targets)

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()

        # IC
        with torch.no_grad():
            pred_flat = predictions.cpu().numpy().flatten()
            target_flat = targets.cpu().numpy().flatten()

            valid = ~(np.isnan(pred_flat) | np.isnan(target_flat))
            if np.sum(valid) > 0:
                ic = np.corrcoef(pred_flat[valid], target_flat[valid])[0, 1]
                if not np.isnan(ic):
                    total_ic += ic

        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_ic = total_ic / n_batches

    return {'loss': avg_loss, 'ic': avg_ic}


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """Validate for one epoch."""

    model.eval()

    total_loss = 0.0
    n_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (returns, volumes, fractal_features, targets) in enumerate(dataloader):
            returns = returns.to(device)
            volumes = volumes.to(device)
            fractal_features = fractal_features.to(device)
            targets = targets.to(device)

            outputs = model(returns, volumes, fractal_features)
            predictions = outputs['return_pred']

            loss = nn.functional.mse_loss(predictions, targets)
            total_loss += loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            n_batches += 1

    # Calculate IC
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

    return {'loss': avg_loss, 'ic': ic}


def train_fractal_semantic(
    dataset: dict,
    n_epochs: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-4,
    sequence_length: int = 20,
    val_split: float = 0.2
):
    """Train fractal semantic network."""

    print("=" * 80)
    print("FRACTAL SEMANTIC NETWORK TRAINING")
    print("=" * 80)
    print()
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Sequence length: {sequence_length}")
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # Create datasets
    print("Creating datasets...")

    n_times = dataset['n_times']
    split_idx = int(n_times * (1 - val_split))

    train_dataset = FractalSemanticDataset(
        returns=dataset['returns'][:split_idx],
        volumes=dataset['volumes'][:split_idx],
        sequence_length=sequence_length,
        prediction_horizon=1
    )

    val_dataset = FractalSemanticDataset(
        returns=dataset['returns'][split_idx:],
        volumes=dataset['volumes'][split_idx:],
        sequence_length=sequence_length,
        prediction_horizon=1
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()

    # Dataloaders
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
    print("Creating Fractal Semantic Network...")

    # Get fractal feature dimension from dataset
    n_fractal_features = train_dataset.fractal_features.shape[1]

    model = FractalSemanticNetwork(
        n_tickers=dataset['n_tickers'],
        n_fractal_features=n_fractal_features,
        embed_dim=256,
        n_scales=3,  # Use 3 scales for speed
        n_heads=8,
        n_layers=4,
        dropout=0.1
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler
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
    print("TRAINING FRACTAL SEMANTIC NETWORK")
    print("=" * 80)
    print()

    start_time = datetime.now()

    for epoch in range(n_epochs):
        epoch_start = datetime.now()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, device)

        # Scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_ic'].append(train_metrics['ic'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_ic'].append(val_metrics['ic'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save best
        if val_metrics['ic'] > best_val_ic:
            best_val_ic = val_metrics['ic']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ic': val_metrics['ic'],
                'val_loss': val_metrics['loss'],
            }, 'fractal_semantic_network_best.pt')

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

        # Checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, f'fractal_semantic_network_checkpoint_epoch_{epoch+1}.pt')

    # Complete
    total_time = (datetime.now() - start_time).total_seconds()

    print("=" * 80)
    print("FRACTAL TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation IC: {best_val_ic:+.4f} (epoch {best_epoch+1})")
    print()

    # Save final
    torch.save({
        'epoch': n_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_ic': best_val_ic,
        'best_epoch': best_epoch,
    }, 'fractal_semantic_network_final.pt')

    # Save history
    with open('fractal_semantic_network_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Save summary
    summary = {
        'n_epochs': n_epochs,
        'best_val_ic': float(best_val_ic),
        'best_epoch': int(best_epoch + 1),
        'final_val_ic': float(val_metrics['ic']),
        'training_time_hours': total_time / 3600,
        'model_params': total_params,
    }

    with open('fractal_semantic_network_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Models saved")
    print()

    return model, history


if __name__ == '__main__':
    print("=" * 80)
    print("FRACTAL SEMANTIC SPACE - 1000 EPOCH TRAINING")
    print("=" * 80)
    print()
    print("Training fractal-enhanced semantic network in parallel")
    print("with standard semantic network for comparison.")
    print()
    print("=" * 80)
    print()

    # Load data
    print("Loading dataset...")
    from datetime import timedelta

    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'GS', 'MS',
        'WMT', 'HD', 'MCD', 'NKE',
        'JNJ', 'UNH', 'PFE',
        'XOM', 'CVX',
        'SPY', 'QQQ', 'IWM',
    ]

    dataset = load_semantic_dataset(
        tickers=TICKERS,
        interval='1d',
        days=3650
    )

    if dataset is None:
        print("Failed to load dataset")
        exit(1)

    print()

    # Train
    model, history = train_fractal_semantic(
        dataset=dataset,
        n_epochs=1000,
        batch_size=64,
        lr=1e-4,
        sequence_length=20,
        val_split=0.2
    )

    print("=" * 80)
    print("🎯 FRACTAL SEMANTIC NETWORK TRAINED")
    print("=" * 80)
    print()
    print("Ready to compare with standard semantic network!")
