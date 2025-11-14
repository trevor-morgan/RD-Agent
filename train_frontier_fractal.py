"""
TRAIN FRONTIER FRACTAL NETWORK
State-of-the-art fractal semantic network with advanced training

IMPROVEMENTS:
1. Curriculum learning (simple patterns → complex patterns)
2. Advanced fractal features (7 vs 3 baseline)
3. Multi-task learning (returns + volatility + regime)
4. Better optimization (AdamW + warmup + cosine schedule)
5. Gradient accumulation for larger effective batch size

TARGET: IC > 0.025 (25% better than baseline +0.0199)

Author: RD-Agent Research Team
Date: 2025-11-14
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from semantic_space_data_loader import load_semantic_dataset
from frontier_fractal_network import FrontierFractalNetwork, AdvancedFractalExtractor


class FrontierFractalDataset(Dataset):
    """
    Dataset with advanced fractal features.

    Features per ticker per timestep:
    - 2 basic: returns, volume
    - 7 fractal per scale × 4 scales = 28 fractal features
    Total: 30 features per ticker
    """

    def __init__(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        sequence_length: int = 20,
        prediction_horizon: int = 1,
        curriculum_mode: str = 'all'  # 'simple', 'medium', 'complex', 'all'
    ):
        self.returns = returns
        self.volumes = volumes
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.curriculum_mode = curriculum_mode

        n_times, n_tickers = returns.shape

        # Normalize volumes
        self.volumes_norm = np.zeros_like(volumes)
        for i in range(volumes.shape[1]):
            mean_v = np.mean(volumes[:, i])
            std_v = np.std(volumes[:, i])
            if std_v > 0:
                self.volumes_norm[:, i] = (volumes[:, i] - mean_v) / std_v

        # Extract advanced fractal features
        print(f"Extracting advanced fractal features (curriculum: {curriculum_mode})...")
        self.fractal_features = self._extract_all_fractal_features()
        print(f"✓ Fractal features shape: {self.fractal_features.shape}")

        # Valid indices
        self.valid_indices = list(range(
            sequence_length,
            len(returns) - prediction_horizon
        ))

        # Curriculum learning: filter by difficulty
        if curriculum_mode != 'all':
            self.valid_indices = self._filter_by_difficulty(curriculum_mode)
            print(f"  Curriculum filtered to {len(self.valid_indices)} samples ({curriculum_mode})")

    def _extract_all_fractal_features(self) -> np.ndarray:
        """Extract advanced fractal features for all timesteps and tickers."""

        extractor = AdvancedFractalExtractor(scales=[5, 10, 20, 40])

        n_times, n_tickers = self.returns.shape

        # 7 features per scale × 4 scales = 28 features per ticker
        n_features_per_ticker = 7 * 4
        n_total_features = n_tickers * n_features_per_ticker

        fractal_features = np.zeros((n_times, n_total_features))

        # Extract for each ticker
        for ticker_idx in range(n_tickers):
            ticker_returns = self.returns[:, ticker_idx]

            # For each timestep
            for t in range(self.sequence_length, n_times):
                past_returns = ticker_returns[max(0, t - self.sequence_length):t]

                if len(past_returns) < 10:
                    continue

                feature_idx = ticker_idx * n_features_per_ticker

                # Extract features at each scale
                for scale_idx, scale in enumerate([5, 10, 20, 40]):
                    if len(past_returns) >= scale * 2:
                        window = past_returns[-scale:]

                        # Extract all features
                        features = extractor.extract_all_features(window)

                        # Pack into array (7 features per scale)
                        base_idx = feature_idx + scale_idx * 7
                        fractal_features[t, base_idx + 0] = features['hurst']
                        fractal_features[t, base_idx + 1] = features['fractal_dim']
                        fractal_features[t, base_idx + 2] = features['dfa_alpha']
                        fractal_features[t, base_idx + 3] = features['mf_spectrum_width']
                        fractal_features[t, base_idx + 4] = features['holder_exp']
                        fractal_features[t, base_idx + 5] = features['regime_indicator']
                        fractal_features[t, base_idx + 6] = features['complexity']

            if (ticker_idx + 1) % 5 == 0:
                print(f"  Processed {ticker_idx + 1}/{n_tickers} tickers")

        return fractal_features

    def _filter_by_difficulty(self, mode: str) -> list:
        """
        Filter samples by difficulty for curriculum learning.

        Simple: Low volatility, high Hurst (trending)
        Medium: Moderate volatility, moderate Hurst
        Complex: High volatility, low Hurst (chaotic)
        """
        filtered_indices = []

        for idx in self.valid_indices:
            # Calculate difficulty metrics
            window_returns = self.returns[idx - self.sequence_length:idx, :]
            volatility = np.std(window_returns)

            # Get average Hurst from fractal features
            fractal_slice = self.fractal_features[idx, :]
            hurst_values = fractal_slice[0::7]  # Every 7th feature is a Hurst
            avg_hurst = np.mean(hurst_values)

            # Difficulty scoring
            if mode == 'simple':
                # Low vol, high Hurst
                if volatility < np.percentile(self.returns.std(axis=0), 33) and avg_hurst > 0.6:
                    filtered_indices.append(idx)
            elif mode == 'medium':
                # Medium vol, medium Hurst
                vol_low = np.percentile(self.returns.std(axis=0), 33)
                vol_high = np.percentile(self.returns.std(axis=0), 67)
                if vol_low <= volatility <= vol_high and 0.4 <= avg_hurst <= 0.6:
                    filtered_indices.append(idx)
            elif mode == 'complex':
                # High vol, low Hurst
                if volatility > np.percentile(self.returns.std(axis=0), 67) and avg_hurst < 0.4:
                    filtered_indices.append(idx)

        return filtered_indices if len(filtered_indices) > 100 else self.valid_indices

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
        target_volatility = np.abs(target_returns)

        # Regime target (0=mean-revert, 1=neutral, 2=trending)
        # Based on average Hurst at prediction time
        future_fractal = self.fractal_features[i, :]
        future_hurst = future_fractal[0::7]
        avg_hurst = np.mean(future_hurst)

        if avg_hurst > 0.6:
            regime_target = 2  # Trending
        elif avg_hurst < 0.4:
            regime_target = 0  # Mean-reverting
        else:
            regime_target = 1  # Neutral

        return (
            torch.FloatTensor(seq_returns),
            torch.FloatTensor(seq_volumes),
            torch.FloatTensor(seq_fractal),
            torch.FloatTensor(target_returns),
            torch.FloatTensor(target_volatility),
            torch.LongTensor([regime_target])
        )


def train_frontier_fractal(
    epochs: int = 1000,
    batch_size: int = 32,  # Smaller batch for gradient accumulation
    learning_rate: float = 2e-4,
    seq_length: int = 20,
    device: str = 'cpu',
    curriculum: bool = True,  # Enable curriculum learning
    gradient_accumulation_steps: int = 2  # Effective batch = 32 * 2 = 64
):
    """Train frontier fractal network with advanced techniques."""

    print("=" * 80)
    print("FRONTIER FRACTAL NETWORK TRAINING")
    print("=" * 80)
    print()
    print("ADVANCED FEATURES:")
    print("- 7 fractal features per scale (vs 3 baseline)")
    print("- 4 scales (5, 10, 20, 40 periods)")
    print("- Multi-task learning (returns + volatility + regime)")
    print("- Curriculum learning" if curriculum else "- Standard training")
    print("- Gradient accumulation (effective batch: {})".format(batch_size * gradient_accumulation_steps))
    print()
    print("TARGET: IC > 0.025 (25% improvement over baseline +0.0199)")
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
    n_tickers = returns.shape[1]

    print()
    print("=" * 80)
    print("FRONTIER FRACTAL NETWORK TRAINING")
    print("=" * 80)
    print()
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"Learning rate: {learning_rate}")
    print(f"Sequence length: {seq_length}")
    print(f"Device: {device}")
    print()

    # Split data
    n_total = len(returns)
    n_train = int(0.8 * n_total)

    train_returns = returns[:n_train]
    train_volumes = volumes[:n_train]

    val_returns = returns[n_train:]
    val_volumes = volumes[n_train:]

    # Create model
    print("Creating Frontier Fractal Network...")
    model = FrontierFractalNetwork(
        n_tickers=n_tickers,
        embed_dim=384,
        n_heads=12,
        n_layers=6,
        dropout=0.15
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Optimizer with weight decay (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate schedule with warmup
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    # Training loop
    print("=" * 80)
    print("TRAINING FRONTIER FRACTAL NETWORK")
    print("=" * 80)
    print()

    best_val_ic = -999.0
    best_epoch = 0

    # Curriculum learning stages
    if curriculum:
        stages = [
            ('simple', epochs // 3),
            ('medium', epochs // 3),
            ('all', epochs // 3)
        ]
        print("Curriculum stages:")
        for stage, stage_epochs in stages:
            print(f"  {stage}: {stage_epochs} epochs")
        print()
    else:
        stages = [('all', epochs)]

    global_epoch = 0

    for curriculum_mode, stage_epochs in stages:
        if curriculum:
            print(f"\n{'='*80}")
            print(f"CURRICULUM STAGE: {curriculum_mode.upper()}")
            print(f"{'='*80}\n")

        # Create datasets for this curriculum stage
        train_dataset = FrontierFractalDataset(
            train_returns,
            train_volumes,
            sequence_length=seq_length,
            prediction_horizon=1,
            curriculum_mode=curriculum_mode
        )

        val_dataset = FrontierFractalDataset(
            val_returns,
            val_volumes,
            sequence_length=seq_length,
            prediction_horizon=1,
            curriculum_mode='all'  # Always validate on all data
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print()

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

        # Train this curriculum stage
        for epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            epoch_start = time.time()

            # Train
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                seq_returns, seq_volumes, seq_fractal, target_ret, target_vol, target_regime = batch

                seq_returns = seq_returns.to(device)
                seq_volumes = seq_volumes.to(device)
                seq_fractal = seq_fractal.to(device)
                target_ret = target_ret.to(device)
                target_vol = target_vol.to(device)
                target_regime = target_regime.squeeze().to(device)

                outputs = model(seq_returns, seq_volumes, seq_fractal)

                # Multi-task loss
                loss_ret = mse_loss(outputs['return_pred'], target_ret)
                loss_vol = mse_loss(outputs['volatility_pred'], target_vol)
                loss_regime = ce_loss(outputs['regime_pred'], target_regime)

                # Weighted combination
                loss = loss_ret + 0.3 * loss_vol + 0.2 * loss_regime

                # Gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss_ret.item()
                train_preds.append(outputs['return_pred'].detach().cpu().numpy())
                train_targets.append(target_ret.detach().cpu().numpy())

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

            with torch.no_grad():
                for batch in val_loader:
                    seq_returns, seq_volumes, seq_fractal, target_ret, target_vol, target_regime = batch

                    seq_returns = seq_returns.to(device)
                    seq_volumes = seq_volumes.to(device)
                    seq_fractal = seq_fractal.to(device)
                    target_ret = target_ret.to(device)

                    outputs = model(seq_returns, seq_volumes, seq_fractal)

                    loss = mse_loss(outputs['return_pred'], target_ret)

                    val_loss += loss.item()
                    val_preds.append(outputs['return_pred'].cpu().numpy())
                    val_targets.append(target_ret.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)

            # Calculate val IC
            val_ic = np.mean([
                np.corrcoef(val_preds[:, i], val_targets[:, i])[0, 1]
                for i in range(n_tickers)
                if not np.isnan(np.corrcoef(val_preds[:, i], val_targets[:, i])[0, 1])
            ])

            # Update scheduler
            scheduler.step()

            epoch_time = time.time() - epoch_start
            total_time = epoch_time * global_epoch / 60

            # Print progress every 10 epochs
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {global_epoch}/{sum(s[1] for s in stages)} ({epoch_time:.1f}s, total {total_time:.1f}m)")
                print(f"  Stage: {curriculum_mode}")
                print(f"  Train - Loss: {train_loss:.6f}, IC: {train_ic:+.4f}")
                print(f"  Val   - Loss: {val_loss:.6f}, IC: {val_ic:+.4f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
                print(f"  Best IC: {best_val_ic:+.4f} (epoch {best_epoch})")
                print()

            # Save best model
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_epoch = global_epoch

                checkpoint = {
                    'epoch': global_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_ic': val_ic,
                    'val_loss': val_loss,
                    'train_ic': train_ic,
                    'train_loss': train_loss,
                    'curriculum_mode': curriculum_mode
                }

                torch.save(checkpoint, 'frontier_fractal_network_best.pt')

    print()
    print("=" * 80)
    print("FRONTIER FRACTAL TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Best validation IC: {best_val_ic:+.4f} (epoch {best_epoch})")
    print(f"Total training time: {total_time:.1f} minutes")
    print()
    print("Model saved: frontier_fractal_network_best.pt")
    print()

    if best_val_ic > 0.025:
        print("🎯 TARGET ACHIEVED! IC > 0.025")
    elif best_val_ic > 0.0199:
        print("✅ IMPROVEMENT! IC better than baseline (+0.0199)")
    else:
        print("⚠️  Did not beat baseline. May need more training or tuning.")
    print()


if __name__ == "__main__":
    train_frontier_fractal(
        epochs=300,  # 300 epochs with curriculum (100 each stage)
        batch_size=32,
        learning_rate=2e-4,
        seq_length=20,
        device='cpu',
        curriculum=True,
        gradient_accumulation_steps=2
    )
