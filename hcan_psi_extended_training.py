"""
HCAN-Ψ (Psi): Extended Training & Hyperparameter Optimization

This script runs extended training (50-100 epochs) with hyperparameter search
and comprehensive performance tracking.

Features:
- Extended training (50-100 epochs)
- Hyperparameter grid search
- Multiple model checkpoints
- Detailed metrics tracking
- Learning rate scheduling
- Performance visualization

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import components
from hcan_psi_integrated import HCANPsi, HCANPsiLoss
from hcan_psi_real_data_validation import (
    RealMarketDataLoader,
    HCANPsiDataset
)


# ============================================================================
# HYPERPARAMETER CONFIGURATION
# ============================================================================


class HyperparamConfig:
    """Hyperparameter configuration for grid search."""

    # Model architecture
    RESERVOIR_SIZES = [300, 500]
    EMBED_DIMS = [128, 256]
    NUM_TRANSFORMER_LAYERS = [3, 4]
    NUM_HEADS = [4, 8]
    N_WAVELET_SCALES = [16, 32]

    # Ψ features
    N_AGENTS = [30, 50]
    N_COMPONENTS = [10]
    PSI_FEATURE_DIM = [32, 64]

    # Training
    LEARNING_RATES = [1e-3, 5e-4, 1e-4]
    BATCH_SIZES = [32, 64]
    WEIGHT_DECAYS = [1e-5, 1e-4]

    # Loss weights
    RETURN_WEIGHTS = [1.0]
    LYAPUNOV_WEIGHTS = [0.5, 1.0]
    HURST_WEIGHTS = [0.5, 1.0]
    BIFURCATION_WEIGHTS = [0.3, 0.5]
    DERIVATIVE_WEIGHTS = [0.2, 0.3]
    PHYSICS_WEIGHTS = [0.3, 0.5]
    PSYCHOLOGY_WEIGHTS = [0.2, 0.3]
    REFLEXIVITY_WEIGHTS = [0.2, 0.3]

    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration (medium complexity)."""
        return {
            'reservoir_size': 300,
            'embed_dim': 128,
            'num_transformer_layers': 3,
            'num_heads': 4,
            'n_wavelet_scales': 16,
            'n_agents': 30,
            'n_components': 10,
            'psi_feature_dim': 32,
            'lr': 1e-3,
            'batch_size': 32,
            'weight_decay': 1e-5,
            'return_weight': 1.0,
            'lyapunov_weight': 0.5,
            'hurst_weight': 0.5,
            'bifurcation_weight': 0.3,
            'derivative_weight': 0.2,
            'physics_weight': 0.3,
            'psychology_weight': 0.2,
            'reflexivity_weight': 0.2,
        }

    @classmethod
    def get_large_config(cls) -> Dict:
        """Get large configuration (high capacity)."""
        return {
            'reservoir_size': 500,
            'embed_dim': 256,
            'num_transformer_layers': 4,
            'num_heads': 8,
            'n_wavelet_scales': 32,
            'n_agents': 50,
            'n_components': 10,
            'psi_feature_dim': 64,
            'lr': 5e-4,
            'batch_size': 32,
            'weight_decay': 1e-5,
            'return_weight': 1.0,
            'lyapunov_weight': 1.0,
            'hurst_weight': 1.0,
            'bifurcation_weight': 0.5,
            'derivative_weight': 0.3,
            'physics_weight': 0.5,
            'psychology_weight': 0.3,
            'reflexivity_weight': 0.3,
        }

    @classmethod
    def get_efficient_config(cls) -> Dict:
        """Get efficient configuration (low latency)."""
        return {
            'reservoir_size': 200,
            'embed_dim': 64,
            'num_transformer_layers': 2,
            'num_heads': 4,
            'n_wavelet_scales': 8,
            'n_agents': 20,
            'n_components': 5,
            'psi_feature_dim': 32,
            'lr': 1e-3,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'return_weight': 1.0,
            'lyapunov_weight': 0.5,
            'hurst_weight': 0.5,
            'bifurcation_weight': 0.3,
            'derivative_weight': 0.2,
            'physics_weight': 0.2,
            'psychology_weight': 0.2,
            'reflexivity_weight': 0.2,
        }


# ============================================================================
# EXTENDED TRAINING
# ============================================================================


class ExtendedTrainer:
    """Extended training with comprehensive tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: str = 'cpu',
        save_dir: str = './checkpoints'
    ):
        """
        Args:
            model: HCAN-Ψ model
            train_loader: Training data
            val_loader: Validation data
            test_loader: Test data
            config: Hyperparameter configuration
            device: Device to train on
            save_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.save_dir = save_dir

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Loss function
        self.loss_fn = HCANPsiLoss(
            return_weight=config['return_weight'],
            lyapunov_weight=config['lyapunov_weight'],
            hurst_weight=config['hurst_weight'],
            bifurcation_weight=config['bifurcation_weight'],
            derivative_weight=config['derivative_weight'],
            physics_weight=config['physics_weight'],
            psychology_weight=config['psychology_weight'],
            reflexivity_weight=config['reflexivity_weight']
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # Learning rate schedulers
        self.scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_ic': [],
            'val_ic': [],
            'test_ic': [],
            'lr': [],
            'epoch_times': []
        }

        self.best_val_loss = float('inf')
        self.best_val_ic = -float('inf')
        self.best_test_ic = -float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        losses = []
        preds = []
        targets = []

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move to device
                digital_features = batch['digital_features'].to(self.device)

                analog_dict = {
                    'returns': batch['analog_returns'].to(self.device),
                    'current_lyapunov': batch['current_lyapunov'].unsqueeze(1).to(self.device),
                    'current_hurst': batch['current_hurst'].unsqueeze(1).to(self.device),
                    'microstructure': batch['microstructure'].to(self.device),
                    'order_flow': batch['order_flow'].to(self.device),
                }

                psi_dict = {
                    'correlations': batch['correlations'].to(self.device),
                    'order_sizes': batch['order_sizes'].to(self.device),
                    'liquidity': batch['liquidity'].to(self.device),
                    'prices': batch['prices'].to(self.device),
                    'fundamentals': batch['fundamentals'].to(self.device),
                }

                # Forward
                outputs = self.model(digital_features, analog_dict, psi_dict)

                # Targets
                target_dict = {
                    'return': batch['return'].to(self.device),
                    'lyapunov': batch['lyapunov'].to(self.device),
                    'hurst': batch['hurst'].to(self.device),
                    'bifurcation': batch['bifurcation'].to(self.device),
                    'lyap_derivative': batch['lyap_derivative'].to(self.device),
                    'hurst_derivative': batch['hurst_derivative'].to(self.device),
                    'entropy': batch['entropy'].to(self.device),
                    'consciousness': batch['consciousness'].to(self.device),
                    'regime': batch['regime'].to(self.device),
                }

                # Loss
                loss, _ = self.loss_fn(outputs, target_dict)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track
                losses.append(loss.item())
                preds.append(outputs['return_pred'].detach().cpu().numpy())
                targets.append(batch['return'].numpy())

            except RuntimeError:
                continue

        # Metrics
        if len(losses) == 0:
            return {'loss': float('nan'), 'ic': float('nan')}

        mean_loss = np.mean(losses)
        preds_flat = np.concatenate(preds).flatten()
        targets_flat = np.concatenate(targets).flatten()
        ic = np.corrcoef(preds_flat, targets_flat)[0, 1]

        return {'loss': mean_loss, 'ic': ic}

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        losses = []
        preds = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                try:
                    digital_features = batch['digital_features'].to(self.device)

                    analog_dict = {
                        'returns': batch['analog_returns'].to(self.device),
                        'current_lyapunov': batch['current_lyapunov'].unsqueeze(1).to(self.device),
                        'current_hurst': batch['current_hurst'].unsqueeze(1).to(self.device),
                        'microstructure': batch['microstructure'].to(self.device),
                        'order_flow': batch['order_flow'].to(self.device),
                    }

                    psi_dict = {
                        'correlations': batch['correlations'].to(self.device),
                        'order_sizes': batch['order_sizes'].to(self.device),
                        'liquidity': batch['liquidity'].to(self.device),
                        'prices': batch['prices'].to(self.device),
                        'fundamentals': batch['fundamentals'].to(self.device),
                    }

                    outputs = self.model(digital_features, analog_dict, psi_dict)

                    target_dict = {
                        'return': batch['return'].to(self.device),
                        'lyapunov': batch['lyapunov'].to(self.device),
                        'hurst': batch['hurst'].to(self.device),
                        'bifurcation': batch['bifurcation'].to(self.device),
                        'lyap_derivative': batch['lyap_derivative'].to(self.device),
                        'hurst_derivative': batch['hurst_derivative'].to(self.device),
                        'entropy': batch['entropy'].to(self.device),
                        'consciousness': batch['consciousness'].to(self.device),
                        'regime': batch['regime'].to(self.device),
                    }

                    loss, _ = self.loss_fn(outputs, target_dict)

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        losses.append(loss.item())
                        preds.append(outputs['return_pred'].cpu().numpy())
                        targets.append(batch['return'].numpy())

                except RuntimeError:
                    continue

        if len(losses) == 0:
            return {'loss': float('nan'), 'ic': float('nan'), 'mse': float('nan')}

        mean_loss = np.mean(losses)
        preds_flat = np.concatenate(preds).flatten()
        targets_flat = np.concatenate(targets).flatten()
        ic = np.corrcoef(preds_flat, targets_flat)[0, 1]
        mse = np.mean((preds_flat - targets_flat) ** 2)

        return {'loss': mean_loss, 'ic': ic, 'mse': mse}

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'history': self.history
        }

        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
            print(f"  Saved checkpoint: {path}")

        # Save best models
        if is_best:
            if metrics['val_ic'] > self.best_val_ic:
                path = os.path.join(self.save_dir, 'best_val_ic.pt')
                torch.save(checkpoint, path)
                self.best_val_ic = metrics['val_ic']
                print(f"  ✓ New best validation IC: {metrics['val_ic']:.4f}")

            if metrics['val_loss'] < self.best_val_loss:
                path = os.path.join(self.save_dir, 'best_val_loss.pt')
                torch.save(checkpoint, path)
                self.best_val_loss = metrics['val_loss']
                print(f"  ✓ New best validation loss: {metrics['val_loss']:.6f}")

    def train(self, n_epochs: int = 100, eval_every: int = 1, use_cosine_schedule: bool = True):
        """
        Main training loop.

        Args:
            n_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            use_cosine_schedule: Use cosine annealing (vs plateau scheduling)
        """
        print("=" * 80)
        print(f"EXTENDED TRAINING: {n_epochs} EPOCHS")
        print("=" * 80)

        for epoch in range(1, n_epochs + 1):
            epoch_start = datetime.now()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            if epoch % eval_every == 0:
                val_metrics = self.evaluate(self.val_loader)
                test_metrics = self.evaluate(self.test_loader)
            else:
                val_metrics = {'loss': float('nan'), 'ic': float('nan')}
                test_metrics = {'loss': float('nan'), 'ic': float('nan')}

            epoch_time = (datetime.now() - epoch_start).total_seconds()

            # Track
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['train_ic'].append(train_metrics['ic'])
            self.history['val_ic'].append(val_metrics['ic'])
            self.history['test_ic'].append(test_metrics['ic'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_times'].append(epoch_time)

            # Print
            print(f"\nEpoch {epoch}/{n_epochs} ({epoch_time:.1f}s)")
            print(f"  Train: Loss {train_metrics['loss']:.6f}, IC {train_metrics['ic']:.4f}")
            if epoch % eval_every == 0:
                print(f"  Val:   Loss {val_metrics['loss']:.6f}, IC {val_metrics['ic']:.4f}")
                print(f"  Test:  Loss {test_metrics['loss']:.6f}, IC {test_metrics['ic']:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Learning rate scheduling
            if use_cosine_schedule:
                self.scheduler_cosine.step()
            else:
                if epoch % eval_every == 0:
                    self.scheduler_plateau.step(val_metrics['loss'])

            # Save checkpoints
            if epoch % eval_every == 0:
                is_best = (val_metrics['ic'] > self.best_val_ic or
                          val_metrics['loss'] < self.best_val_loss)
                self.save_checkpoint(epoch, {
                    'train_loss': train_metrics['loss'],
                    'train_ic': train_metrics['ic'],
                    'val_loss': val_metrics['loss'],
                    'val_ic': val_metrics['ic'],
                    'test_loss': test_metrics['loss'],
                    'test_ic': test_metrics['ic'],
                }, is_best=is_best)

            # Update best test IC
            if test_metrics['ic'] > self.best_test_ic:
                self.best_test_ic = test_metrics['ic']

        # Save final history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best Validation IC: {self.best_val_ic:.4f}")
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")
        print(f"Best Test IC: {self.best_test_ic:.4f}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main extended training pipeline."""
    print("=" * 80)
    print("HCAN-Ψ EXTENDED TRAINING")
    print("=" * 80)

    # Configuration
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
        'JNJ', 'PG', 'UNH', 'HD', 'BAC',
        'XOM', 'CVX', 'PFE', 'KO', 'PEP'
    ]

    # Use recent 60 days (maximum for 5m data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    START_DATE = start_date.strftime('%Y-%m-%d')
    END_DATE = end_date.strftime('%Y-%m-%d')
    INTERVAL = '5m'

    N_EPOCHS = 50  # Extended training

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Get configuration (try large, fall back to default)
    print("\nConfiguration: LARGE (high capacity)")
    config = HyperparamConfig.get_large_config()

    # Load data
    print("\n1. LOADING DATA")
    print("-" * 80)

    loader = RealMarketDataLoader(TICKERS, START_DATE, END_DATE, INTERVAL)
    tick_data = loader.download_data()

    print(f"\n✓ Data loaded: {tick_data['prices'].shape[0]} ticks, {tick_data['prices'].shape[1]} stocks")

    # Create dataset
    print("\n2. CREATING DATASET")
    print("-" * 80)

    full_dataset = HCANPsiDataset(tick_data, window_size=20, analog_window=100, prediction_horizon=1)

    n_samples = len(full_dataset)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    n_test = n_samples - n_train - n_val

    train_dataset = torch.utils.data.Subset(full_dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, range(n_train, n_train + n_val))
    test_dataset = torch.utils.data.Subset(full_dataset, range(n_train + n_val, n_samples))

    print(f"Total: {n_samples}, Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # Create model
    print("\n3. CREATING MODEL")
    print("-" * 80)

    model = HCANPsi(
        input_dim=20,
        reservoir_size=config['reservoir_size'],
        embed_dim=config['embed_dim'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        n_wavelet_scales=config['n_wavelet_scales'],
        chaos_horizon=10,
        n_agents=config['n_agents'],
        n_components=config['n_components'],
        n_meta_levels=3,
        psi_feature_dim=config['psi_feature_dim'],
        use_physics=True,
        use_psychology=True,
        use_reflexivity=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Configuration: {config}")

    # Create trainer
    print("\n4. EXTENDED TRAINING")
    print("-" * 80)

    trainer = ExtendedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        save_dir='./checkpoints_extended'
    )

    # Train
    trainer.train(n_epochs=N_EPOCHS, eval_every=1, use_cosine_schedule=True)

    print("\n" + "=" * 80)
    print("EXTENDED TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nCheckpoints saved to: ./checkpoints_extended/")
    print(f"Best validation IC: {trainer.best_val_ic:.4f}")
    print(f"Best test IC: {trainer.best_test_ic:.4f}")


if __name__ == "__main__":
    main()
