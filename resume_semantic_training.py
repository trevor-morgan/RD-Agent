"""
RESUME SEMANTIC NETWORK TRAINING
Continue from checkpoint to complete 1000 epochs

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
from train_semantic_network import SemanticDataset, train_epoch, validate_epoch


def resume_training(
    checkpoint_path: str,
    dataset: dict,
    n_epochs: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-4,
    sequence_length: int = 20,
    val_split: float = 0.2
):
    """Resume training from checkpoint."""

    print("=" * 80)
    print("RESUMING SEMANTIC NETWORK TRAINING")
    print("=" * 80)
    print()

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    start_epoch = checkpoint.get('epoch', 0) + 1

    print(f"  Resuming from epoch {start_epoch}")
    if 'val_ic' in checkpoint:
        print(f"  Previous best IC: {checkpoint.get('val_ic', 0):+.4f}")
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

    # Create network and load weights
    model = create_network(dataset)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=lr / 100,
        last_epoch=start_epoch - 1
    )

    # Load history if available
    if 'history' in checkpoint:
        history = checkpoint['history']
    else:
        history = {
            'train_loss': [],
            'train_ic': [],
            'val_loss': [],
            'val_ic': [],
            'lr': [],
        }

    # Track best
    best_val_ic = checkpoint.get('val_ic', -float('inf'))
    best_epoch = checkpoint.get('epoch', 0)

    print("=" * 80)
    print("CONTINUING TRAINING")
    print("=" * 80)
    print()

    start_time = datetime.now()

    for epoch in range(start_epoch, n_epochs):
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
        'resumed_from_epoch': int(start_epoch),
    }

    with open('semantic_network_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Models saved")
    print()

    return model, history


if __name__ == '__main__':
    print("=" * 80)
    print("SEMANTIC SPACE TRADING - RESUME TO 1000 EPOCHS")
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

    # Resume training
    model, history = resume_training(
        checkpoint_path='semantic_network_checkpoint_epoch_100.pt',
        dataset=dataset,
        n_epochs=1000,
        batch_size=64,
        lr=1e-4,
        sequence_length=20,
        val_split=0.2
    )

    print("=" * 80)
    print("🎯 1000 EPOCH TRAINING COMPLETE")
    print("=" * 80)
