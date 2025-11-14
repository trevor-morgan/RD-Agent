#!/usr/bin/env python3
"""
QUICK 10-EPOCH TRAINER
Returns detailed results for each epoch.
"""

import sys
sys.path.insert(0, '/home/user/RD-Agent')

from unified_model_trainer import *

def main():
    print("\n" + "="*80)
    print("QUICK 10-EPOCH TRAINING WITH DETAILED EPOCH RESULTS")
    print("="*80 + "\n")

    # Configuration for 10 epochs
    config = {
        'n_assets': 5,
        'n_days': 500,
        'window_size': 20,
        'batch_size': 64,
        'learning_rate': 0.001,  # Higher LR for faster convergence
        'weight_decay': 0.01,
        'num_epochs': 10,
        'patience': 100,  # Disable early stopping
        'd_model': 64,
        'nhead': 8,
        'num_layers': 3,
        'seed': 42
    }

    # Generate data
    data_by_asset = generate_realistic_market_data(
        n_assets=config['n_assets'],
        n_days=config['n_days'],
        seed=config['seed']
    )

    # Create datasets
    full_dataset = MarketDataset(data_by_asset, window_size=config['window_size'])
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Create model
    model = FractalTransformer(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['d_model'] * 4,
        n_assets=config['n_assets'],
        max_seq_len=config['window_size'] + 1
    )

    # Train
    trainer = ModelTrainer(model, train_loader, val_loader, config)

    print("Training for 10 epochs...\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val IC':>8} | {'Status':>15}")
    print("-" * 70)

    history = trainer.train(config['num_epochs'])

    # Detailed epoch breakdown
    print("\n" + "="*80)
    print("EPOCH-BY-EPOCH BREAKDOWN")
    print("="*80 + "\n")

    for i, (train_loss, val_loss, val_ic) in enumerate(
        zip(history['train_loss'], history['val_loss'], history['val_ic']), 1
    ):
        status = "✓ BEST" if val_ic == max(history['val_ic']) else ""
        print(f"Epoch {i:2d}:")
        print(f"  Train Loss:  {train_loss:.6f}")
        print(f"  Val Loss:    {val_loss:.6f}")
        print(f"  Val IC:      {val_ic:+.6f}  {status}")
        print()

    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nBest Epoch: {np.argmax(history['val_ic']) + 1}")
    print(f"Best Val IC: {max(history['val_ic']):+.6f}")
    print(f"Mean Val IC: {np.mean(history['val_ic']):+.6f}")
    print(f"Std Val IC:  {np.std(history['val_ic']):.6f}")
    print(f"\nFinal Val IC: {history['val_ic'][-1]:+.6f}")
    print(f"IC Improvement: {history['val_ic'][-1] - history['val_ic'][0]:+.6f}")

    # IC progression visualization
    print(f"\n{'='*80}")
    print("IC PROGRESSION (ASCII CHART)")
    print("="*80 + "\n")

    ic_values = history['val_ic']
    min_ic = min(ic_values)
    max_ic = max(ic_values)
    range_ic = max_ic - min_ic if max_ic != min_ic else 1.0

    for i, ic in enumerate(ic_values, 1):
        # Normalize to 0-50 range for visualization
        bar_length = int(((ic - min_ic) / range_ic) * 50)
        bar = '█' * bar_length
        print(f"Epoch {i:2d} ({ic:+.4f}): {bar}")

    print("\n" + "="*80)
    print(f"MODEL SAVED: best_model.pt (IC: {max(history['val_ic']):+.4f})")
    print("="*80 + "\n")

    return history


if __name__ == "__main__":
    history = main()
