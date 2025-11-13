"""
SEMANTIC NETWORK TRAINING MONITOR
Real-time monitoring and visualization of training progress

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import time
import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(filename: str) -> dict:
    """Parse training log to extract metrics."""

    epochs = []
    train_loss = []
    train_ic = []
    val_loss = []
    val_ic = []
    best_ic = []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for epoch line
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))

                # Next lines should have metrics
                if i + 4 < len(lines):
                    train_line = lines[i + 1]
                    val_line = lines[i + 2]
                    best_line = lines[i + 4]

                    # Parse train metrics
                    train_match = re.search(r'Loss: ([\d.]+), IC: ([+-]?[\d.]+)', train_line)
                    if train_match:
                        train_l = float(train_match.group(1))
                        train_i = float(train_match.group(2))
                    else:
                        train_l = train_i = np.nan

                    # Parse val metrics
                    val_match = re.search(r'Loss: ([\d.]+), IC: ([+-]?[\d.]+)', val_line)
                    if val_match:
                        val_l = float(val_match.group(1))
                        val_i = float(val_match.group(2))
                    else:
                        val_l = val_i = np.nan

                    # Parse best IC
                    best_match = re.search(r'Best IC: ([+-]?[\d.]+)', best_line)
                    if best_match:
                        best_i = float(best_match.group(1))
                    else:
                        best_i = np.nan

                    epochs.append(epoch_num)
                    train_loss.append(train_l)
                    train_ic.append(train_i)
                    val_loss.append(val_l)
                    val_ic.append(val_i)
                    best_ic.append(best_i)

            i += 1

    except FileNotFoundError:
        pass

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_ic': train_ic,
        'val_loss': val_loss,
        'val_ic': val_ic,
        'best_ic': best_ic,
    }


def plot_training_progress(data: dict, save_path: str = 'training_progress.png'):
    """Create visualization of training progress."""

    if len(data['epochs']) == 0:
        print("No data to plot yet")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = np.array(data['epochs'])

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, data['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, data['val_loss'], label='Val Loss', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Information Coefficient
    ax = axes[0, 1]
    ax.plot(epochs, data['train_ic'], label='Train IC', linewidth=2, alpha=0.8)
    ax.plot(epochs, data['val_ic'], label='Val IC', linewidth=2, alpha=0.8)
    ax.plot(epochs, data['best_ic'], label='Best IC', linewidth=2, alpha=0.8, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Information Coefficient')
    ax.set_title('Prediction Quality (IC)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. IC smoothed
    ax = axes[1, 0]
    if len(data['val_ic']) > 10:
        window = min(10, len(data['val_ic']) // 5)
        val_ic_smooth = np.convolve(data['val_ic'], np.ones(window)/window, mode='valid')
        epochs_smooth = epochs[:len(val_ic_smooth)]
        ax.plot(epochs_smooth, val_ic_smooth, linewidth=3, alpha=0.8, label=f'Val IC (smoothed, window={window})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Information Coefficient')
        ax.set_title('Validation IC (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Learning statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Current metrics
    current_epoch = epochs[-1]
    current_train_ic = data['train_ic'][-1]
    current_val_ic = data['val_ic'][-1]
    current_best_ic = data['best_ic'][-1]

    # Calculate improvement
    if len(data['val_ic']) > 10:
        initial_ic = np.mean(data['val_ic'][:10])
        recent_ic = np.mean(data['val_ic'][-10:])
        improvement = recent_ic - initial_ic
        improvement_text = f"""Improvement:
  Initial IC (avg first 10): {initial_ic:+.4f}
  Recent IC (avg last 10): {recent_ic:+.4f}
  Change: {improvement:+.4f}"""
    else:
        improvement_text = "Improvement:\n  (need 10+ epochs)"

    stats_text = f"""
TRAINING STATISTICS

Current Epoch: {current_epoch}

Latest Metrics:
  Train IC: {current_train_ic:+.4f}
  Val IC: {current_val_ic:+.4f}

Best Val IC: {current_best_ic:+.4f}

{improvement_text}

Progress: {current_epoch}/1000 ({current_epoch/10:.1f}%)
"""

    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved to {save_path}")


def monitor_training(
    log_file: str = 'semantic_training_log.txt',
    plot_file: str = 'training_progress.png',
    update_interval: int = 60
):
    """
    Monitor training in real-time.

    Args:
        log_file: Path to training log
        plot_file: Path to save plot
        update_interval: Seconds between updates
    """

    print("=" * 80)
    print("SEMANTIC NETWORK TRAINING MONITOR")
    print("=" * 80)
    print()
    print(f"Log file: {log_file}")
    print(f"Plot file: {plot_file}")
    print(f"Update interval: {update_interval}s")
    print()
    print("Monitoring... (Ctrl+C to stop)")
    print()

    last_epoch = 0

    try:
        while True:
            # Parse log
            data = parse_log_file(log_file)

            if len(data['epochs']) > 0:
                current_epoch = data['epochs'][-1]

                # Only update if new data
                if current_epoch > last_epoch:
                    print(f"\nEpoch {current_epoch}/1000 ({current_epoch/10:.1f}%)")
                    print(f"  Train IC: {data['train_ic'][-1]:+.4f}")
                    print(f"  Val IC: {data['val_ic'][-1]:+.4f}")
                    print(f"  Best IC: {data['best_ic'][-1]:+.4f}")

                    # Update plot
                    plot_training_progress(data, plot_file)

                    last_epoch = current_epoch

            # Wait
            time.sleep(update_interval)

    except KeyboardInterrupt:
        print()
        print("Monitoring stopped")
        print()

        # Final plot
        if len(data['epochs']) > 0:
            print("Creating final plot...")
            plot_training_progress(data, plot_file)
            print()


if __name__ == '__main__':
    import sys

    # Check if training is running
    log_file = 'semantic_training_log.txt'

    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if 'TRAINING' in content:
                print("Training log found!")
                print()

                # Parse current state
                data = parse_log_file(log_file)
                if len(data['epochs']) > 0:
                    print(f"Latest epoch: {data['epochs'][-1]}")
                    print(f"Latest val IC: {data['val_ic'][-1]:+.4f}")
                    print(f"Best IC: {data['best_ic'][-1]:+.4f}")
                    print()

                    # Create current plot
                    plot_training_progress(data, 'training_progress.png')
                    print()

                # Start monitoring
                if len(sys.argv) > 1 and sys.argv[1] == 'watch':
                    monitor_training(log_file, 'training_progress.png', update_interval=30)
                else:
                    print("One-time plot created.")
                    print("Run with 'watch' argument for continuous monitoring:")
                    print("  python monitor_training.py watch")

            else:
                print("Training not started yet")

    except FileNotFoundError:
        print("Training log not found - training not started yet")
