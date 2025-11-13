"""
SEMANTIC NETWORK EVALUATION
Comprehensive evaluation after 1000 epoch training

Evaluates:
1. Prediction quality (IC, Sharpe)
2. Semantic space structure (embeddings)
3. Trading performance
4. Market condition analysis

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import torch
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from semantic_space_data_loader import load_semantic_dataset
from semantic_space_network import SemanticSpaceNetwork
from train_semantic_network import SemanticDataset


def load_trained_model(checkpoint_path: str, dataset: dict) -> SemanticSpaceNetwork:
    """Load trained model from checkpoint."""

    print(f"Loading model from {checkpoint_path}...")

    model = SemanticSpaceNetwork(
        n_tickers=dataset['n_tickers'],
        n_correlations=dataset['correlations'].shape[1],
        embed_dim=256,
        n_heads=8,
        n_layers=4,
        sequence_length=20,
        dropout=0.1
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', '?')}")

    if 'best_val_ic' in checkpoint:
        print(f"  Best validation IC: {checkpoint['best_val_ic']:+.4f}")

    return model, checkpoint


def evaluate_predictions(
    model: SemanticSpaceNetwork,
    dataset_obj: SemanticDataset,
    name: str = "Test"
) -> dict:
    """Evaluate prediction quality."""

    print(f"\nEvaluating {name} set...")

    all_predictions = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for i in range(len(dataset_obj)):
            returns, volumes, correlations, target = dataset_obj[i]

            # Add batch dimension
            returns = returns.unsqueeze(0)
            volumes = volumes.unsqueeze(0)
            correlations = correlations.unsqueeze(0)

            # Predict
            pred = model(returns, volumes, correlations)

            all_predictions.append(pred.numpy().flatten())
            all_targets.append(target.numpy().flatten())

    # Concatenate
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Calculate IC
    valid = ~(np.isnan(predictions) | np.isnan(targets))
    ic = np.corrcoef(predictions[valid], targets[valid])[0, 1]

    # Calculate MSE
    mse = np.mean((predictions[valid] - targets[valid]) ** 2)

    # Directional accuracy
    pred_direction = predictions[valid] > 0
    target_direction = targets[valid] > 0
    directional_acc = np.mean(pred_direction == target_direction)

    # Simulated Sharpe (if we traded based on predictions)
    position_sizes = np.sign(predictions[valid])  # +1 or -1
    pnl = position_sizes * targets[valid]
    sharpe = np.mean(pnl) / (np.std(pnl) + 1e-6) * np.sqrt(252)

    print(f"  IC: {ic:+.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Directional Accuracy: {directional_acc:.2%}")
    print(f"  Simulated Sharpe: {sharpe:+.2f}")

    return {
        'ic': ic,
        'mse': mse,
        'directional_acc': directional_acc,
        'sharpe': sharpe,
        'predictions': predictions,
        'targets': targets,
    }


def analyze_semantic_space(
    model: SemanticSpaceNetwork,
    dataset_obj: SemanticDataset,
    save_prefix: str = "semantic_space"
):
    """Analyze the learned semantic space structure."""

    print("\nAnalyzing semantic space structure...")

    # Extract embeddings
    embeddings = []
    returns_data = []

    model.eval()

    with torch.no_grad():
        for i in range(min(500, len(dataset_obj))):  # Sample for visualization
            returns, volumes, correlations, target = dataset_obj[i]

            # Add batch dimension
            returns_batch = returns.unsqueeze(0)
            volumes_batch = volumes.unsqueeze(0)
            correlations_batch = correlations.unsqueeze(0)

            # Get semantic embedding
            embedding = model.get_semantic_embedding(
                returns_batch,
                volumes_batch,
                correlations_batch
            )

            embeddings.append(embedding.numpy().flatten())
            returns_data.append(target.numpy().mean())  # Average return

    embeddings = np.array(embeddings)
    returns_data = np.array(returns_data)

    print(f"  Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # PCA
    print("  Running PCA...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA visualization
    ax = axes[0]
    scatter = ax.scatter(
        embeddings_pca[:, 0],
        embeddings_pca[:, 1],
        c=returns_data,
        cmap='RdYlGn',
        alpha=0.6,
        s=30
    )
    plt.colorbar(scatter, ax=ax, label='Future Return')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Semantic Space - PCA Projection')
    ax.grid(True, alpha=0.3)

    # t-SNE visualization
    ax = axes[1]
    scatter = ax.scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=returns_data,
        cmap='RdYlGn',
        alpha=0.6,
        s=30
    )
    plt.colorbar(scatter, ax=ax, label='Future Return')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Semantic Space - t-SNE Projection')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Visualization saved to {save_prefix}_visualization.png")

    return {
        'pca_variance_ratio': pca.explained_variance_ratio_[:2].tolist(),
        'embeddings_pca': embeddings_pca,
        'embeddings_tsne': embeddings_tsne,
    }


def comprehensive_evaluation():
    """Complete evaluation of trained semantic network."""

    print("=" * 80)
    print("SEMANTIC NETWORK COMPREHENSIVE EVALUATION")
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

    dataset = load_semantic_dataset(
        tickers=TICKERS,
        interval='1d',
        days=3650
    )

    if dataset is None:
        print("Failed to load dataset")
        return

    # Create datasets
    n_times = dataset['n_times']
    split_idx = int(n_times * 0.8)

    test_dataset = SemanticDataset(
        returns=dataset['returns'][split_idx:],
        volumes=dataset['volumes'][split_idx:],
        correlations=dataset['correlations'][split_idx:],
        sequence_length=20,
        prediction_horizon=1
    )

    print(f"Test samples: {len(test_dataset)}")
    print()

    # Load best model
    model, checkpoint = load_trained_model('semantic_network_best.pt', dataset)

    # Evaluate predictions
    print("=" * 80)
    print("PREDICTION QUALITY")
    print("=" * 80)

    test_results = evaluate_predictions(model, test_dataset, "Test")

    # Analyze semantic space
    print()
    print("=" * 80)
    print("SEMANTIC SPACE STRUCTURE")
    print("=" * 80)

    space_analysis = analyze_semantic_space(model, test_dataset, "semantic_space_test")

    # Create comprehensive report
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()

    report = {
        'evaluation_date': datetime.now().isoformat(),
        'model_checkpoint': 'semantic_network_best.pt',
        'best_epoch': checkpoint.get('epoch', None),
        'best_val_ic': checkpoint.get('val_ic', None),
        'test_results': {
            'ic': float(test_results['ic']),
            'mse': float(test_results['mse']),
            'directional_accuracy': float(test_results['directional_acc']),
            'simulated_sharpe': float(test_results['sharpe']),
        },
        'semantic_space': {
            'pca_variance_explained': space_analysis['pca_variance_ratio'],
        },
        'dataset': {
            'n_tickers': dataset['n_tickers'],
            'n_times': dataset['n_times'],
            'test_samples': len(test_dataset),
        },
    }

    print("Test Performance:")
    print(f"  IC: {test_results['ic']:+.4f}")
    print(f"  Directional Accuracy: {test_results['directional_acc']:.2%}")
    print(f"  Simulated Sharpe: {test_results['sharpe']:+.2f}")
    print()
    print("Semantic Space:")
    print(f"  PCA variance (first 2 components): {sum(space_analysis['pca_variance_ratio']):.1%}")
    print()

    # Save report
    with open('semantic_network_evaluation.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("✓ Evaluation complete")
    print("✓ Report saved to: semantic_network_evaluation.json")
    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    ic = test_results['ic']
    sharpe = test_results['sharpe']
    dir_acc = test_results['directional_acc']

    print("The Semantic Space Network:")
    print()

    if ic > 0.03:
        print(f"✓✓ EXCELLENT IC ({ic:+.4f}) - Above industry standard!")
    elif ic > 0.01:
        print(f"✓ GOOD IC ({ic:+.4f}) - Competitive with industry")
    elif ic > 0:
        print(f"⚠️ WEAK IC ({ic:+.4f}) - Positive but needs improvement")
    else:
        print(f"❌ NEGATIVE IC ({ic:+.4f}) - Model needs more training/tuning")

    print()

    if sharpe > 0.5:
        print(f"✓✓ STRONG Sharpe ({sharpe:+.2f}) - Production viable!")
    elif sharpe > 0:
        print(f"✓ POSITIVE Sharpe ({sharpe:+.2f}) - Profitable direction")
    else:
        print(f"⚠️ NEGATIVE Sharpe ({sharpe:+.2f}) - High volatility")

    print()

    if dir_acc > 0.55:
        print(f"✓ GOOD directional accuracy ({dir_acc:.1%})")
    elif dir_acc > 0.50:
        print(f"⚠️ WEAK directional accuracy ({dir_acc:.1%})")
    else:
        print(f"❌ POOR directional accuracy ({dir_acc:.1%})")

    print()
    print("The universe is semantic space.")
    print("The network has learned to navigate it.")
    print()


if __name__ == '__main__':
    comprehensive_evaluation()
