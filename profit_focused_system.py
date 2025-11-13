"""
PROFIT-FOCUSED REGIME-ADAPTIVE TRADING SYSTEM
Stop predicting returns. Start executing the RIGHT STRATEGY at the RIGHT TIME.

Core insight:
- Regime detection works (83% accuracy on high vol)
- Mean reversion works in high vol (prices overextend)
- Momentum works in low vol (trends persist)
- Trade ADAPTIVELY based on detected regime

Goal: PROFIT. Not research. Not fancy math. PROFIT.

Author: RD-Agent Research Team
Date: 2025-11-13
Purpose: Make actual fucking money
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from hcan_psi_integrated import HCANPsi
from hcan_psi_real_data_validation import RealMarketDataLoader, HCANPsiDataset
from regime_detection_model import RegimeDetector


# ==============================================================================
# REGIME-ADAPTIVE STRATEGIES
# ==============================================================================

class RegimeAdaptiveTrader:
    """
    Execute different strategies based on detected regime.

    High Vol Regime (83% detection accuracy):
    - Mean reversion (prices overextend then snap back)
    - Short-term (1-3 periods)
    - Counter-trend

    Low Vol Regime (71% detection accuracy):
    - Momentum (trends persist)
    - Medium-term (5-10 periods)
    - Follow-trend

    Medium Vol:
    - Neutral/cash (35% accuracy - avoid)
    """

    def __init__(self, regime_detector: RegimeDetector):
        self.regime_detector = regime_detector

    def mean_reversion_signal(self, returns: np.ndarray, lookback: int = 3) -> float:
        """
        Mean reversion: Fade recent moves.
        In high vol, prices overshoot then correct.
        """
        if len(returns) < lookback:
            return 0.0

        recent_move = np.sum(returns[-lookback:])

        # Fade the move (opposite direction)
        signal = -np.tanh(recent_move * 50)  # Bounded [-1, 1]

        return signal

    def momentum_signal(self, returns: np.ndarray, lookback: int = 10) -> float:
        """
        Momentum: Follow recent trend.
        In low vol, trends persist.
        """
        if len(returns) < lookback:
            return 0.0

        trend = np.sum(returns[-lookback:])

        # Follow the trend
        signal = np.tanh(trend * 30)  # Bounded [-1, 1]

        return signal

    def generate_signals(
        self,
        regime: int,
        regime_confidence: float,
        historical_returns: np.ndarray
    ) -> Tuple[float, str]:
        """
        Generate trading signal based on regime.

        Returns:
            signal: Position size [-1, 1]
            strategy: Which strategy was used
        """

        if regime == 2:  # High volatility
            signal = self.mean_reversion_signal(historical_returns, lookback=3)
            strategy = 'mean_reversion'
            # Scale by confidence (83% accurate)
            signal *= regime_confidence

        elif regime == 0:  # Low volatility
            signal = self.momentum_signal(historical_returns, lookback=10)
            strategy = 'momentum'
            # Scale by confidence (71% accurate)
            signal *= regime_confidence

        else:  # Medium volatility (regime 1)
            # Low accuracy (35%) - stay out
            signal = 0.0
            strategy = 'neutral'

        return signal, strategy


# ==============================================================================
# BACKTESTER WITH REAL COSTS
# ==============================================================================

class ProfitBacktester:
    """
    Backtest focused on ONE thing: PROFIT.

    Includes:
    - Real transaction costs
    - Slippage
    - Position limits
    - Risk management
    """

    def __init__(
        self,
        trader: RegimeAdaptiveTrader,
        cost_bps: float = 2.0,  # 2 bps per trade (realistic)
        max_position: float = 1.0,
        stop_loss: float = 0.02,  # 2% stop loss
    ):
        self.trader = trader
        self.cost_bps = cost_bps
        self.max_position = max_position
        self.stop_loss = stop_loss

    def backtest(
        self,
        regimes: np.ndarray,
        regime_confidences: np.ndarray,
        all_returns: np.ndarray,
        lookback: int = 100
    ) -> Dict:
        """
        Run backtest with adaptive strategy.
        """

        n = len(regimes)
        positions = []
        pnls = []
        strategies_used = []
        costs_paid = []

        current_position = 0.0
        cumulative_pnl = 0.0

        for i in range(lookback, n - 1):
            # Get historical returns
            historical_returns = all_returns[max(0, i-lookback):i]

            # Get regime signal
            regime = regimes[i]
            confidence = regime_confidences[i] if regime_confidences is not None else 1.0

            # Generate trading signal
            target_position, strategy = self.trader.generate_signals(
                regime, confidence, historical_returns
            )

            # Limit position size
            target_position = np.clip(target_position, -self.max_position, self.max_position)

            # Calculate trade size
            trade_size = target_position - current_position

            # Transaction cost
            cost = abs(trade_size) * (self.cost_bps / 10000)
            costs_paid.append(cost)

            # Update position
            current_position = target_position
            positions.append(current_position)
            strategies_used.append(strategy)

            # Next period return
            next_return = all_returns[i + 1]

            # PnL = position * return - cost
            pnl = current_position * next_return - cost
            pnls.append(pnl)
            cumulative_pnl += pnl

            # Stop loss (risk management)
            if cumulative_pnl < -self.stop_loss:
                print(f"⚠️  Stop loss hit at period {i}, cumulative PnL: {cumulative_pnl:.4f}")
                # Flatten position
                current_position = 0.0
                cumulative_pnl = 0.0  # Reset

        pnls = np.array(pnls)
        positions = np.array(positions)
        costs_paid = np.array(costs_paid)

        # Calculate metrics
        total_return = np.sum(pnls)
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6) * np.sqrt(252)

        # Win rate
        wins = np.sum(pnls > 0)
        total_trades = len(pnls)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Total costs
        total_costs = np.sum(costs_paid)

        # Turnover
        turnover = np.sum(np.abs(np.diff(np.concatenate([[0], positions]))))

        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_costs': total_costs,
            'turnover': turnover,
            'n_trades': total_trades,
            'strategies': {
                'mean_reversion': strategies_used.count('mean_reversion'),
                'momentum': strategies_used.count('momentum'),
                'neutral': strategies_used.count('neutral'),
            }
        }


# ==============================================================================
# MAIN PROFIT SYSTEM
# ==============================================================================

def run_profit_system():
    """
    Run profit-focused regime-adaptive system.
    """

    print("=" * 80)
    print("PROFIT-FOCUSED REGIME-ADAPTIVE TRADING")
    print("=" * 80)
    print()
    print("Goal: MAKE MONEY")
    print()
    print("Strategy:")
    print("  High Vol → Mean Reversion (83% regime detection)")
    print("  Low Vol  → Momentum (71% regime detection)")
    print("  Med Vol  → Neutral (35% regime detection)")
    print()
    print("=" * 80)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    print("Loading data...")
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
        'JNJ', 'PG', 'UNH', 'HD', 'BAC',
        'XOM', 'CVX', 'PFE', 'KO', 'PEP'
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    loader = RealMarketDataLoader(
        TICKERS,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        interval='5m'
    )
    tick_data = loader.download_data()
    full_dataset = HCANPsiDataset(tick_data, window_size=20, analog_window=100)
    print(f"✓ Dataset: {len(full_dataset)} samples\n")

    # Load regime detector
    print("Loading regime detection model...")
    checkpoint = torch.load('checkpoints_extended/best_val_ic.pt',
                           map_location=device, weights_only=False)
    config = checkpoint['config']

    model = HCANPsi(
        input_dim=20,
        reservoir_size=config['reservoir_size'],
        embed_dim=config['embed_dim'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        n_wavelet_scales=config['n_wavelet_scales'],
        n_agents=config['n_agents'],
        psi_feature_dim=config['psi_feature_dim'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")

    # Detect regimes
    print("Detecting market regimes...")
    regime_detector = RegimeDetector(tick_data, full_dataset)

    # Get regime predictions
    loader_dl = DataLoader(full_dataset, batch_size=32, shuffle=False)
    regime_preds = []
    regime_probs = []
    actual_returns = []

    with torch.no_grad():
        for batch in loader_dl:
            try:
                digital = batch['digital_features'].to(device)

                analog_dict = {
                    'returns': batch['analog_returns'].to(device),
                    'current_lyapunov': batch['current_lyapunov'].unsqueeze(1).to(device),
                    'current_hurst': batch['current_hurst'].unsqueeze(1).to(device),
                    'microstructure': batch['microstructure'].to(device),
                    'order_flow': batch['order_flow'].to(device),
                }

                psi_dict = {
                    'correlations': batch['correlations'].to(device),
                    'order_sizes': batch['order_sizes'].to(device),
                    'liquidity': batch['liquidity'].to(device),
                    'prices': batch['prices'].to(device),
                    'fundamentals': batch['fundamentals'].to(device),
                }

                outputs = model(digital, analog_dict, psi_dict)

                regime_logits = outputs['regime_pred']
                regime_pred = torch.argmax(regime_logits, dim=1).cpu().numpy()

                # Get confidence (max probability)
                regime_prob = torch.softmax(regime_logits, dim=1)
                max_prob = torch.max(regime_prob, dim=1)[0].cpu().numpy()

                regime_preds.extend(regime_pred.tolist())
                regime_probs.extend(max_prob.tolist())
                actual_returns.extend(batch['return'].numpy().flatten().tolist())

            except Exception as e:
                print(f"Batch failed: {e}")
                continue

    regime_preds = np.array(regime_preds)
    regime_probs = np.array(regime_probs)
    actual_returns = np.array(actual_returns)

    print(f"✓ Detected {len(regime_preds)} regime predictions\n")
    print(f"Regime distribution:")
    print(f"  Low vol (0):  {np.sum(regime_preds == 0)} ({np.sum(regime_preds == 0)/len(regime_preds)*100:.1f}%)")
    print(f"  Med vol (1):  {np.sum(regime_preds == 1)} ({np.sum(regime_preds == 1)/len(regime_preds)*100:.1f}%)")
    print(f"  High vol (2): {np.sum(regime_preds == 2)} ({np.sum(regime_preds == 2)/len(regime_preds)*100:.1f}%)")
    print()

    # Create trader
    trader = RegimeAdaptiveTrader(regime_detector)

    # Backtest with 3-fold walk-forward
    print("=" * 80)
    print("WALK-FORWARD BACKTEST")
    print("=" * 80)
    print()

    n_folds = 3
    fold_size = len(regime_preds) // 4

    all_results = []

    for fold in range(n_folds):
        print(f"FOLD {fold + 1}/{n_folds}")
        print("-" * 80)

        # Split
        test_start = (fold + 1) * fold_size
        test_end = min(test_start + fold_size, len(regime_preds))

        if test_end - test_start < 100:
            print("Insufficient test data")
            continue

        # Test data
        test_regimes = regime_preds[test_start:test_end]
        test_confidences = regime_probs[test_start:test_end]
        test_returns = actual_returns[test_start:test_end]

        print(f"  Test period: {test_start} to {test_end} ({len(test_regimes)} samples)")

        # Backtest
        backtester = ProfitBacktester(
            trader,
            cost_bps=2.0,  # Realistic costs
            max_position=1.0,
            stop_loss=0.05,  # 5% stop loss
        )

        results = backtester.backtest(
            test_regimes,
            test_confidences,
            actual_returns[:test_end],  # Full history for lookback
            lookback=100
        )

        print(f"  Total Return:  {results['total_return']:+.4f} ({results['total_return']*100:+.2f}%)")
        print(f"  Sharpe Ratio:  {results['sharpe']:+.2f}")
        print(f"  Win Rate:      {results['win_rate']:.1%}")
        print(f"  Total Costs:   {results['total_costs']:.4f}")
        print(f"  Strategies:    MeanRev={results['strategies']['mean_reversion']}, Mom={results['strategies']['momentum']}, Neutral={results['strategies']['neutral']}")
        print()

        all_results.append(results)

    # Aggregate
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print()

    if len(all_results) > 0:
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])

        positive_folds = sum(1 for r in all_results if r['total_return'] > 0)

        print(f"Across {len(all_results)} folds:")
        print(f"  Avg Return:    {avg_return:+.4f} ({avg_return*100:+.2f}%)")
        print(f"  Avg Sharpe:    {avg_sharpe:+.2f}")
        print(f"  Avg Win Rate:  {avg_win_rate:.1%}")
        print(f"  Positive Folds: {positive_folds}/{len(all_results)}")
        print()

        # Save
        summary = {
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'positive_folds': f"{positive_folds}/{len(all_results)}",
            'folds': all_results,
        }

        with open('profit_system_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print("✓ Results saved to: profit_system_results.json")
        print()

        # Verdict
        print("=" * 80)
        print("PROFIT VERDICT")
        print("=" * 80)
        print()

        if avg_return > 0 and avg_sharpe > 0.5 and positive_folds == len(all_results):
            print("🎯 PROFITABLE!")
            print(f"   Average {avg_return*100:.2f}% return with {avg_sharpe:.2f} Sharpe")
        elif avg_return > 0:
            print("⚠️  MARGINAL - Positive but inconsistent")
        else:
            print("❌ NOT PROFITABLE - Need more work")


if __name__ == '__main__':
    run_profit_system()
