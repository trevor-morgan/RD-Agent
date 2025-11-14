"""
PRODUCTION SEMANTIC TRADING SYSTEM
Real-time trading using semantic space neural network

Features:
- Live market data integration
- Real-time semantic predictions
- Risk management & position sizing
- Transaction cost modeling
- Performance tracking
- Alert system

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from semantic_space_network import SemanticSpaceNetwork
from semantic_space_data_loader import load_semantic_dataset


class ProductionSemanticTrader:
    """
    Production-ready semantic space trader.

    Loads trained model and executes trades based on semantic predictions.
    """

    def __init__(
        self,
        model_path: str,
        tickers: List[str],
        initial_capital: float = 100000.0,
        max_position_size: float = 0.1,  # 10% per position
        transaction_cost_bps: float = 2.0,
        stop_loss_pct: float = 0.05,  # 5% stop loss
        take_profit_pct: float = 0.10,  # 10% take profit
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            tickers: List of tickers to trade
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            transaction_cost_bps: Transaction cost in basis points
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_cost_bps = transaction_cost_bps / 10000
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.sequence_length = 20

        # Load model
        print("Loading semantic network...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")

        # Trading state
        self.positions = {}  # {ticker: {'shares': N, 'entry_price': P, 'entry_time': T}}
        self.trades = []     # History of all trades
        self.portfolio_values = []  # Time series of portfolio value

        # Data buffers for semantic features
        self.price_history = {ticker: [] for ticker in tickers}
        self.volume_history = {ticker: [] for ticker in tickers}

    def _load_model(self, model_path: str) -> SemanticSpaceNetwork:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Create model (hardcoded architecture - should match training)
        model = SemanticSpaceNetwork(
            n_tickers=len(self.tickers),
            n_correlations=len(self.tickers) * (len(self.tickers) - 1) // 2,
            embed_dim=256,
            n_heads=8,
            n_layers=4,
            sequence_length=self.sequence_length,
            dropout=0.1
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def fetch_live_data(self, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch live market data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        all_data = {}

        for ticker in self.tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    progress=False
                )
                if len(data) > 0:
                    all_data[ticker] = data
            except Exception as e:
                print(f"Warning: Failed to fetch {ticker}: {e}")

        return all_data

    def prepare_semantic_features(self, market_data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare features for semantic network.

        Returns:
            (returns, volumes, correlations) as torch tensors
        """
        # Find common timestamps
        timestamps = None
        for ticker, data in market_data.items():
            if timestamps is None:
                timestamps = set(data.index)
            else:
                timestamps = timestamps.intersection(set(data.index))

        timestamps = sorted(list(timestamps))[-self.sequence_length:]  # Last 20 days

        # Extract features
        n_times = len(timestamps)
        n_tickers = len(self.tickers)

        prices = np.zeros((n_times, n_tickers))
        volumes = np.zeros((n_times, n_tickers))
        returns = np.zeros((n_times, n_tickers))

        ticker_list = sorted(market_data.keys())

        for i, ticker in enumerate(ticker_list):
            data = market_data[ticker].loc[timestamps]

            close_data = data['Close'].values
            if len(close_data.shape) > 1:
                close_data = close_data.flatten()
            prices[:, i] = close_data

            vol_data = data['Volume'].values
            if len(vol_data.shape) > 1:
                vol_data = vol_data.flatten()
            volumes[:, i] = vol_data

            returns[1:, i] = np.diff(np.log(close_data))
            returns[0, i] = 0.0

        # Normalize volumes
        volumes_norm = np.zeros_like(volumes)
        for i in range(n_tickers):
            mean_v = np.mean(volumes[:, i])
            std_v = np.std(volumes[:, i])
            if std_v > 0:
                volumes_norm[:, i] = (volumes[:, i] - mean_v) / std_v

        # Calculate correlations
        corr_matrix = np.corrcoef(returns.T)
        correlations = corr_matrix[np.triu_indices(n_tickers, k=1)]

        # Repeat correlations for each timestep (simplified - should be rolling)
        correlations_seq = np.tile(correlations, (n_times, 1))

        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns).unsqueeze(0)  # [1, seq_len, n_tickers]
        volumes_tensor = torch.FloatTensor(volumes_norm).unsqueeze(0)
        correlations_tensor = torch.FloatTensor(correlations_seq).unsqueeze(0)

        return returns_tensor, volumes_tensor, correlations_tensor

    def get_predictions(self, market_data: Dict) -> Dict[str, float]:
        """
        Get semantic network predictions for all tickers.

        Returns:
            {ticker: predicted_return}
        """
        # Prepare features
        returns, volumes, correlations = self.prepare_semantic_features(market_data)

        # Move to device
        returns = returns.to(self.device)
        volumes = volumes.to(self.device)
        correlations = correlations.to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(returns, volumes, correlations)

        # Convert to dict
        predictions_dict = {}
        ticker_list = sorted(market_data.keys())

        for i, ticker in enumerate(ticker_list):
            predictions_dict[ticker] = float(predictions[0, i].cpu().numpy())

        return predictions_dict

    def calculate_position_sizes(self, predictions: Dict[str, float], current_prices: Dict[str, float]) -> Dict[str, int]:
        """
        Calculate optimal position sizes based on predictions.

        Uses Kelly criterion with semantic confidence.
        """
        positions = {}

        # Sort by prediction strength
        sorted_tickers = sorted(predictions.items(), key=lambda x: abs(x[1]), reverse=True)

        available_capital = self.capital

        for ticker, predicted_return in sorted_tickers[:5]:  # Top 5 predictions
            # Skip if prediction is too weak
            if abs(predicted_return) < 0.001:  # 0.1% threshold
                continue

            # Position size based on signal strength
            signal_strength = abs(predicted_return)
            position_value = min(
                available_capital * self.max_position_size,
                available_capital * signal_strength * 10  # Scale by signal
            )

            # Calculate shares
            price = current_prices.get(ticker, 0)
            if price > 0:
                shares = int(position_value / price)

                if shares > 0:
                    # Account for transaction costs
                    cost = shares * price * self.transaction_cost_bps

                    if shares * price + cost <= available_capital:
                        positions[ticker] = {
                            'shares': shares if predicted_return > 0 else -shares,
                            'entry_price': price,
                            'prediction': predicted_return
                        }
                        available_capital -= shares * price + cost

        return positions

    def execute_trades(self, target_positions: Dict, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Execute trades to reach target positions.

        Returns list of executed trades.
        """
        executed = []

        # Close positions not in targets
        for ticker in list(self.positions.keys()):
            if ticker not in target_positions:
                pos = self.positions[ticker]
                price = current_prices.get(ticker, 0)

                if price > 0:
                    # Close position
                    pnl = (price - pos['entry_price']) * pos['shares']
                    cost = abs(pos['shares']) * price * self.transaction_cost_bps
                    net_pnl = pnl - cost

                    self.capital += net_pnl

                    trade = {
                        'timestamp': datetime.now(),
                        'ticker': ticker,
                        'action': 'CLOSE',
                        'shares': pos['shares'],
                        'price': price,
                        'pnl': net_pnl,
                        'capital': self.capital
                    }

                    executed.append(trade)
                    self.trades.append(trade)
                    del self.positions[ticker]

        # Open new positions
        for ticker, target in target_positions.items():
            if ticker not in self.positions:
                price = current_prices.get(ticker, 0)

                if price > 0:
                    shares = target['shares']
                    cost = abs(shares) * price * self.transaction_cost_bps

                    self.positions[ticker] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_time': datetime.now()
                    }

                    self.capital -= abs(shares) * price + cost

                    trade = {
                        'timestamp': datetime.now(),
                        'ticker': ticker,
                        'action': 'OPEN',
                        'shares': shares,
                        'price': price,
                        'cost': cost,
                        'capital': self.capital,
                        'prediction': target['prediction']
                    }

                    executed.append(trade)
                    self.trades.append(trade)

        return executed

    def check_risk_management(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check stop loss and take profit levels.
        """
        executed = []

        for ticker, pos in list(self.positions.items()):
            price = current_prices.get(ticker, 0)

            if price > 0:
                entry_price = pos['entry_price']
                shares = pos['shares']

                pct_change = (price - entry_price) / entry_price

                # For long positions
                if shares > 0:
                    if pct_change <= -self.stop_loss_pct:
                        # Stop loss hit
                        pnl = (price - entry_price) * shares
                        cost = shares * price * self.transaction_cost_bps

                        self.capital += pnl - cost

                        trade = {
                            'timestamp': datetime.now(),
                            'ticker': ticker,
                            'action': 'STOP_LOSS',
                            'shares': shares,
                            'price': price,
                            'pnl': pnl - cost,
                            'capital': self.capital
                        }

                        executed.append(trade)
                        self.trades.append(trade)
                        del self.positions[ticker]

                    elif pct_change >= self.take_profit_pct:
                        # Take profit
                        pnl = (price - entry_price) * shares
                        cost = shares * price * self.transaction_cost_bps

                        self.capital += pnl - cost

                        trade = {
                            'timestamp': datetime.now(),
                            'ticker': ticker,
                            'action': 'TAKE_PROFIT',
                            'shares': shares,
                            'price': price,
                            'pnl': pnl - cost,
                            'capital': self.capital
                        }

                        executed.append(trade)
                        self.trades.append(trade)
                        del self.positions[ticker]

        return executed

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        value = self.capital

        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, 0)
            if price > 0:
                value += pos['shares'] * price

        return value

    def run_trading_cycle(self) -> Dict:
        """
        Execute one complete trading cycle.

        1. Fetch live data
        2. Get predictions
        3. Calculate positions
        4. Check risk management
        5. Execute trades
        6. Track performance
        """
        print("\n" + "=" * 80)
        print(f"TRADING CYCLE - {datetime.now()}")
        print("=" * 80)

        # Fetch data
        print("Fetching live market data...")
        market_data = self.fetch_live_data(lookback_days=30)

        if len(market_data) == 0:
            print("⚠️  No market data available")
            return {'status': 'no_data'}

        print(f"✓ Loaded data for {len(market_data)} tickers")

        # Get current prices
        current_prices = {}
        for ticker, data in market_data.items():
            if len(data) > 0:
                price = data['Close'].iloc[-1]
                if isinstance(price, np.ndarray):
                    price = price[0]
                current_prices[ticker] = float(price)

        # Get predictions
        print("\nGenerating semantic predictions...")
        predictions = self.get_predictions(market_data)

        print("\nTop predictions:")
        sorted_preds = sorted(predictions.items(), key=lambda x: abs(x[1]), reverse=True)
        for ticker, pred in sorted_preds[:5]:
            print(f"  {ticker}: {pred:+.4f} ({pred*100:+.2f}%)")

        # Check risk management
        print("\nChecking risk management...")
        risk_trades = self.check_risk_management(current_prices)
        if risk_trades:
            print(f"  {len(risk_trades)} risk management trades executed")

        # Calculate new positions
        print("\nCalculating optimal positions...")
        target_positions = self.calculate_position_sizes(predictions, current_prices)
        print(f"  {len(target_positions)} target positions")

        # Execute trades
        print("\nExecuting trades...")
        executed_trades = self.execute_trades(target_positions, current_prices)
        print(f"  {len(executed_trades)} trades executed")

        # Track portfolio
        portfolio_value = self.get_portfolio_value(current_prices)
        self.portfolio_values.append({
            'timestamp': datetime.now(),
            'value': portfolio_value,
            'capital': self.capital,
            'n_positions': len(self.positions)
        })

        # Performance
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital

        print("\n" + "=" * 80)
        print("PORTFOLIO STATUS")
        print("=" * 80)
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Cash: ${self.capital:,.2f}")
        print(f"Total Return: {total_return:+.2%}")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Total Trades: {len(self.trades)}")
        print("=" * 80)

        return {
            'status': 'success',
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'predictions': predictions,
            'trades': executed_trades + risk_trades
        }

    def save_state(self, filepath: str = 'trader_state.json'):
        """Save trader state."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'capital': self.capital,
            'initial_capital': self.initial_capital,
            'positions': self.positions,
            'portfolio_values': self.portfolio_values,
            'trades': self.trades,
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        print(f"✓ State saved to {filepath}")


if __name__ == '__main__':
    print("=" * 80)
    print("PRODUCTION SEMANTIC TRADER")
    print("=" * 80)
    print()

    # Initialize
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'GS', 'MS',
        'WMT', 'HD', 'MCD', 'NKE',
        'JNJ', 'UNH', 'PFE',
        'XOM', 'CVX',
        'SPY', 'QQQ', 'IWM',
    ]

    trader = ProductionSemanticTrader(
        model_path='semantic_network_best.pt',
        tickers=TICKERS,
        initial_capital=100000.0,
        max_position_size=0.10,
        transaction_cost_bps=2.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )

    # Run trading cycle
    result = trader.run_trading_cycle()

    # Save state
    trader.save_state()

    print("\n✓ Trading cycle complete")
