"""
SEMANTIC SPACE DATA LOADER
Gather maximum available data for semantic trading

The universe is semantic space:
- Market states are embeddings
- Similar conditions cluster together
- Transformers learn the language of markets

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def test_data_availability():
    """Test what data we can actually get from yfinance."""

    print("=" * 80)
    print("TESTING DATA AVAILABILITY")
    print("=" * 80)
    print()

    ticker = "AAPL"

    # Test different intervals and periods
    tests = [
        ('1m', 7),    # 1-minute, 7 days
        ('5m', 60),   # 5-minute, 60 days
        ('15m', 60),  # 15-minute, 60 days
        ('1h', 730),  # 1-hour, 2 years
        ('1d', 3650), # Daily, 10 years
    ]

    results = []

    for interval, days in tests:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            print(f"Testing {interval} for {days} days...")

            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                progress=False
            )

            if len(data) > 0:
                print(f"  ✓ Got {len(data)} bars")
                print(f"  Date range: {data.index[0]} to {data.index[-1]}")

                # Calculate actual days
                actual_days = (data.index[-1] - data.index[0]).days
                print(f"  Actual span: {actual_days} days")

                results.append({
                    'interval': interval,
                    'requested_days': days,
                    'actual_days': actual_days,
                    'bars': len(data),
                    'success': True
                })
            else:
                print(f"  ✗ No data")
                results.append({
                    'interval': interval,
                    'requested_days': days,
                    'success': False
                })

            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'interval': interval,
                'requested_days': days,
                'success': False,
                'error': str(e)
            })
            print()

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find best option
    successful = [r for r in results if r.get('success', False)]

    if successful:
        # Sort by total bars
        successful.sort(key=lambda x: x['bars'], reverse=True)

        print("Best options (by total data points):")
        for i, r in enumerate(successful[:3], 1):
            print(f"{i}. {r['interval']} interval: {r['bars']} bars over {r['actual_days']} days")

        print()

        # Recommendation
        best = successful[0]
        print(f"RECOMMENDED: {best['interval']} interval")
        print(f"  {best['bars']} bars")
        print(f"  {best['actual_days']} days of data")
        print(f"  ~{best['bars'] / max(best['actual_days'], 1):.1f} bars per day")

        return best['interval'], best['actual_days']

    else:
        print("⚠️  No successful data retrieval")
        return None, None


def load_semantic_dataset(
    tickers: list,
    interval: str,
    days: int
) -> dict:
    """
    Load dataset for semantic space training.

    Returns rich dataset with:
    - Price data
    - Volume
    - Returns
    - Cross-asset correlations
    - Market regime features
    """

    print("=" * 80)
    print("LOADING SEMANTIC DATASET")
    print("=" * 80)
    print()

    print(f"Tickers: {len(tickers)}")
    print(f"Interval: {interval}")
    print(f"Period: {days} days")
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_data = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                progress=False
            )

            if len(data) > 0:
                all_data[ticker] = data
                print(f"  ✓ {len(data)} bars")
            else:
                print(f"  ✗ No data")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()
    print(f"Successfully loaded {len(all_data)}/{len(tickers)} tickers")
    print()

    # Create unified dataset
    print("Creating unified dataset...")

    # Find common timestamps
    if len(all_data) == 0:
        print("⚠️  No data loaded")
        return None

    # Get intersection of all timestamps
    timestamps = None
    for ticker, data in all_data.items():
        if timestamps is None:
            timestamps = set(data.index)
        else:
            timestamps = timestamps.intersection(set(data.index))

    timestamps = sorted(list(timestamps))
    print(f"  Common timestamps: {len(timestamps)}")

    if len(timestamps) == 0:
        print("⚠️  No common timestamps")
        return None

    # Create features
    n_times = len(timestamps)
    n_tickers = len(all_data)

    # Initialize arrays
    prices = np.zeros((n_times, n_tickers))
    volumes = np.zeros((n_times, n_tickers))
    returns = np.zeros((n_times, n_tickers))

    ticker_list = sorted(all_data.keys())

    for i, ticker in enumerate(ticker_list):
        data = all_data[ticker].loc[timestamps]

        # Price (close) - flatten to handle multi-index
        close_data = data['Close'].values
        if len(close_data.shape) > 1:
            close_data = close_data.flatten()
        prices[:, i] = close_data

        # Volume - flatten to handle multi-index
        vol_data = data['Volume'].values
        if len(vol_data.shape) > 1:
            vol_data = vol_data.flatten()
        volumes[:, i] = vol_data

        # Returns
        returns[1:, i] = np.diff(np.log(close_data))
        returns[0, i] = 0.0

    print(f"  Prices: {prices.shape}")
    print(f"  Volumes: {volumes.shape}")
    print(f"  Returns: {returns.shape}")
    print()

    # Create semantic features
    print("Creating semantic features...")

    # Normalize prices to [0, 1] per ticker
    price_norm = np.zeros_like(prices)
    for i in range(n_tickers):
        min_p = np.min(prices[:, i])
        max_p = np.max(prices[:, i])
        if max_p > min_p:
            price_norm[:, i] = (prices[:, i] - min_p) / (max_p - min_p)

    # Normalize volumes
    vol_norm = np.zeros_like(volumes)
    for i in range(n_tickers):
        mean_v = np.mean(volumes[:, i])
        std_v = np.std(volumes[:, i])
        if std_v > 0:
            vol_norm[:, i] = (volumes[:, i] - mean_v) / std_v

    # Cross-asset correlations (rolling 20-period)
    correlations = []
    for t in range(20, n_times):
        window_returns = returns[t-20:t, :]
        corr_matrix = np.corrcoef(window_returns.T)
        # Take upper triangle (unique correlations)
        upper_tri = corr_matrix[np.triu_indices(n_tickers, k=1)]
        correlations.append(upper_tri)

    # Pad with zeros for first 20 periods
    for t in range(20):
        correlations.insert(0, np.zeros(len(correlations[0]) if correlations else 0))

    correlations = np.array(correlations)
    print(f"  Correlations: {correlations.shape}")
    print()

    dataset = {
        'timestamps': timestamps,
        'tickers': ticker_list,
        'prices': prices,
        'volumes': volumes,
        'returns': returns,
        'price_norm': price_norm,
        'vol_norm': vol_norm,
        'correlations': correlations,
        'n_times': n_times,
        'n_tickers': n_tickers,
    }

    print("✓ Semantic dataset ready")
    print()

    return dataset


if __name__ == '__main__':
    # Test what we can get
    best_interval, best_days = test_data_availability()

    if best_interval:
        print()
        print("=" * 80)
        print("LOADING FULL DATASET")
        print("=" * 80)
        print()

        # Expanded universe for richer semantic space
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
            # Indices (if available)
            'SPY', 'QQQ', 'IWM',
        ]

        dataset = load_semantic_dataset(
            TICKERS,
            interval=best_interval,
            days=best_days
        )

        if dataset:
            print("=" * 80)
            print("DATASET SUMMARY")
            print("=" * 80)
            print()
            print(f"Timestamps: {dataset['n_times']}")
            print(f"Tickers: {dataset['n_tickers']}")
            print(f"Returns shape: {dataset['returns'].shape}")
            print(f"Correlations shape: {dataset['correlations'].shape}")
            print(f"Date range: {dataset['timestamps'][0]} to {dataset['timestamps'][-1]}")
            print()
            print(f"Total data points: {dataset['n_times'] * dataset['n_tickers']:,}")
            print()
            print("✓ Ready for semantic space neural network training")
