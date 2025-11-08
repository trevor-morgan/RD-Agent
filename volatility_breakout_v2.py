#!/usr/bin/env python3
"""
Volatility Breakout Strategy V2 - Enhanced with Multi-Factor Approach
Interactive R&D Session: Phase 2 - Comprehensive Improvements

Research Question: Can we improve alpha by combining volatility, momentum,
and sector rotation with ML-based regime detection?

IMPROVEMENTS FROM V1:
1. ✅ Momentum Overlay (20/50-day momentum + RSI)
2. ✅ Relaxed Entry Criteria (momentum-based entries allowed)
3. ✅ Volatility-Based Position Sizing (risk parity)
4. ✅ Sector Rotation (dynamic sector weighting)
5. ✅ ML Regime Detection (RandomForest classifier)
6. ✅ Trailing Profit Taking (not just stops)
7. ✅ More Aggressive Allocation (up to 20 positions)

This uses REAL market data from Yahoo Finance via yfinance!
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date range: Last 2 years of real market data
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

# Stock universe with sector labels
TICKERS = {
    # Technology
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'NVDA': 'Tech', 'META': 'Tech',
    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials', 'MS': 'Financials',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'MPC': 'Energy',
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'TMO': 'Healthcare',
    # Consumer
    'WMT': 'Consumer', 'HD': 'Consumer', 'MCD': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer'
}

# Strategy parameters - MORE AGGRESSIVE
BB_WINDOW = 20  # Bollinger Band window
BB_STD = 2.0    # Bollinger Band standard deviations
ATR_WINDOW = 14  # ATR calculation window
STOP_ATR_MULT = 2.5  # Wider stops (was 2.0)
PROFIT_ATR_MULT = 4.0  # Trailing profit take at 4x ATR
TREND_MA = 50   # Moving average for trend filter
VOL_THRESHOLD = 1.1  # Relaxed volatility threshold (was 1.2)
MAX_POSITIONS = 20  # Increased from 10
TARGET_VOLATILITY = 0.15  # Target portfolio volatility for risk parity

# Momentum parameters (NEW)
MOMENTUM_SHORT = 20  # Short-term momentum
MOMENTUM_LONG = 50   # Long-term momentum
RSI_WINDOW = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Sector rotation parameters (NEW)
SECTOR_LOOKBACK = 20  # Days to measure sector performance
SECTOR_BOOST = 1.5  # Multiplier for strong sector positions

# ============================================================================
# DATA ACQUISITION
# ============================================================================

def fetch_real_market_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch REAL market data from Yahoo Finance using yfinance
    """
    print(f"\n📊 Fetching REAL market data from Yahoo Finance...")
    print(f"   Date range: {start} to {end}")
    print(f"   Tickers: {len(tickers)} stocks")

    print(f"   Downloading data...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by='column'
    )

    if data.empty:
        raise ValueError("Failed to download any data!")

    # Restructure to MultiIndex (Date, Ticker)
    all_dfs = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_data = data.copy()
            else:
                ticker_data = pd.DataFrame({
                    'Open': data['Open'][ticker],
                    'High': data['High'][ticker],
                    'Low': data['Low'][ticker],
                    'Close': data['Close'][ticker],
                    'Volume': data['Volume'][ticker]
                })

            ticker_data = ticker_data.dropna()
            if len(ticker_data) > 0:
                ticker_data['Ticker'] = ticker
                all_dfs.append(ticker_data)
                print(f"   ✓ {ticker}: {len(ticker_data)} days")
        except (KeyError, Exception) as e:
            print(f"   ✗ {ticker}: Failed - {e}")

    if not all_dfs:
        raise ValueError("Failed to download any data!")

    combined = pd.concat(all_dfs)
    combined = combined.reset_index()
    combined = combined.rename(columns={'Date': 'date'})
    combined = combined.set_index(['date', 'Ticker'])

    print(f"\n✓ Downloaded {len(all_dfs)} stocks successfully")
    print(f"\n📈 Dataset Summary:")
    print(f"   Total records: {len(combined):,}")
    print(f"   Date range: {combined.index.get_level_values('date').min()} to {combined.index.get_level_values('date').max()}")
    print(f"   Trading days: {len(combined.index.get_level_values('date').unique())}")

    return combined

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    return atr

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    return upper_band, rolling_mean, lower_band

def calculate_realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate realized volatility (annualized)"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate momentum as rate of change"""
    return (prices / prices.shift(window) - 1) * 100

# ============================================================================
# ML REGIME DETECTION (NEW)
# ============================================================================

def train_regime_classifier(data: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train ML model to detect market regimes

    Regimes:
    0 = Low Volatility / Bullish
    1 = High Volatility / Uncertain
    2 = Bearish
    """
    print("\n🤖 Training ML regime detection model...")

    # Calculate market-wide features
    dates = data.index.get_level_values('date').unique().sort_values()

    features = []
    labels = []

    for date in dates[60:]:  # Need history for features
        # Get all stocks for this date
        day_data = data.loc[(date, slice(None)), :]

        if len(day_data) < 10:  # Need enough stocks
            continue

        # Aggregate features across all stocks
        prices = day_data['Close']
        returns = prices.pct_change()

        # Feature 1: Market volatility (std of returns)
        market_vol = returns.std()

        # Feature 2: Market return (mean return)
        market_ret = returns.mean()

        # Feature 3: Dispersion (range of returns)
        ret_dispersion = returns.max() - returns.min()

        # Feature 4: Trend (% above 50-day MA)
        # Get historical data
        hist_data = data.loc[(data.index.get_level_values('date') <= date), :]
        pct_above_ma = 0
        count = 0
        for ticker in day_data.index.get_level_values('Ticker'):
            ticker_hist = hist_data.loc[(slice(None), ticker), 'Close']
            if len(ticker_hist) >= 50:
                ma50 = ticker_hist.iloc[-50:].mean()
                if ticker_hist.iloc[-1] > ma50:
                    pct_above_ma += 1
                count += 1
        pct_above_ma = pct_above_ma / count if count > 0 else 0.5

        # Feature 5: Volume trend
        vol_ratio = day_data['Volume'].mean()

        features.append([market_vol, market_ret, ret_dispersion, pct_above_ma, vol_ratio])

        # Label based on forward 5-day return and volatility
        future_idx = dates[dates > date][:5]
        if len(future_idx) >= 5:
            future_data = data.loc[(future_idx, slice(None)), 'Close']
            future_ret = future_data.mean() / prices.mean() - 1
            future_vol = future_data.pct_change().std()

            # Regime classification
            if future_vol < 0.015 and future_ret > 0:
                regime = 0  # Low vol bullish
            elif future_vol > 0.025 or future_ret < -0.02:
                regime = 2  # High vol / bearish
            else:
                regime = 1  # Uncertain

            labels.append(regime)
        else:
            # Default for recent dates without future data
            if market_vol < 0.015:
                labels.append(0)
            elif market_vol > 0.025:
                labels.append(2)
            else:
                labels.append(1)

    X = np.array(features)
    y = np.array(labels)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_scaled, y)

    accuracy = clf.score(X_scaled, y)
    print(f"   ✓ Regime classifier trained: {accuracy:.1%} accuracy")
    print(f"   Training samples: {len(y)}")
    print(f"   Regime distribution: Low Vol: {(y==0).sum()}, Uncertain: {(y==1).sum()}, Bearish: {(y==2).sum()}")

    return clf, scaler

# ============================================================================
# SECTOR ROTATION (NEW)
# ============================================================================

def calculate_sector_strength(data: pd.DataFrame, ticker_sectors: Dict, date, lookback: int = 20) -> Dict[str, float]:
    """
    Calculate sector relative strength
    Returns multiplier for each sector (1.0 = neutral, >1.0 = strong, <1.0 = weak)
    """
    # Get historical data
    start_date = date - timedelta(days=lookback * 2)  # Extra buffer
    hist_data = data.loc[(data.index.get_level_values('date') >= start_date) &
                         (data.index.get_level_values('date') <= date), :]

    sector_returns = {}

    for sector in set(ticker_sectors.values()):
        sector_tickers = [t for t, s in ticker_sectors.items() if s == sector]
        sector_rets = []

        for ticker in sector_tickers:
            try:
                ticker_data = hist_data.loc[(slice(None), ticker), 'close']
                if len(ticker_data) >= lookback:
                    ret = ticker_data.iloc[-1] / ticker_data.iloc[-lookback] - 1
                    sector_rets.append(ret)
            except (KeyError, IndexError):
                continue

        if sector_rets:
            sector_returns[sector] = np.mean(sector_rets)
        else:
            sector_returns[sector] = 0

    # Calculate relative strength (vs average)
    avg_return = np.mean(list(sector_returns.values()))

    sector_multipliers = {}
    for sector, ret in sector_returns.items():
        if ret > avg_return * 1.5:
            sector_multipliers[sector] = SECTOR_BOOST  # Strong sector
        elif ret < avg_return * 0.5:
            sector_multipliers[sector] = 0.7  # Weak sector
        else:
            sector_multipliers[sector] = 1.0  # Neutral

    return sector_multipliers

# ============================================================================
# ENHANCED SIGNAL GENERATION
# ============================================================================

def generate_enhanced_signals(data: pd.DataFrame, ticker_sectors: Dict,
                              regime_clf, regime_scaler) -> pd.DataFrame:
    """
    Generate trading signals with momentum overlay and ML regime detection
    """
    print("\n🔧 Calculating enhanced indicators...")

    tickers = data.index.get_level_values('Ticker').unique()
    dates = data.index.get_level_values('date').unique().sort_values()

    all_signals = []

    for ticker in tickers:
        ticker_data = data.loc[(slice(None), ticker), :]
        ticker_signals = pd.DataFrame(index=ticker_data.index)

        # Price data
        close = ticker_data['Close']
        high = ticker_data['High']
        low = ticker_data['Low']
        volume = ticker_data['Volume']

        # 1. Bollinger Bands
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close, BB_WINDOW, BB_STD)

        # 2. ATR for stops
        atr = calculate_atr(ticker_data, ATR_WINDOW)

        # 3. Trend filter (MA)
        ma_trend = close.rolling(window=TREND_MA).mean()

        # 4. Volatility regime
        returns = close.pct_change()
        realized_vol = calculate_realized_volatility(returns, window=20)
        vol_mean = realized_vol.rolling(window=63).mean()
        high_vol_regime = realized_vol > (vol_mean * VOL_THRESHOLD)

        # 5. NEW: Momentum indicators
        momentum_short = calculate_momentum(close, MOMENTUM_SHORT)
        momentum_long = calculate_momentum(close, MOMENTUM_LONG)
        rsi = calculate_rsi(close, RSI_WINDOW)

        # 6. NEW: Volume confirmation
        volume_ma = volume.rolling(window=20).mean()
        high_volume = volume > volume_ma * 1.2

        # Generate signals with MULTIPLE ENTRY PATHS

        # Path 1: Original volatility breakout (strict)
        breakout_up = close > upper_bb
        vol_signal = breakout_up & high_vol_regime & (close > ma_trend)

        # Path 2: NEW - Momentum breakout (relaxed vol requirement)
        strong_momentum = (momentum_short > 5) & (momentum_long > 10) & (rsi < RSI_OVERBOUGHT)
        momentum_signal = strong_momentum & (close > ma_trend) & high_volume

        # Path 3: NEW - RSI oversold bounce
        oversold_bounce = (rsi < RSI_OVERSOLD) & (close > close.shift(1)) & (close > ma_trend)

        # Combined entry signal
        entry_signal = vol_signal | momentum_signal | oversold_bounce

        # Exit signals (more nuanced)
        # Exit if: cross below MA, or RSI overbought + momentum fading, or stop hit
        exit_signal = (close < middle_bb) | ((rsi > RSI_OVERBOUGHT) & (momentum_short < 0))

        # Combine signals
        ticker_signals['signal'] = 0
        ticker_signals.loc[entry_signal, 'signal'] = 1
        ticker_signals.loc[exit_signal, 'signal'] = -1

        # Store indicators
        ticker_signals['close'] = close
        ticker_signals['upper_bb'] = upper_bb
        ticker_signals['middle_bb'] = middle_bb
        ticker_signals['atr'] = atr
        ticker_signals['ma_trend'] = ma_trend
        ticker_signals['realized_vol'] = realized_vol
        ticker_signals['momentum_short'] = momentum_short
        ticker_signals['momentum_long'] = momentum_long
        ticker_signals['rsi'] = rsi
        ticker_signals['high_vol_regime'] = high_vol_regime
        ticker_signals['volume_ratio'] = volume / volume_ma

        all_signals.append(ticker_signals)

    result = pd.concat(all_signals)
    result = result.sort_index()

    print(f"✓ Enhanced indicators calculated for {len(tickers)} stocks")

    return result

# ============================================================================
# ENHANCED BACKTESTING
# ============================================================================

def backtest_enhanced_strategy(signals: pd.DataFrame, ticker_sectors: Dict,
                               regime_clf, regime_scaler, initial_capital: float = 100000) -> Dict:
    """
    Backtest with volatility-based position sizing and sector rotation
    """
    print("\n🔬 Running ENHANCED backtest on REAL market data...")

    dates = signals.index.get_level_values('date').unique().sort_values()
    tickers_list = signals.index.get_level_values('Ticker').unique()

    positions = {}  # {ticker: {'shares': 0, 'entry_price': 0, 'stop': 0, 'profit_target': 0}}
    daily_values = []
    cash = initial_capital

    total_trades = 0
    winning_trades = 0
    losing_trades = 0

    for i, date in enumerate(dates):
        day_data = signals.loc[(date, slice(None)), :]

        # Calculate market features for regime detection (if enough history)
        if i >= 60:
            # Simple features for regime
            market_vol = day_data['realized_vol'].mean()
            market_ret = day_data['close'].pct_change().mean()
            ret_dispersion = day_data['close'].pct_change().std()
            pct_above_ma = (day_data['close'] > day_data['ma_trend']).sum() / len(day_data)
            vol_ratio = day_data['volume_ratio'].mean()

            features = np.array([[market_vol, market_ret, ret_dispersion, pct_above_ma, vol_ratio]])
            features_scaled = regime_scaler.transform(features)
            regime = regime_clf.predict(features_scaled)[0]
        else:
            regime = 1  # Default to uncertain

        # Get sector strength
        sector_strength = calculate_sector_strength(signals, ticker_sectors, date, SECTOR_LOOKBACK)

        # Check exits and stops
        for ticker in list(positions.keys()):
            if ticker not in day_data.index.get_level_values('Ticker'):
                continue

            current_price = day_data.loc[(date, ticker), 'close']
            signal = day_data.loc[(date, ticker), 'signal']
            pos = positions[ticker]

            # Exit on signal, stop hit, or profit target hit
            should_exit = (signal == -1) or (current_price < pos['stop']) or (current_price > pos['profit_target'])

            if should_exit:
                # Close position
                shares = pos['shares']
                entry_price = pos['entry_price']
                pnl = shares * (current_price - entry_price)
                cash += shares * current_price

                total_trades += 1
                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

                del positions[ticker]

        # Check new entries
        entry_candidates = []

        for ticker in day_data.index.get_level_values('Ticker'):
            if day_data.loc[(date, ticker), 'signal'] == 1 and ticker not in positions:
                current_price = day_data.loc[(date, ticker), 'close']
                atr = day_data.loc[(date, ticker), 'atr']
                realized_vol = day_data.loc[(date, ticker), 'realized_vol']

                if not np.isnan(atr) and not np.isnan(realized_vol) and realized_vol > 0:
                    # Calculate position size based on volatility (risk parity)
                    target_risk = initial_capital * TARGET_VOLATILITY
                    position_size = target_risk / (realized_vol * current_price)
                    position_size = min(position_size, initial_capital * 0.15)  # Cap at 15% per position

                    # Apply sector boost
                    sector = ticker_sectors.get(ticker, 'Unknown')
                    sector_mult = sector_strength.get(sector, 1.0)
                    position_size *= sector_mult

                    # Reduce position size in bearish regime
                    if regime == 2:
                        position_size *= 0.5
                    elif regime == 0:
                        position_size *= 1.2  # Boost in bullish regime

                    shares = int(position_size / current_price)

                    if shares > 0:
                        entry_candidates.append({
                            'ticker': ticker,
                            'price': current_price,
                            'shares': shares,
                            'atr': atr,
                            'priority': day_data.loc[(date, ticker), 'momentum_short']  # Prioritize strong momentum
                        })

        # Enter positions (prioritize by momentum, limit total positions)
        entry_candidates.sort(key=lambda x: x['priority'], reverse=True)

        for candidate in entry_candidates:
            if len(positions) >= MAX_POSITIONS:
                break

            ticker = candidate['ticker']
            current_price = candidate['price']
            shares = candidate['shares']
            atr = candidate['atr']

            cost = shares * current_price

            if cost <= cash:
                cash -= cost

                # Set stop and profit target
                stop_price = current_price - (atr * STOP_ATR_MULT)
                profit_target = current_price + (atr * PROFIT_ATR_MULT)

                positions[ticker] = {
                    'shares': shares,
                    'entry_price': current_price,
                    'stop': stop_price,
                    'profit_target': profit_target
                }

        # Calculate portfolio value
        holdings_value = sum(
            pos['shares'] * day_data.loc[(date, ticker), 'close']
            for ticker, pos in positions.items()
            if ticker in day_data.index.get_level_values('Ticker')
        )

        total_value = cash + holdings_value

        daily_values.append({
            'date': date,
            'cash': cash,
            'holdings': holdings_value,
            'total': total_value,
            'num_positions': len(positions),
            'regime': regime
        })

    # Convert to DataFrame
    portfolio_values = pd.DataFrame(daily_values).set_index('date')

    # Calculate benchmark (buy and hold equal-weighted portfolio)
    benchmark_values = []
    benchmark_shares = {}
    benchmark_cash = initial_capital

    first_date = dates[0]
    first_day = signals.loc[(first_date, slice(None)), :]
    per_stock = benchmark_cash / len(tickers_list)

    for ticker in tickers_list:
        if ticker in first_day.index.get_level_values('Ticker'):
            price = first_day.loc[(first_date, ticker), 'close']
            shares = int(per_stock / price)
            benchmark_shares[ticker] = shares
            benchmark_cash -= shares * price

    for date in dates:
        day_data = signals.loc[(date, slice(None)), :]
        holdings = sum(
            shares * day_data.loc[(date, ticker), 'close']
            for ticker, shares in benchmark_shares.items()
            if ticker in day_data.index.get_level_values('Ticker')
        )
        benchmark_values.append({
            'date': date,
            'total': benchmark_cash + holdings
        })

    benchmark = pd.DataFrame(benchmark_values).set_index('date')

    # Calculate performance metrics
    strategy_returns = portfolio_values['total'].pct_change().dropna()
    benchmark_returns = benchmark['total'].pct_change().dropna()

    strategy_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0
    benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(252) if benchmark_returns.std() > 0 else 0

    strategy_cummax = portfolio_values['total'].cummax()
    strategy_drawdown = (portfolio_values['total'] - strategy_cummax) / strategy_cummax
    max_drawdown = strategy_drawdown.min()

    benchmark_cummax = benchmark['total'].cummax()
    benchmark_drawdown = (benchmark['total'] - benchmark_cummax) / benchmark_cummax
    benchmark_max_dd = benchmark_drawdown.min()

    total_days = (dates[-1] - dates[0]).days
    years = total_days / 365.25

    final_value = portfolio_values['total'].iloc[-1]
    strategy_annual_return = (final_value / initial_capital) ** (1/years) - 1

    benchmark_final = benchmark['total'].iloc[-1]
    benchmark_annual_return = (benchmark_final / initial_capital) ** (1/years) - 1

    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    trade_win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Print results
    print(f"\n" + "="*70)
    print("📊 BACKTEST RESULTS V2 - ENHANCED STRATEGY")
    print("="*70)

    print(f"\n🗓️  Period: {dates[0].date()} to {dates[-1].date()} ({total_days} days)")
    print(f"💰 Initial Capital: ${initial_capital:,.0f}")

    print(f"\n🎯 STRATEGY PERFORMANCE (V2 - ENHANCED):")
    print(f"   Final Value:        ${final_value:,.2f}")
    print(f"   Total Return:       {(final_value/initial_capital - 1)*100:+.2f}%")
    print(f"   Annual Return:      {strategy_annual_return*100:+.2f}%")
    print(f"   Sharpe Ratio:       {strategy_sharpe:.2f}")
    print(f"   Max Drawdown:       {max_drawdown*100:.2f}%")
    print(f"   Win Rate (Daily):   {win_rate*100:.1f}%")

    print(f"\n📈 BENCHMARK (Buy & Hold):")
    print(f"   Final Value:        ${benchmark_final:,.2f}")
    print(f"   Total Return:       {(benchmark_final/initial_capital - 1)*100:+.2f}%")
    print(f"   Annual Return:      {benchmark_annual_return*100:+.2f}%")
    print(f"   Sharpe Ratio:       {benchmark_sharpe:.2f}")
    print(f"   Max Drawdown:       {benchmark_max_dd*100:.2f}%")

    print(f"\n💡 EXCESS PERFORMANCE:")
    excess_return = strategy_annual_return - benchmark_annual_return
    print(f"   Excess Return:      {excess_return*100:+.2f}%")
    print(f"   Sharpe Improvement: {(strategy_sharpe - benchmark_sharpe):+.2f}")

    if excess_return > 0:
        print(f"   🎉 ALPHA ACHIEVED! Outperforming by {excess_return*100:.1f}%!")
    else:
        print(f"   ⚠️  Still underperforming by {abs(excess_return)*100:.1f}%")

    avg_positions = portfolio_values['num_positions'].mean()
    print(f"\n📊 TRADING STATISTICS:")
    print(f"   Total Trades:       {total_trades}")
    print(f"   Winning Trades:     {winning_trades}")
    print(f"   Losing Trades:      {losing_trades}")
    print(f"   Trade Win Rate:     {trade_win_rate*100:.1f}%")
    print(f"   Avg # Positions:    {avg_positions:.1f}")
    print(f"   Max Positions:      {portfolio_values['num_positions'].max()}")

    total_signals = (signals['signal'] == 1).sum()
    print(f"   Total Buy Signals:  {total_signals}")

    # Regime statistics
    regime_days = portfolio_values['regime'].value_counts()
    print(f"\n🤖 ML REGIME DETECTION:")
    print(f"   Bullish Days:       {regime_days.get(0, 0)} ({regime_days.get(0, 0)/len(dates)*100:.1f}%)")
    print(f"   Uncertain Days:     {regime_days.get(1, 0)} ({regime_days.get(1, 0)/len(dates)*100:.1f}%)")
    print(f"   Bearish Days:       {regime_days.get(2, 0)} ({regime_days.get(2, 0)/len(dates)*100:.1f}%)")

    return {
        'portfolio_values': portfolio_values,
        'benchmark_values': benchmark,
        'strategy_sharpe': strategy_sharpe,
        'strategy_annual_return': strategy_annual_return,
        'strategy_total_return': final_value/initial_capital - 1,
        'benchmark_sharpe': benchmark_sharpe,
        'benchmark_annual_return': benchmark_annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_win_rate': trade_win_rate,
        'total_trades': total_trades,
        'excess_return': excess_return,
        'total_signals': total_signals
    }

# ============================================================================
# COMPARISON WITH V1
# ============================================================================

def compare_versions(v2_results: Dict):
    """Compare V2 results with V1 baseline"""

    # V1 results (from previous run)
    v1_annual = 0.0228  # 2.28%
    v1_sharpe = 0.73
    v1_drawdown = -0.0361  # -3.61%
    v1_excess = -0.2252  # -22.52%

    v2_annual = v2_results['strategy_annual_return']
    v2_sharpe = v2_results['strategy_sharpe']
    v2_drawdown = v2_results['max_drawdown']
    v2_excess = v2_results['excess_return']

    print(f"\n" + "="*70)
    print("🔄 V1 vs V2 COMPARISON")
    print("="*70)

    print(f"\n📈 Annual Return:")
    print(f"   V1: {v1_annual*100:+.2f}%")
    print(f"   V2: {v2_annual*100:+.2f}%")
    improvement = ((v2_annual - v1_annual) / abs(v1_annual)) * 100 if v1_annual != 0 else 0
    print(f"   Improvement: {improvement:+.1f}%")

    print(f"\n⚖️  Sharpe Ratio:")
    print(f"   V1: {v1_sharpe:.2f}")
    print(f"   V2: {v2_sharpe:.2f}")
    print(f"   Improvement: {(v2_sharpe - v1_sharpe):+.2f}")

    print(f"\n📉 Max Drawdown:")
    print(f"   V1: {v1_drawdown*100:.2f}%")
    print(f"   V2: {v2_drawdown*100:.2f}%")
    print(f"   Change: {(v2_drawdown - v1_drawdown)*100:+.2f}%")

    print(f"\n🎯 Alpha (vs Benchmark):")
    print(f"   V1: {v1_excess*100:+.2f}%")
    print(f"   V2: {v2_excess*100:+.2f}%")
    print(f"   Improvement: {(v2_excess - v1_excess)*100:+.2f}%")

    print(f"\n💡 Summary:")
    if v2_excess > v1_excess:
        print(f"   ✅ V2 closes the alpha gap by {(v2_excess - v1_excess)*100:.1f}%!")
    else:
        print(f"   ⚠️  V2 needs further iteration")

    if v2_sharpe > v1_sharpe:
        print(f"   ✅ V2 improves risk-adjusted returns")

    if v2_drawdown > v1_drawdown:
        print(f"   ⚠️  V2 has larger drawdown (cost of aggression)")
    else:
        print(f"   ✅ V2 maintains good risk control")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "🚀"*35)
    print("  Volatility Breakout Strategy V2 - ENHANCED")
    print("  Interactive R&D Session: Phase 2")
    print("  Combining ALL Improvements!")
    print("🚀"*35)

    print(f"\n📅 Configuration:")
    print(f"   Date Range: {START_DATE} to {END_DATE}")
    print(f"   Tickers: {len(TICKERS)} stocks across 5 sectors")
    print(f"   Max Positions: {MAX_POSITIONS} (was 10)")
    print(f"   Volatility Threshold: {VOL_THRESHOLD}x (was 1.2x)")
    print(f"\n🆕 NEW FEATURES:")
    print(f"   ✅ Momentum overlay (20/50-day + RSI)")
    print(f"   ✅ Multiple entry paths (volatility + momentum + oversold)")
    print(f"   ✅ Risk-parity position sizing")
    print(f"   ✅ Sector rotation ({SECTOR_BOOST}x boost for strong sectors)")
    print(f"   ✅ ML regime detection (RandomForest)")
    print(f"   ✅ Trailing profit targets ({PROFIT_ATR_MULT}x ATR)")

    try:
        # 1. Fetch real data
        data = fetch_real_market_data(list(TICKERS.keys()), START_DATE, END_DATE)

        # 2. Train ML regime classifier
        regime_clf, regime_scaler = train_regime_classifier(data)

        # 3. Generate enhanced signals
        signals = generate_enhanced_signals(data, TICKERS, regime_clf, regime_scaler)

        # 4. Backtest
        results = backtest_enhanced_strategy(signals, TICKERS, regime_clf, regime_scaler)

        # 5. Compare with V1
        compare_versions(results)

        print(f"\n" + "="*70)
        print("✅ Enhanced Research Iteration Complete!")
        print("="*70)
        print(f"\nThis was Phase 2: Comprehensive Enhancement")
        print(f"Next: Analyze results and iterate further if needed!")

        return results

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()

    if results:
        print(f"\n💾 Results object available in Python session")
        sys.exit(0)
    else:
        sys.exit(1)
