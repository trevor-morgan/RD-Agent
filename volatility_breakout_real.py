#!/usr/bin/env python3
"""
Volatility Breakout Strategy - Using REAL Market Data
Interactive R&D Session: Phase 1 - Initial Implementation

Research Question: Can a volatility breakout strategy outperform buy-and-hold in 2025 markets?

Strategy Components:
1. Bollinger Band Breakouts (20-day, 2σ)
2. ATR-based Trailing Stops (2x 14-day ATR)
3. Volatility Regime Detection
4. Trend Filter (50-day MA)

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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date range: Last 2 years of real market data
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

# Stock universe: Diverse sectors for robust testing
TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'MPC',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO',
    # Consumer
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX'
]

# Strategy parameters
BB_WINDOW = 20  # Bollinger Band window
BB_STD = 2.0    # Bollinger Band standard deviations
ATR_WINDOW = 14  # ATR calculation window
STOP_ATR_MULT = 2.0  # Stop distance as multiple of ATR
TREND_MA = 50   # Moving average for trend filter
VOL_THRESHOLD = 1.2  # Volatility regime threshold (>1.2x mean = high vol)
POSITION_SIZE = 1.0  # Position size per signal (1.0 = 100%)

# ============================================================================
# DATA ACQUISITION
# ============================================================================

def fetch_real_market_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch REAL market data from Yahoo Finance using yfinance

    Returns DataFrame with MultiIndex (Date, Ticker) and columns [Open, High, Low, Close, Volume]
    """
    print(f"\n📊 Fetching REAL market data from Yahoo Finance...")
    print(f"   Date range: {start} to {end}")
    print(f"   Tickers: {len(tickers)} stocks")

    # Download all data in one go (yfinance is optimized for batch downloads)
    print(f"   Downloading data...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,  # Use adjusted prices
        group_by='column'   # Group by price type (Open, High, Low, Close, Volume)
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

    # Combine all data
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

# ============================================================================
# STRATEGY IMPLEMENTATION
# ============================================================================

def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on volatility breakout strategy

    Signals:
    1 = Long
    0 = Flat
    -1 = Short (not used in this long-only strategy)
    """
    print("\n🔧 Calculating technical indicators...")

    signals = pd.DataFrame(index=data.index)
    signals['close'] = data['Close']
    signals['signal'] = 0
    signals['position'] = 0.0

    # Group by ticker for indicator calculation
    tickers = data.index.get_level_values('Ticker').unique()

    all_signals = []

    for ticker in tickers:
        ticker_data = data.loc[(slice(None), ticker), :]
        ticker_signals = pd.DataFrame(index=ticker_data.index)

        # Price data
        close = ticker_data['Close']
        high = ticker_data['High']
        low = ticker_data['Low']

        # 1. Bollinger Bands
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close, BB_WINDOW, BB_STD)

        # 2. ATR for stops
        atr = calculate_atr(ticker_data, ATR_WINDOW)

        # 3. Trend filter (MA)
        ma_trend = close.rolling(window=TREND_MA).mean()

        # 4. Volatility regime
        returns = close.pct_change()
        realized_vol = calculate_realized_volatility(returns, window=20)
        vol_mean = realized_vol.rolling(window=63).mean()  # Quarterly average
        high_vol_regime = realized_vol > (vol_mean * VOL_THRESHOLD)

        # Generate signals
        # Long signal: Price breaks above upper BB + high vol regime + above MA
        breakout_up = close > upper_bb
        above_trend = close > ma_trend

        long_signal = breakout_up & high_vol_regime & above_trend

        # Exit signal: Price crosses below middle BB or below MA
        exit_signal = (close < middle_bb) | (close < ma_trend)

        # Combine signals
        ticker_signals['signal'] = 0
        ticker_signals.loc[long_signal, 'signal'] = 1
        ticker_signals.loc[exit_signal, 'signal'] = -1

        # Store indicators for analysis
        ticker_signals['close'] = close
        ticker_signals['upper_bb'] = upper_bb
        ticker_signals['middle_bb'] = middle_bb
        ticker_signals['lower_bb'] = lower_bb
        ticker_signals['atr'] = atr
        ticker_signals['ma_trend'] = ma_trend
        ticker_signals['realized_vol'] = realized_vol
        ticker_signals['high_vol_regime'] = high_vol_regime

        all_signals.append(ticker_signals)

    result = pd.concat(all_signals)
    result = result.sort_index()

    print(f"✓ Indicators calculated for {len(tickers)} stocks")

    return result

def backtest_strategy(signals: pd.DataFrame, initial_capital: float = 100000) -> Dict:
    """
    Backtest the volatility breakout strategy

    Returns dictionary with performance metrics
    """
    print("\n🔬 Running backtest on REAL market data...")

    # Initialize portfolio
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['signal'] = signals['signal']
    portfolio['close'] = signals['close']

    # Position tracking
    portfolio['position'] = 0.0
    portfolio['cash'] = initial_capital
    portfolio['holdings'] = 0.0
    portfolio['total'] = initial_capital

    # Track positions by ticker
    dates = signals.index.get_level_values('date').unique().sort_values()
    tickers = signals.index.get_level_values('Ticker').unique()

    positions = {}  # {ticker: {'shares': 0, 'entry_price': 0, 'stop': 0}}
    daily_values = []
    cash = initial_capital

    for date in dates:
        day_data = signals.loc[(date, slice(None)), :]

        # Check exits and stops
        for ticker in list(positions.keys()):
            if ticker not in day_data.index.get_level_values('Ticker'):
                continue

            current_price = day_data.loc[(date, ticker), 'close']
            signal = day_data.loc[(date, ticker), 'signal']

            # Exit on signal or stop hit
            if signal == -1 or current_price < positions[ticker]['stop']:
                # Close position
                shares = positions[ticker]['shares']
                entry_price = positions[ticker]['entry_price']
                pnl = shares * (current_price - entry_price)
                cash += shares * current_price

                del positions[ticker]

        # Check new entries
        for ticker in day_data.index.get_level_values('Ticker'):
            if day_data.loc[(date, ticker), 'signal'] == 1 and ticker not in positions:
                # Enter position
                current_price = day_data.loc[(date, ticker), 'close']
                atr = day_data.loc[(date, ticker), 'atr']

                if not np.isnan(atr) and cash > 0:
                    # Size position (equal weight across signals, but respect cash)
                    max_positions = 10  # Max 10 concurrent positions
                    position_value = min(cash / max_positions, cash)
                    shares = int(position_value / current_price)

                    if shares > 0:
                        cost = shares * current_price
                        cash -= cost

                        # Set stop at entry - 2*ATR
                        stop_price = current_price - (atr * STOP_ATR_MULT)

                        positions[ticker] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'stop': stop_price
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
            'num_positions': len(positions)
        })

    # Convert to DataFrame
    portfolio_values = pd.DataFrame(daily_values).set_index('date')

    # Calculate benchmark (buy and hold equal-weighted portfolio)
    benchmark_values = []
    benchmark_shares = {}
    benchmark_cash = initial_capital

    # Buy equal weight at start
    first_date = dates[0]
    first_day = signals.loc[(first_date, slice(None)), :]
    per_stock = benchmark_cash / len(tickers)

    for ticker in tickers:
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

    # Sharpe ratio (annualized)
    strategy_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0
    benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(252) if benchmark_returns.std() > 0 else 0

    # Max drawdown
    strategy_cummax = portfolio_values['total'].cummax()
    strategy_drawdown = (portfolio_values['total'] - strategy_cummax) / strategy_cummax
    max_drawdown = strategy_drawdown.min()

    benchmark_cummax = benchmark['total'].cummax()
    benchmark_drawdown = (benchmark['total'] - benchmark_cummax) / benchmark_cummax
    benchmark_max_dd = benchmark_drawdown.min()

    # Annualized returns
    total_days = (dates[-1] - dates[0]).days
    years = total_days / 365.25

    final_value = portfolio_values['total'].iloc[-1]
    strategy_annual_return = (final_value / initial_capital) ** (1/years) - 1

    benchmark_final = benchmark['total'].iloc[-1]
    benchmark_annual_return = (benchmark_final / initial_capital) ** (1/years) - 1

    # Win rate (daily)
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)

    # Print results
    print(f"\n" + "="*70)
    print("📊 BACKTEST RESULTS - REAL MARKET DATA")
    print("="*70)

    print(f"\n🗓️  Period: {dates[0].date()} to {dates[-1].date()} ({total_days} days)")
    print(f"💰 Initial Capital: ${initial_capital:,.0f}")

    print(f"\n🎯 STRATEGY PERFORMANCE:")
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
    print(f"   Excess Return:      {(strategy_annual_return - benchmark_annual_return)*100:+.2f}%")
    print(f"   Sharpe Improvement: {(strategy_sharpe - benchmark_sharpe):+.2f}")

    avg_positions = portfolio_values['num_positions'].mean()
    print(f"\n📊 TRADING STATISTICS:")
    print(f"   Avg # Positions:    {avg_positions:.1f}")
    print(f"   Max Positions:      {portfolio_values['num_positions'].max()}")

    # Signal statistics
    total_signals = (signals['signal'] == 1).sum()
    high_vol_days = signals['high_vol_regime'].sum()
    print(f"   Total Buy Signals:  {total_signals}")
    print(f"   High Vol Days:      {high_vol_days} ({high_vol_days/len(dates)*100:.1f}%)")

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
        'excess_return': strategy_annual_return - benchmark_annual_return,
        'total_signals': total_signals
    }

# ============================================================================
# RESEARCH INSIGHTS
# ============================================================================

def generate_research_insights(results: Dict, signals: pd.DataFrame):
    """Generate research insights and suggestions for next iteration"""

    print(f"\n" + "="*70)
    print("🔬 RESEARCH INSIGHTS & ITERATION SUGGESTIONS")
    print("="*70)

    strategy_return = results['strategy_annual_return']
    benchmark_return = results['benchmark_annual_return']
    excess = results['excess_return']
    sharpe = results['strategy_sharpe']

    print(f"\n💡 Key Findings:")

    if excess > 0.05:  # 5% outperformance
        print(f"   ✅ Strategy shows strong alpha generation ({excess*100:+.1f}%)")
    elif excess > 0:
        print(f"   ⚠️  Strategy shows modest alpha ({excess*100:+.1f}%)")
    else:
        print(f"   ❌ Strategy underperforms benchmark ({excess*100:+.1f}%)")

    if sharpe > 1.0:
        print(f"   ✅ Excellent risk-adjusted returns (Sharpe {sharpe:.2f})")
    elif sharpe > 0.5:
        print(f"   ⚠️  Acceptable risk-adjusted returns (Sharpe {sharpe:.2f})")
    else:
        print(f"   ❌ Poor risk-adjusted returns (Sharpe {sharpe:.2f})")

    if results['max_drawdown'] > -0.25:
        print(f"   ✅ Drawdown well-controlled ({results['max_drawdown']*100:.1f}%)")
    else:
        print(f"   ❌ Large drawdown ({results['max_drawdown']*100:.1f}%)")

    print(f"\n🔄 Suggestions for Next Iteration:")
    print(f"\n   1. Parameter Optimization:")
    print(f"      • Test different BB windows (15, 20, 25)")
    print(f"      • Adjust volatility threshold ({VOL_THRESHOLD}x)")
    print(f"      • Vary ATR stop multiplier (1.5x, 2.0x, 2.5x)")

    print(f"\n   2. Additional Filters:")
    print(f"      • Add volume confirmation (breakout on high volume)")
    print(f"      • Include fundamental screen (e.g., PE ratio, momentum)")
    print(f"      • Add sector rotation component")

    print(f"\n   3. Risk Management:")
    print(f"      • Implement position sizing based on volatility")
    print(f"      • Add portfolio-level stop loss")
    print(f"      • Dynamic allocation based on market regime")

    print(f"\n   4. Advanced Features:")
    print(f"      • Machine learning for regime detection")
    print(f"      • Option overlay for downside protection")
    print(f"      • Market neutral version (add short signals)")

    print(f"\n   5. Data Expansion:")
    print(f"      • Test on more tickers ({len(TICKERS)} currently)")
    print(f"      • Extend backtest period (2+ years currently)")
    print(f"      • Add international markets")

    print(f"\n📚 Next Steps:")
    print(f"   1. Choose 2-3 high-impact improvements")
    print(f"   2. Implement and backtest on same real data")
    print(f"   3. Compare performance vs current version")
    print(f"   4. Iterate until Sharpe > 1.5 and alpha > 10%")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "🔬"*35)
    print("  Volatility Breakout Strategy - Interactive R&D Session")
    print("  Using REAL Market Data from Yahoo Finance")
    print("🔬"*35)

    print(f"\n📅 Configuration:")
    print(f"   Date Range: {START_DATE} to {END_DATE}")
    print(f"   Tickers: {len(TICKERS)} stocks across 5 sectors")
    print(f"   Bollinger Bands: {BB_WINDOW}-day, {BB_STD}σ")
    print(f"   ATR Stop: {STOP_ATR_MULT}x {ATR_WINDOW}-day ATR")
    print(f"   Volatility Threshold: {VOL_THRESHOLD}x mean")

    try:
        # 1. Fetch real data
        data = fetch_real_market_data(TICKERS, START_DATE, END_DATE)

        # 2. Generate signals
        signals = generate_signals(data)

        # 3. Backtest
        results = backtest_strategy(signals)

        # 4. Research insights
        generate_research_insights(results, signals)

        print(f"\n" + "="*70)
        print("✅ Research Iteration Complete!")
        print("="*70)
        print(f"\nThis was Phase 1: Initial Implementation")
        print(f"Based on the results above, we can now iterate and improve!")

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
        print(f"   Access via: results['portfolio_values']")
        sys.exit(0)
    else:
        sys.exit(1)
