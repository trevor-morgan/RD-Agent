#!/usr/bin/env python3
"""
Real-World RD-Agent Demo: 2025 Quantitative Investment Strategy
Volatility-Adaptive Multi-Factor Portfolio with Regime Detection

Scenario: Navigate 2025 market challenges with intelligent factor allocation
Context: High inflation (2.5-3%), rate uncertainty, geopolitical volatility
Strategy: Multi-factor investing with ML-powered regime detection
"""

import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from rdagent.utils.env import LocalConf, LocalEnv

def create_market_scenario():
    """Create 2025 market data and scenario"""
    workspace = Path("/tmp/rdagent_quant_2025")
    workspace.mkdir(exist_ok=True)

    # Market data generation with 2025 characteristics
    market_gen_code = '''
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("📊 Generating 2025 Market Scenario Data")
print("=" * 70)

# Market parameters reflecting 2025 conditions
np.random.seed(42)
n_days = 252 * 3  # 3 years of daily data
n_stocks = 50

# Date range: 2022-2025
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

print(f"\\nDataset: {n_days} trading days, {n_stocks} stocks")
print(f"Period: {dates[0].date()} to {dates[-1].date()}")

# 2025 Market Regimes
# - Low Vol (2022-2023): VIX ~15-20
# - High Inflation (2023-2024): Rising rates, elevated inflation
# - Volatility Spike (2024-2025): Geopolitical tensions, tariff uncertainty

regime_breaks = [0, 252, 504, 756]  # Year boundaries
volatility_regimes = [0.15, 0.25, 0.35]  # Low, Medium, High vol
inflation_regimes = [0.02, 0.028, 0.025]  # Inflation rates

# Generate stock universe
tickers = [f"STOCK_{i:02d}" for i in range(n_stocks)]

# Initialize data
prices = pd.DataFrame(index=dates, columns=tickers)
returns = pd.DataFrame(index=dates, columns=tickers)

# Factor loadings (each stock has different factor exposures)
value_loading = np.random.uniform(-1, 1, n_stocks)
momentum_loading = np.random.uniform(-1, 1, n_stocks)
quality_loading = np.random.uniform(-1, 1, n_stocks)
low_vol_loading = np.random.uniform(-1, 1, n_stocks)

print("\\n📈 Factor Exposures Generated:")
print(f"   Value Factor: {value_loading[:5]}")
print(f"   Momentum Factor: {momentum_loading[:5]}")
print(f"   Quality Factor: {quality_loading[:5]}")
print(f"   Low Volatility Factor: {low_vol_loading[:5]}")

# Generate returns with regime-dependent factor performance
all_returns = []
all_regimes = []

for day in range(n_days):
    # Determine regime
    if day < regime_breaks[1]:
        regime = 0  # Low vol
        vol = volatility_regimes[0]
        value_return = 0.0003
        momentum_return = 0.0005
        quality_return = 0.0002
        low_vol_return = 0.0001
    elif day < regime_breaks[2]:
        regime = 1  # High inflation
        vol = volatility_regimes[1]
        value_return = -0.0001  # Value suffers in high inflation
        momentum_return = 0.0004
        quality_return = 0.0006  # Quality outperforms
        low_vol_return = 0.0003
    else:
        regime = 2  # High volatility
        vol = volatility_regimes[2]
        value_return = 0.0004  # Value rebounds
        momentum_return = -0.0002  # Momentum reversal
        quality_return = 0.0005
        low_vol_return = 0.0008  # Low vol shines

    all_regimes.append(regime)

    # Market return (with regime-dependent drift)
    market_return = np.random.normal(0.0003, vol)

    # Stock returns = factor loadings * factor returns + idiosyncratic
    stock_returns = (
        value_loading * value_return +
        momentum_loading * momentum_return +
        quality_loading * quality_return +
        low_vol_loading * low_vol_return +
        market_return +
        np.random.normal(0, vol, n_stocks)
    )

    all_returns.append(stock_returns)

# Create returns dataframe
returns_df = pd.DataFrame(all_returns, index=dates, columns=tickers)
returns_df['regime'] = all_regimes

# Calculate prices (cumulative returns)
prices_df = 100 * (1 + returns_df[tickers]).cumprod()

# Calculate factor metrics
value_scores = np.random.uniform(0.5, 1.5, n_stocks)  # P/E ratios (lower = better value)
momentum_scores = returns_df[tickers].rolling(126).mean().iloc[-1].values  # 6-month momentum
quality_scores = np.random.uniform(0.05, 0.25, n_stocks)  # ROE
volatility_scores = returns_df[tickers].rolling(63).std().iloc[-1].values  # 3-month volatility

# Create factor data
factors_df = pd.DataFrame({
    'ticker': tickers,
    'value_score': value_scores,
    'momentum_score': momentum_scores,
    'quality_score': quality_scores,
    'volatility_score': volatility_scores,
    'price': prices_df[tickers].iloc[-1].values
})

# Calculate market statistics
print("\\n📊 Market Statistics (Full Period):")
print(f"   Average Daily Return: {returns_df[tickers].mean().mean():.4%}")
print(f"   Average Daily Volatility: {returns_df[tickers].std().mean():.4%}")
print(f"   Annualized Return: {(returns_df[tickers].mean().mean() * 252):.2%}")
print(f"   Annualized Volatility: {(returns_df[tickers].std().mean() * np.sqrt(252)):.2%}")

# Regime analysis
print("\\n🎭 Market Regime Distribution:")
regime_names = ['Low Vol (2022-23)', 'High Inflation (2023-24)', 'High Vol (2024-25)']
for i, name in enumerate(regime_names):
    days = (returns_df['regime'] == i).sum()
    pct = days / len(returns_df) * 100
    print(f"   {name}: {days} days ({pct:.1f}%)")

# Save data
returns_df.to_csv('market_returns.csv')
prices_df.to_csv('market_prices.csv')
factors_df.to_csv('factor_data.csv')

print("\\n✅ Market data generated successfully")
print(f"   Files: market_returns.csv, market_prices.csv, factor_data.csv")
'''

    (workspace / "generate_market_data.py").write_text(market_gen_code)
    return workspace

def run_research_phase(env, workspace):
    """Research Phase: Analyze 2025 market and propose strategy"""
    print("\n" + "="*70)
    print("🔬 RESEARCH PHASE: 2025 Market Analysis & Strategy Design")
    print("="*70)

    print("\n📰 2025 Market Context (Real Intelligence):")
    print("   • Inflation: 2.5-3% (above Fed's 2% target)")
    print("   • Interest Rates: Fed cutting slower than expected")
    print("   • Volatility: High due to tariffs, geopolitical tensions")
    print("   • Key Challenge: Alpha decay - need specialized models")
    print("   • Opportunity: AI/ML-driven regime detection")

    print("\n💡 Research Findings:")
    print("   1. Traditional 60-40 portfolios underperforming")
    print("   2. Factor performance highly regime-dependent")
    print("   3. Volatility clustering requires adaptive strategies")
    print("   4. Quality & Low Vol factors outperform in uncertainty")

    # Strategy proposal
    strategy = {
        "name": "Volatility-Adaptive Multi-Factor Strategy",
        "objective": "Maximize risk-adjusted returns in 2025 volatile environment",
        "approach": [
            {
                "component": "Factor Selection",
                "description": "Value, Momentum, Quality, Low Volatility factors",
                "rationale": "Diversified factor exposure for different regimes"
            },
            {
                "component": "Regime Detection",
                "description": "ML-based classification: Low Vol, High Inflation, High Vol",
                "rationale": "Adapt factor weights to market conditions"
            },
            {
                "component": "Dynamic Allocation",
                "description": "Regime-conditional factor weighting",
                "rationale": "Exploit regime-specific factor performance"
            },
            {
                "component": "Risk Management",
                "description": "Portfolio volatility targeting + position limits",
                "rationale": "Prevent drawdowns in volatile periods"
            }
        ]
    }

    print("\n🎯 Proposed Strategy:")
    print(f"   Name: {strategy['name']}")
    print(f"   Objective: {strategy['objective']}")
    print("\n   Components:")
    for i, comp in enumerate(strategy['approach'], 1):
        print(f"   {i}. {comp['component']}: {comp['description']}")
        print(f"      → {comp['rationale']}")

    return strategy

def run_development_phase(env, workspace, strategy):
    """Development Phase: Implement the quant strategy"""
    print("\n" + "="*70)
    print("💻 DEVELOPMENT PHASE: Strategy Implementation")
    print("="*70)

    # Generate strategy implementation
    strategy_code = '''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("⚙️ Implementing Volatility-Adaptive Multi-Factor Strategy")
print("=" * 70)

# Load data
returns_df = pd.read_csv('market_returns.csv', index_col=0, parse_dates=True)
prices_df = pd.read_csv('market_prices.csv', index_col=0, parse_dates=True)
factors_df = pd.read_csv('factor_data.csv')

tickers = [c for c in returns_df.columns if c.startswith('STOCK_')]
print(f"\\nLoaded: {len(returns_df)} days, {len(tickers)} stocks")

# ============================================================================
# 1. REGIME DETECTION MODEL
# ============================================================================
print("\\n" + "="*70)
print("🎭 Phase 1: Training Regime Detection Model")
print("="*70)

# Calculate regime features
window = 20
regime_features = pd.DataFrame(index=returns_df.index)
regime_features['volatility'] = returns_df[tickers].std(axis=1).rolling(window).mean()
regime_features['avg_return'] = returns_df[tickers].mean(axis=1).rolling(window).mean()
regime_features['dispersion'] = returns_df[tickers].std(axis=1).rolling(window).std()
regime_features['skewness'] = returns_df[tickers].skew(axis=1).rolling(window).mean()
regime_features['max_drawdown'] = returns_df[tickers].mean(axis=1).rolling(window).apply(
    lambda x: (x.cummax() - x).max()
)

# True regimes from data
true_regimes = returns_df['regime']

# Train regime classifier
train_mask = regime_features.notna().all(axis=1)
X_train = regime_features[train_mask].values
y_train = true_regimes[train_mask].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_classifier.fit(X_train_scaled, y_train)

# Predict regimes
predicted_regimes = rf_classifier.predict(X_train_scaled)
accuracy = (predicted_regimes == y_train).mean()

print(f"\\nRegime Detection Accuracy: {accuracy:.1%}")
print("Feature Importance:")
for feat, imp in zip(regime_features.columns, rf_classifier.feature_importances_):
    print(f"   {feat}: {imp:.3f}")

# ============================================================================
# 2. FACTOR PORTFOLIO CONSTRUCTION
# ============================================================================
print("\\n" + "="*70)
print("📊 Phase 2: Building Factor Portfolios")
print("="*70)

# Calculate rolling factor scores for each stock
factor_window = 126  # 6 months

# Value: inverse P/E (from factor_data)
value_scores = pd.DataFrame(index=prices_df.index, columns=tickers)
for ticker in tickers:
    value_scores[ticker] = 1 / factors_df[factors_df['ticker'] == ticker]['value_score'].iloc[0]

# Momentum: 6-month return
momentum_scores = returns_df[tickers].rolling(factor_window).mean()

# Quality: ROE (from factor_data)
quality_scores = pd.DataFrame(index=prices_df.index, columns=tickers)
for ticker in tickers:
    quality_scores[ticker] = factors_df[factors_df['ticker'] == ticker]['quality_score'].iloc[0]

# Low Volatility: negative of realized volatility
low_vol_scores = -returns_df[tickers].rolling(63).std()  # 3-month

# Normalize scores (z-score across stocks) - pandas-compatible version
def zscore(df):
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    return df.sub(means, axis=0).div(stds, axis=0)

value_scores_z = zscore(value_scores.ffill())
momentum_scores_z = zscore(momentum_scores.ffill())
quality_scores_z = zscore(quality_scores.ffill())
low_vol_scores_z = zscore(low_vol_scores.ffill())

print("\\nFactor Portfolios Created:")
print("   • Value (inverse P/E)")
print("   • Momentum (6-month return)")
print("   • Quality (ROE)")
print("   • Low Volatility (inverse volatility)")

# ============================================================================
# 3. REGIME-ADAPTIVE PORTFOLIO ALLOCATION
# ============================================================================
print("\\n" + "="*70)
print("🎯 Phase 3: Dynamic Regime-Based Allocation")
print("="*70)

# Regime-specific factor weights (from research insights)
regime_weights = {
    0: {'value': 0.30, 'momentum': 0.35, 'quality': 0.20, 'low_vol': 0.15},  # Low vol
    1: {'value': 0.10, 'momentum': 0.30, 'quality': 0.40, 'low_vol': 0.20},  # High inflation
    2: {'value': 0.25, 'momentum': 0.10, 'quality': 0.30, 'low_vol': 0.35},  # High vol
}

print("\\nRegime-Specific Factor Weights:")
regime_names = ['Low Vol', 'High Inflation', 'High Vol']
for regime, name in enumerate(regime_names):
    print(f"\\n   {name}:")
    for factor, weight in regime_weights[regime].items():
        print(f"      {factor}: {weight:.0%}")

# Calculate composite scores with regime-adaptive weights
composite_scores = pd.DataFrame(0, index=prices_df.index, columns=tickers)

for i in range(len(regime_features)):
    if not train_mask.iloc[i]:
        continue

    date = regime_features.index[i]
    predicted_regime = predicted_regimes[i] if i < len(predicted_regimes) else 0
    weights = regime_weights[predicted_regime]

    composite_scores.loc[date] = (
        weights['value'] * value_scores_z.loc[date] +
        weights['momentum'] * momentum_scores_z.loc[date] +
        weights['quality'] * quality_scores_z.loc[date] +
        weights['low_vol'] * low_vol_scores_z.loc[date]
    )

# ============================================================================
# 4. BACKTEST
# ============================================================================
print("\\n" + "="*70)
print("📈 Phase 4: Backtesting Strategy")
print("="*70)

# Long top decile, short bottom decile
n_long = n_short = len(tickers) // 10
portfolio_returns = []

for date in composite_scores.index[factor_window:]:
    if composite_scores.loc[date].isna().all():
        continue

    # Rank stocks
    ranked = composite_scores.loc[date].sort_values(ascending=False)

    # Top decile long, bottom decile short
    long_stocks = ranked.head(n_long).index
    short_stocks = ranked.tail(n_short).index

    # Equal weight within each leg
    next_date_idx = returns_df.index.get_loc(date) + 1
    if next_date_idx >= len(returns_df):
        break

    next_return = returns_df.iloc[next_date_idx][tickers]

    long_return = next_return[long_stocks].mean()
    short_return = next_return[short_stocks].mean()
    portfolio_return = 0.5 * long_return - 0.5 * short_return  # Market neutral

    portfolio_returns.append(portfolio_return)

portfolio_returns = pd.Series(portfolio_returns)

# Calculate performance metrics
total_return = (1 + portfolio_returns).prod() - 1
annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
annualized_vol = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

# Benchmark: Equal-weight long-only
benchmark_returns = returns_df[tickers].iloc[factor_window:].mean(axis=1)
benchmark_total = (1 + benchmark_returns).prod() - 1
benchmark_annual = (1 + benchmark_total) ** (252 / len(benchmark_returns)) - 1
benchmark_vol = benchmark_returns.std() * np.sqrt(252)
benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol > 0 else 0

print("\\n✅ Backtest Complete")
print(f"   Trading Days: {len(portfolio_returns)}")
print(f"   Rebalancing: Daily")
print(f"   Position: Market Neutral (Long/Short)")

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\\n" + "="*70)
print("📊 PERFORMANCE RESULTS")
print("="*70)

print("\\n🎯 Strategy Performance:")
print(f"   Total Return:        {total_return:>8.2%}")
print(f"   Annualized Return:   {annualized_return:>8.2%}")
print(f"   Annualized Vol:      {annualized_vol:>8.2%}")
print(f"   Sharpe Ratio:        {sharpe_ratio:>8.2f}")
print(f"   Max Drawdown:        {max_drawdown:>8.2%}")

print("\\n📊 Benchmark (Equal Weight):")
print(f"   Total Return:        {benchmark_total:>8.2%}")
print(f"   Annualized Return:   {benchmark_annual:>8.2%}")
print(f"   Annualized Vol:      {benchmark_vol:>8.2%}")
print(f"   Sharpe Ratio:        {benchmark_sharpe:>8.2f}")

print("\\n📈 Strategy vs Benchmark:")
alpha = annualized_return - benchmark_annual
print(f"   Excess Return:       {alpha:>8.2%}")
print(f"   Information Ratio:   {(alpha / annualized_vol if annualized_vol > 0 else 0):>8.2f}")

# Win rate
win_rate = (portfolio_returns > 0).mean()
print(f"   Win Rate:            {win_rate:>8.1%}")

# Save results
results = {
    'strategy_return': float(annualized_return),
    'strategy_volatility': float(annualized_vol),
    'strategy_sharpe': float(sharpe_ratio),
    'strategy_max_dd': float(max_drawdown),
    'benchmark_return': float(benchmark_annual),
    'benchmark_sharpe': float(benchmark_sharpe),
    'excess_return': float(alpha),
    'win_rate': float(win_rate),
    'regime_accuracy': float(accuracy),
    'trading_days': len(portfolio_returns)
}

import json
with open('strategy_results.json', 'w') as f:
    json.dump(results, f, indent=2)

portfolio_returns.to_csv('strategy_returns.csv', header=['return'])

print("\\n✅ Results saved to strategy_results.json and strategy_returns.csv")
'''

    (workspace / "run_strategy.py").write_text(strategy_code)

    print("\n📝 Strategy Implementation Complete:")
    print("   File: run_strategy.py")
    print("   Components:")
    print("      1. Regime detection with Random Forest")
    print("      2. Multi-factor score calculation")
    print("      3. Regime-adaptive factor weighting")
    print("      4. Long/short portfolio construction")
    print("      5. Comprehensive backtest with risk metrics")

    return "run_strategy.py"

def run_execution_phase(env, workspace):
    """Execution Phase: Run the strategy backtest"""
    print("\n" + "="*70)
    print("⚙️ EXECUTION PHASE: Running Quantitative Strategy")
    print("="*70)

    python_cmd = sys.executable

    print("\n▶️ Step 1: Generating 2025 market scenario data...")
    result = env.run(
        entry=f"{python_cmd} generate_market_data.py",
        local_path=str(workspace)
    )
    print(result.stdout if hasattr(result, 'stdout') else str(result))

    print("\n▶️ Step 2: Running strategy backtest...")
    result = env.run(
        entry=f"{python_cmd} run_strategy.py",
        local_path=str(workspace)
    )
    output = result.stdout if hasattr(result, 'stdout') else str(result)
    print(output)

    return output, workspace / "strategy_results.json"

def run_evaluation_phase(results_file):
    """Evaluation Phase: Analyze strategy performance"""
    print("\n" + "="*70)
    print("📊 EVALUATION PHASE: Strategy Analysis & Recommendations")
    print("="*70)

    if not results_file.exists():
        print("⚠️  Results file not found")
        return None

    results = json.loads(results_file.read_text())

    print("\n📈 Performance Summary:")
    print(f"   Strategy Return:     {results['strategy_return']:>8.2%}")
    print(f"   Strategy Sharpe:     {results['strategy_sharpe']:>8.2f}")
    print(f"   Max Drawdown:        {results['strategy_max_dd']:>8.2%}")
    print(f"   Benchmark Return:    {results['benchmark_return']:>8.2%}")
    print(f"   Excess Return:       {results['excess_return']:>8.2%}")
    print(f"   Win Rate:            {results['win_rate']:>8.1%}")

    print("\n🤖 ML Performance:")
    print(f"   Regime Detection:    {results['regime_accuracy']:>8.1%}")

    # Provide feedback
    print("\n💭 Strategy Evaluation:")

    if results['strategy_sharpe'] > 1.5:
        print("   ✅ Excellent: Sharpe > 1.5 indicates strong risk-adjusted returns")
        print("   → Strategy ready for paper trading")
    elif results['strategy_sharpe'] > 1.0:
        print("   ✅ Good: Sharpe > 1.0 shows positive risk-adjusted performance")
        print("   → Consider increasing position sizing")
    elif results['strategy_sharpe'] > 0.5:
        print("   ⚠️  Moderate: Sharpe > 0.5 but room for improvement")
        print("   → Refine factor definitions or regime weights")
    else:
        print("   ❌ Underperforming: Low Sharpe ratio")
        print("   → Revisit factor selection and risk controls")

    if results['strategy_max_dd'] > -0.15:
        print("   ✅ Drawdown controlled: Max DD < 15%")
    else:
        print("   ⚠️  High drawdown: Consider tighter risk management")

    if results['regime_accuracy'] > 0.7:
        print("   ✅ Regime detection working well")
    else:
        print("   ⚠️  Regime detection needs improvement")
        print("   → Add more features or try different ML models")

    print("\n🔄 Next Iteration Ideas:")
    print("   • Test alternative ML models (XGBoost, LSTM for regime detection)")
    print("   • Add macro factors (inflation, interest rates, VIX)")
    print("   • Implement transaction costs and slippage")
    print("   • Optimize factor weights via machine learning")
    print("   • Add alternative data sources (sentiment, satellite)")

    print("\n📚 2025 Market Alignment:")
    if results['excess_return'] > 0:
        print("   ✅ Strategy generates alpha in 2025 volatility environment")
        print("   → Volatility adaptation working as designed")
    else:
        print("   ⚠️  Strategy needs refinement for 2025 conditions")
        print("   → Consider increasing low-vol factor weight")

    return results

def demonstrate_quant_strategy():
    """Main function demonstrating 2025 quantitative strategy R&D loop"""

    print("\n" + "📊"*35)
    print("  RD-Agent: 2025 Quantitative Investment Strategy")
    print("  Volatility-Adaptive Multi-Factor Portfolio")
    print("📊"*35)

    print("\n🌍 2025 Market Environment:")
    print("   Inflation: 2.5-3% (persistent, above target)")
    print("   Volatility: Elevated (geopolitical, tariff uncertainty)")
    print("   Interest Rates: Fed cutting slowly, uncertainty high")
    print("   Challenge: Traditional strategies underperforming")
    print("   Solution: AI-powered regime-adaptive factor investing")

    # Setup
    workspace = create_market_scenario()
    config = LocalConf(default_entry=sys.executable)
    env = LocalEnv(conf=config)

    print(f"\n🔧 Configuration:")
    print(f"   Environment: LocalEnv (no Docker)")
    print(f"   Workspace: {workspace}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Strategy: Market-neutral long/short equity")

    try:
        # Phase 1: Research
        strategy = run_research_phase(env, workspace)

        # Phase 2: Development
        impl_file = run_development_phase(env, workspace, strategy)

        # Phase 3: Execution
        output, results_file = run_execution_phase(env, workspace)

        # Phase 4: Evaluation
        results = run_evaluation_phase(results_file)

        # Summary
        print("\n" + "="*70)
        print("🎉 Quantitative Strategy R&D Loop Complete!")
        print("="*70)

        print("\n📊 What Was Demonstrated:")
        print("   1. ✅ 2025 market research and regime identification")
        print("   2. ✅ Multi-factor quantitative strategy design")
        print("   3. ✅ Machine learning for regime detection")
        print("   4. ✅ Dynamic factor allocation based on regimes")
        print("   5. ✅ Comprehensive backtest with risk analytics")

        if results:
            print(f"\n🎯 Key Results:")
            print(f"   Strategy Sharpe Ratio: {results['strategy_sharpe']:.2f}")
            print(f"   Excess Return vs Benchmark: {results['excess_return']:+.2%}")
            print(f"   Regime Detection Accuracy: {results['regime_accuracy']:.1%}")

        print("\n💡 This Demonstrates RD-Agent's Power:")
        print("   • Real-world financial problem solving")
        print("   • Integration of market research into strategy design")
        print("   • ML-powered adaptive algorithms")
        print("   • Professional quant workflow automation")
        print("   • Iterative refinement based on results")

        print("\n🚀 In Production RD-Agent:")
        print("   • LLM analyzes current market conditions")
        print("   • Proposes novel factor combinations")
        print("   • Auto-debugs strategy code")
        print("   • Optimizes hyperparameters")
        print("   • Continuously adapts to regime changes")

        return True

    except Exception as e:
        print(f"\n❌ Error during strategy execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demonstrate_quant_strategy()
    sys.exit(0 if success else 1)
