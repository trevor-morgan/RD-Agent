"""
Predictable Trend Strength (PTS) - Implementation Starter Code

This module provides the foundational components for implementing
the PTS objective in RD-Agent's quantitative trading framework.

Usage:
    1. Add PTS factors to factor generation pipeline
    2. Integrate PTS loss function into model training
    3. Use PTS scores for strategy optimization

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from scipy.stats import spearmanr


# ============================================================================
# PART 1: PTS FACTOR CALCULATIONS
# ============================================================================

class PTSFactorCalculator:
    """
    Calculates Predictable Trend Strength (PTS) components.

    Components:
        1. Trend Clarity (TC): Inverse of residual volatility
        2. Signal-to-Noise Ratio (SNR): Signal strength vs noise
        3. Cross-Sectional Strength (CS): Prediction dispersion
        4. Temporal Stability (TS): Prediction consistency
    """

    def __init__(
        self,
        tc_window: int = 20,
        snr_window: int = 20,
        ts_window: int = 10,
    ):
        self.tc_window = tc_window
        self.snr_window = snr_window
        self.ts_window = ts_window

    def calculate_trend_clarity(
        self,
        returns: pd.Series,
        predictions: pd.Series,
    ) -> pd.Series:
        """
        Calculates Trend Clarity (TC) - how clean is the predicted trend?

        Method:
            1. Compute residuals = actual_returns - predicted_returns
            2. Calculate rolling residual volatility
            3. TC = 1 / (1 + residual_volatility)

        Higher TC = more predictable, lower noise
        """
        residuals = returns - predictions
        residual_vol = residuals.rolling(window=self.tc_window).std()

        # Avoid division by zero
        trend_clarity = 1.0 / (1.0 + residual_vol)

        return trend_clarity

    def calculate_signal_to_noise(
        self,
        predictions: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Calculates Signal-to-Noise Ratio (SNR).

        SNR = |predicted_return| / rolling_volatility

        High SNR = strong directional signal relative to noise
        """
        rolling_vol = returns.rolling(window=self.snr_window).std()

        # Avoid division by zero
        snr = np.abs(predictions) / (rolling_vol + 1e-8)

        return snr

    def calculate_cross_sectional_strength(
        self,
        predictions_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculates Cross-Sectional Strength (CS).

        Measures prediction dispersion across stocks at each time point.
        High dispersion = clear winners/losers (good for long-short).

        Args:
            predictions_df: DataFrame with stocks as columns, dates as index

        Returns:
            Series with CS score for each date
        """
        # Calculate cross-sectional standard deviation
        cs_std = predictions_df.std(axis=1)

        # Normalize by median to get relative strength
        cs_strength = cs_std / (cs_std.median() + 1e-8)

        return cs_strength

    def calculate_temporal_stability(
        self,
        predictions: pd.Series,
    ) -> pd.Series:
        """
        Calculates Temporal Stability (TS).

        Measures consistency of predictions over time.
        High TS = stable directional conviction (not flipping).

        Method:
            Correlation between recent predictions and past predictions
        """
        ts_scores = []

        for i in range(len(predictions)):
            if i < 2 * self.ts_window:
                ts_scores.append(np.nan)
                continue

            recent = predictions.iloc[i - self.ts_window:i]
            past = predictions.iloc[i - 2 * self.ts_window:i - self.ts_window]

            # Correlation between recent and past
            if len(recent) > 0 and len(past) > 0:
                corr = spearmanr(recent, past)[0]
                ts_scores.append(max(0, corr))  # Clip negative correlations
            else:
                ts_scores.append(np.nan)

        return pd.Series(ts_scores, index=predictions.index)

    def calculate_pts_score(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        predictions_df: Optional[pd.DataFrame] = None,
        alpha_weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
    ) -> pd.Series:
        """
        Calculates combined PTS score.

        PTS = α₁·TC + α₂·SNR + α₃·CS + α₄·TS

        Args:
            returns: Actual returns (for TC calculation)
            predictions: Predicted returns (for individual stock)
            predictions_df: All predictions (for CS calculation, optional)
            alpha_weights: Weights for (TC, SNR, CS, TS)

        Returns:
            Combined PTS score normalized to [0, 1]
        """
        α1, α2, α3, α4 = alpha_weights

        # Calculate components
        tc = self.calculate_trend_clarity(returns, predictions)
        snr = self.calculate_signal_to_noise(predictions, returns)
        ts = self.calculate_temporal_stability(predictions)

        # Cross-sectional (optional, if predictions_df provided)
        if predictions_df is not None:
            cs = self.calculate_cross_sectional_strength(predictions_df)
            # Broadcast to match predictions index
            cs = cs.reindex(predictions.index, method='ffill')
        else:
            cs = pd.Series(0.5, index=predictions.index)  # Neutral value
            α3 = 0  # Don't use CS if not available
            # Renormalize other weights
            total = α1 + α2 + α4
            α1, α2, α4 = α1/total, α2/total, α4/total

        # Normalize each component to [0, 1]
        tc_norm = self._normalize(tc)
        snr_norm = self._normalize(snr)
        cs_norm = self._normalize(cs)
        ts_norm = self._normalize(ts)

        # Combined PTS score
        pts_score = (
            α1 * tc_norm +
            α2 * snr_norm +
            α3 * cs_norm +
            α4 * ts_norm
        )

        return pts_score

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Normalize series to [0, 1] using min-max scaling."""
        min_val = series.min()
        max_val = series.max()

        if max_val - min_val < 1e-8:
            return pd.Series(0.5, index=series.index)

        normalized = (series - min_val) / (max_val - min_val)
        return normalized


# ============================================================================
# PART 2: PTS-AWARE NEURAL NETWORK MODEL
# ============================================================================

class PTSDualOutputModel(nn.Module):
    """
    Neural network that predicts both returns and PTS scores.

    Architecture:
        Input → Shared Backbone → {Return Head, PTS Head}

    Outputs:
        - predicted_return: Expected return
        - predicted_pts: Confidence/quality score [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # Shared backbone
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Return prediction head
        self.return_head = nn.Linear(prev_dim, 1)

        # PTS prediction head (outputs confidence score)
        self.pts_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # PTS in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            (predicted_return, predicted_pts)
        """
        features = self.backbone(x)
        pred_return = self.return_head(features)
        pred_pts = self.pts_head(features)

        return pred_return, pred_pts


# ============================================================================
# PART 3: PTS LOSS FUNCTION
# ============================================================================

class PTSLoss(nn.Module):
    """
    Custom loss function for PTS training.

    Components:
        1. Confidence-weighted MSE: Focus on high-PTS predictions
        2. Calibration loss: Ensure PTS matches realized accuracy
        3. Persistence loss: Reward stable predictions over time
        4. Sparsity loss: Encourage concentrated conviction
    """

    def __init__(
        self,
        lambda_calib: float = 0.1,
        lambda_persist: float = 0.05,
        lambda_sparse: float = 0.01,
    ):
        super().__init__()
        self.lambda_calib = lambda_calib
        self.lambda_persist = lambda_persist
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        pred_return: torch.Tensor,
        pred_pts: torch.Tensor,
        actual_return: torch.Tensor,
        prev_pred_return: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PTS loss.

        Args:
            pred_return: Predicted returns [batch_size, 1]
            pred_pts: Predicted PTS scores [batch_size, 1]
            actual_return: Actual returns [batch_size, 1]
            prev_pred_return: Previous predictions (for persistence)

        Returns:
            (total_loss, loss_components_dict)
        """
        # 1. Confidence-weighted MSE
        squared_errors = (pred_return - actual_return) ** 2
        accuracy_loss = (pred_pts * squared_errors).mean()

        # 2. Calibration loss: PTS should match realized accuracy
        # realized_accuracy = 1 / (1 + squared_error)
        realized_accuracy = 1.0 / (1.0 + squared_errors.detach())
        calibration_loss = F.mse_loss(pred_pts, realized_accuracy)

        # 3. Temporal persistence loss (optional)
        persistence_loss = torch.tensor(0.0, device=pred_return.device)
        if prev_pred_return is not None:
            # Penalize large changes in predictions
            # Use cosine similarity: 1 = same direction, 0 = orthogonal, -1 = opposite
            cos_sim = F.cosine_similarity(
                pred_return.flatten(),
                prev_pred_return.flatten(),
                dim=0
            )
            persistence_loss = 1.0 - cos_sim  # Convert to loss (lower is better)

        # 4. Sparsity loss: Encourage concentrated conviction
        # Use entropy: high entropy = uniform distribution (bad)
        # We want low entropy = concentrated on high-confidence predictions
        pts_flat = pred_pts.flatten()
        pts_probs = pts_flat / (pts_flat.sum() + 1e-8)
        entropy = -(pts_probs * torch.log(pts_probs + 1e-8)).sum()
        sparsity_loss = entropy

        # Total loss
        total_loss = (
            accuracy_loss
            + self.lambda_calib * calibration_loss
            + self.lambda_persist * persistence_loss
            + self.lambda_sparse * sparsity_loss
        )

        # Loss components for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'accuracy_loss': accuracy_loss.item(),
            'calibration_loss': calibration_loss.item(),
            'persistence_loss': persistence_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
        }

        return total_loss, loss_components


# ============================================================================
# PART 4: PTS-AWARE PORTFOLIO STRATEGY
# ============================================================================

class PTSWeightedPortfolio:
    """
    Portfolio construction using PTS for position sizing.

    Strategy:
        1. Filter stocks by PTS threshold (only trade high-confidence)
        2. Rank by predicted return
        3. Weight positions by PTS score
    """

    def __init__(
        self,
        topk: int = 50,
        pts_threshold: float = 0.6,
        pts_weighting: bool = True,
    ):
        self.topk = topk
        self.pts_threshold = pts_threshold
        self.pts_weighting = pts_weighting

    def generate_positions(
        self,
        predicted_returns: pd.Series,
        predicted_pts: pd.Series,
    ) -> pd.Series:
        """
        Generate portfolio positions.

        Args:
            predicted_returns: Predicted returns for all stocks
            predicted_pts: PTS scores for all stocks

        Returns:
            Target positions (weights) for each stock
        """
        # Filter by PTS threshold
        high_confidence_mask = predicted_pts >= self.pts_threshold
        filtered_returns = predicted_returns[high_confidence_mask]
        filtered_pts = predicted_pts[high_confidence_mask]

        if len(filtered_returns) == 0:
            # No high-confidence stocks, return empty
            return pd.Series(dtype=float)

        # Select top K by predicted return
        top_stocks = filtered_returns.nlargest(min(self.topk, len(filtered_returns)))

        if self.pts_weighting:
            # Weight by PTS (higher PTS = larger position)
            pts_weights = filtered_pts[top_stocks.index]
            normalized_weights = pts_weights / pts_weights.sum()
        else:
            # Equal weight
            normalized_weights = pd.Series(
                1.0 / len(top_stocks),
                index=top_stocks.index
            )

        return normalized_weights

    def backtest_metrics(
        self,
        returns_df: pd.DataFrame,
        predicted_returns_df: pd.DataFrame,
        predicted_pts_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Backtest the PTS-weighted strategy.

        Args:
            returns_df: Actual returns [dates x stocks]
            predicted_returns_df: Predicted returns [dates x stocks]
            predicted_pts_df: PTS scores [dates x stocks]

        Returns:
            Dictionary of performance metrics
        """
        portfolio_returns = []

        for date in returns_df.index:
            # Get predictions for this date
            pred_ret = predicted_returns_df.loc[date]
            pred_pts = predicted_pts_df.loc[date]

            # Generate positions
            positions = self.generate_positions(pred_ret, pred_pts)

            if len(positions) == 0:
                portfolio_returns.append(0.0)
                continue

            # Realized return
            actual_ret = returns_df.loc[date, positions.index]
            portfolio_ret = (positions * actual_ret).sum()
            portfolio_returns.append(portfolio_ret)

        # Calculate metrics
        portfolio_returns = pd.Series(portfolio_returns, index=returns_df.index)

        metrics = {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'annualized_return': portfolio_returns.mean() * 252,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'win_rate': (portfolio_returns > 0).sum() / len(portfolio_returns),
        }

        return metrics

    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


# ============================================================================
# PART 5: INTEGRATION WITH RD-AGENT
# ============================================================================

class PTSExperimentConfig:
    """
    Configuration for PTS experiments in RD-Agent.

    This provides a template for integrating PTS into the
    existing RD-Agent factor/model experimentation framework.
    """

    @staticmethod
    def get_pts_factor_task_prompt() -> str:
        """
        Prompt for generating PTS-related factors.

        This can be used with RD-Agent's factor generation pipeline.
        """
        return """
        Generate factors that measure the PREDICTABILITY and CLARITY of price trends.

        Focus on factors that capture:
        1. Trend Clarity: How clean/noisy is the recent price movement?
           - Residual volatility after removing trends
           - Autocorrelation structure
           - Regime stability

        2. Signal Strength: How strong is the directional signal?
           - Signal-to-noise ratio
           - Magnitude relative to volatility
           - Momentum persistence

        3. Cross-Sectional Dispersion: How differentiated are stocks?
           - Standard deviation of returns
           - Rank dispersion
           - Winner-loser separation

        4. Temporal Stability: How consistent are predictions?
           - Prediction autocorrelation
           - Direction flipping frequency
           - Conviction stability

        Use qlib's factor syntax with Alpha158 data.
        """

    @staticmethod
    def get_pts_model_task_prompt() -> str:
        """
        Prompt for generating PTS-aware models.
        """
        return """
        Design a neural network model that predicts BOTH:
        1. Expected return (primary prediction)
        2. Predictability score (confidence/quality of prediction)

        Requirements:
        - Shared backbone for feature extraction
        - Dual output heads (return + PTS)
        - PTS head should output sigmoid [0, 1]
        - Loss function should combine:
          * Confidence-weighted MSE
          * Calibration loss (PTS matches realized accuracy)
          * Temporal persistence (stable predictions)

        The model should learn WHEN its predictions are reliable,
        not just WHAT the predictions are.
        """

    @staticmethod
    def get_pts_evaluation_metrics() -> Dict[str, str]:
        """
        Additional evaluation metrics for PTS experiments.
        """
        return {
            'avg_pts': 'Average PTS score across all predictions',
            'pts_calibration': 'Correlation between predicted PTS and realized accuracy',
            'high_pts_ic': 'IC on predictions with PTS > 0.7',
            'low_pts_ic': 'IC on predictions with PTS < 0.4',
            'pts_sharpe_improvement': 'Sharpe ratio improvement vs baseline',
            'high_pts_win_rate': 'Win rate on high-PTS trades',
        }


# ============================================================================
# PART 6: EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example of how to use the PTS components.
    """
    # Example data (replace with real data)
    np.random.seed(42)
    n_days = 1000
    n_stocks = 100

    dates = pd.date_range('2020-01-01', periods=n_days)
    stocks = [f'stock_{i}' for i in range(n_stocks)]

    # Simulate returns and predictions
    returns_df = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.02,
        index=dates,
        columns=stocks
    )

    predictions_df = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.015,
        index=dates,
        columns=stocks
    )

    # 1. Calculate PTS factors
    print("=" * 60)
    print("1. Calculating PTS Factors")
    print("=" * 60)

    pts_calc = PTSFactorCalculator()

    # For first stock
    stock_returns = returns_df.iloc[:, 0]
    stock_predictions = predictions_df.iloc[:, 0]

    pts_score = pts_calc.calculate_pts_score(
        returns=stock_returns,
        predictions=stock_predictions,
        predictions_df=predictions_df,
    )

    print(f"PTS Score (first stock):")
    print(pts_score.describe())
    print()

    # 2. Train PTS model (simplified example)
    print("=" * 60)
    print("2. PTS Model Training")
    print("=" * 60)

    # Create dummy data
    X_train = torch.randn(1000, 20)
    y_train = torch.randn(1000, 1)

    model = PTSDualOutputModel(input_dim=20)
    loss_fn = PTSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop (just 1 epoch for demo)
    model.train()
    pred_return, pred_pts = model(X_train)
    loss, loss_components = loss_fn(pred_return, pred_pts, y_train)

    print("Loss components:")
    for k, v in loss_components.items():
        print(f"  {k}: {v:.6f}")
    print()

    # 3. Generate portfolio with PTS
    print("=" * 60)
    print("3. PTS-Weighted Portfolio")
    print("=" * 60)

    # Simulate PTS scores for all stocks
    pts_df = pd.DataFrame(
        np.random.uniform(0.3, 0.9, size=(n_days, n_stocks)),
        index=dates,
        columns=stocks
    )

    portfolio = PTSWeightedPortfolio(topk=50, pts_threshold=0.6)

    metrics = portfolio.backtest_metrics(
        returns_df=returns_df,
        predicted_returns_df=predictions_df,
        predicted_pts_df=pts_df,
    )

    print("Portfolio metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Predictable Trend Strength (PTS) - Implementation Starter")
    print("="*60 + "\n")

    example_usage()

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Integrate PTS factors into RD-Agent factor generation")
    print("2. Train baseline model with PTS as features")
    print("3. Implement dual-output model with PTS loss")
    print("4. Backtest PTS-weighted strategy vs baseline")
    print("5. Measure Sharpe improvement and win rate")
    print("="*60 + "\n")
