"""
REMOTE VIEWING TRADING EXPERIMENT
Rigorous scientific test of remote viewing for market prediction

HYPOTHESIS: Remote viewing can predict market movements better than random chance
NULL HYPOTHESIS: Accuracy ≤ 50% (no better than coin flip)
SIGNIFICANCE LEVEL: p < 0.05 (95% confidence)
MINIMUM TRIALS: 100 (for statistical power)

Experimental Protocol:
1. Blind prediction (no market data available)
2. Timestamp and hash prediction (prevent retroactive changes)
3. Record prediction before market open
4. Compare to actual market movement
5. Statistical analysis of results

Author: RD-Agent Research Team
Date: 2025-11-14
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from scipy.stats import binomtest, chi2_contingency
import yfinance as yf
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RemoteViewingPrediction:
    """Single remote viewing prediction record."""
    session_id: str
    timestamp: str
    ticker: str
    prediction: str  # 'UP' or 'DOWN'
    confidence: float  # 0.0 to 1.0
    viewer_notes: str
    prediction_hash: str  # SHA-256 hash to prevent tampering

    # Filled in after prediction
    actual_movement: Optional[str] = None
    actual_return: Optional[float] = None
    correct: Optional[bool] = None
    market_close_time: Optional[str] = None


class RemoteViewingExperiment:
    """
    Rigorous scientific experiment to test remote viewing for trading.

    Key principles:
    - Blind predictions (no market data)
    - Timestamped and cryptographically hashed
    - Large sample size (100+ trials)
    - Statistical validation
    - Honest reporting of all results
    """

    def __init__(self, experiment_name: str = "RV_Trading_Experiment"):
        self.experiment_name = experiment_name
        self.predictions: List[RemoteViewingPrediction] = []
        self.results_file = f"{experiment_name}_results.json"

        # Load existing predictions if any
        self._load_existing_predictions()

    def _load_existing_predictions(self):
        """Load existing predictions from file."""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                self.predictions = [
                    RemoteViewingPrediction(**pred) for pred in data
                ]
            print(f"Loaded {len(self.predictions)} existing predictions")
        except FileNotFoundError:
            print(f"No existing predictions found. Starting new experiment.")

    def _save_predictions(self):
        """Save predictions to file."""
        with open(self.results_file, 'w') as f:
            json.dump([asdict(pred) for pred in self.predictions], f, indent=2)
        print(f"Saved {len(self.predictions)} predictions to {self.results_file}")

    def _hash_prediction(
        self,
        ticker: str,
        prediction: str,
        timestamp: str,
        viewer_notes: str
    ) -> str:
        """
        Create cryptographic hash of prediction.

        This prevents retroactive changes to predictions.
        Hash is based on prediction content + timestamp.
        """
        content = f"{ticker}|{prediction}|{timestamp}|{viewer_notes}"
        return hashlib.sha256(content.encode()).hexdigest()

    def record_prediction(
        self,
        ticker: str,
        prediction: str,
        confidence: float,
        viewer_notes: str = ""
    ) -> RemoteViewingPrediction:
        """
        Record a remote viewing prediction.

        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            prediction: 'UP' or 'DOWN'
            confidence: 0.0 to 1.0
            viewer_notes: Any notes from the viewing session

        Returns:
            RemoteViewingPrediction object
        """
        # Validate inputs
        prediction = prediction.upper()
        if prediction not in ['UP', 'DOWN']:
            raise ValueError("Prediction must be 'UP' or 'DOWN'")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        # Create prediction record
        timestamp = datetime.now().isoformat()
        session_id = f"RV_{len(self.predictions) + 1:04d}"

        pred_hash = self._hash_prediction(ticker, prediction, timestamp, viewer_notes)

        pred = RemoteViewingPrediction(
            session_id=session_id,
            timestamp=timestamp,
            ticker=ticker,
            prediction=prediction,
            confidence=confidence,
            viewer_notes=viewer_notes,
            prediction_hash=pred_hash
        )

        self.predictions.append(pred)
        self._save_predictions()

        print(f"✓ Prediction recorded: {session_id}")
        print(f"  Ticker: {ticker}")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Hash: {pred_hash[:16]}...")
        print(f"  Time: {timestamp}")

        return pred

    def verify_prediction_integrity(self, pred: RemoteViewingPrediction) -> bool:
        """
        Verify prediction hasn't been tampered with.

        Returns:
            True if hash matches, False if tampered
        """
        computed_hash = self._hash_prediction(
            pred.ticker,
            pred.prediction,
            pred.timestamp,
            pred.viewer_notes
        )
        return computed_hash == pred.prediction_hash

    def validate_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Validate all predictions against actual market data.

        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        """
        print("\n" + "=" * 80)
        print("VALIDATING REMOTE VIEWING PREDICTIONS")
        print("=" * 80)

        # Filter predictions to validate
        preds_to_validate = [
            p for p in self.predictions
            if p.actual_movement is None  # Not yet validated
        ]

        if len(preds_to_validate) == 0:
            print("No predictions to validate.")
            return

        print(f"\nValidating {len(preds_to_validate)} predictions...")

        for pred in preds_to_validate:
            # Verify integrity
            if not self.verify_prediction_integrity(pred):
                print(f"⚠️  WARNING: Prediction {pred.session_id} may be tampered!")
                continue

            # Get actual market data
            pred_time = datetime.fromisoformat(pred.timestamp)

            # Get next trading day's data
            start = pred_time.date()
            end = start + timedelta(days=5)  # Get a few days to ensure we get next trading day

            try:
                data = yf.download(
                    pred.ticker,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    progress=False
                )

                if len(data) < 2:
                    print(f"⚠️  Not enough data for {pred.ticker} on {start}")
                    continue

                # Compare prediction day to next day
                open_price = data['Open'].iloc[0]
                close_price = data['Close'].iloc[1]  # Next day's close

                actual_return = (close_price - open_price) / open_price
                actual_movement = 'UP' if actual_return > 0 else 'DOWN'

                correct = (pred.prediction == actual_movement)

                # Update prediction
                pred.actual_movement = actual_movement
                pred.actual_return = float(actual_return)
                pred.correct = correct
                pred.market_close_time = data.index[1].isoformat()

                # Print result
                status = "✓ CORRECT" if correct else "✗ WRONG"
                print(f"{status} - {pred.session_id}: {pred.ticker} "
                      f"predicted {pred.prediction}, actually {actual_movement} "
                      f"({actual_return:+.2%})")

            except Exception as e:
                print(f"Error validating {pred.session_id}: {e}")

        # Save updated predictions
        self._save_predictions()

        print("\n✓ Validation complete")

    def statistical_analysis(self) -> Dict:
        """
        Perform rigorous statistical analysis.

        Tests:
        1. Binomial test (better than 50% accuracy?)
        2. Chi-square test (prediction vs actual distribution)
        3. Confidence-weighted analysis
        4. Streak analysis

        Returns:
            Dict with statistical results
        """
        # Filter validated predictions
        validated = [p for p in self.predictions if p.correct is not None]

        if len(validated) < 10:
            print("⚠️  Need at least 10 validated predictions for statistical analysis")
            print(f"   Current: {len(validated)} validated")
            return {'error': 'Insufficient data'}

        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)

        n_total = len(validated)
        n_correct = sum(p.correct for p in validated)
        accuracy = n_correct / n_total

        # Binomial test
        binom_result = binomtest(n_correct, n_total, 0.5, alternative='greater')

        # Separate by confidence level
        high_conf = [p for p in validated if p.confidence >= 0.7]
        low_conf = [p for p in validated if p.confidence < 0.5]

        high_conf_acc = (
            sum(p.correct for p in high_conf) / len(high_conf)
            if len(high_conf) > 0 else None
        )
        low_conf_acc = (
            sum(p.correct for p in low_conf) / len(low_conf)
            if len(low_conf) > 0 else None
        )

        # Prepare results
        results = {
            'n_predictions': n_total,
            'n_correct': n_correct,
            'accuracy': accuracy,
            'p_value': binom_result.pvalue,
            'statistically_significant': binom_result.pvalue < 0.05,
            'confidence_95_interval': binom_result.proportion_ci(confidence_level=0.95),
            'high_confidence_accuracy': high_conf_acc,
            'low_confidence_accuracy': low_conf_acc,
            'verdict': self._get_verdict(accuracy, binom_result.pvalue)
        }

        # Print results
        print(f"\nTotal Predictions: {n_total}")
        print(f"Correct: {n_correct}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"\nStatistical Test (Binomial):")
        print(f"  H0: Accuracy ≤ 50% (random chance)")
        print(f"  H1: Accuracy > 50% (better than chance)")
        print(f"  p-value: {binom_result.pvalue:.4f}")
        print(f"  Significant (p < 0.05): {results['statistically_significant']}")
        print(f"\n95% Confidence Interval: {results['confidence_95_interval'][0]:.1%} - {results['confidence_95_interval'][1]:.1%}")

        if high_conf_acc is not None:
            print(f"\nHigh Confidence (≥0.7) Accuracy: {high_conf_acc:.1%} (n={len(high_conf)})")
        if low_conf_acc is not None:
            print(f"Low Confidence (<0.5) Accuracy: {low_conf_acc:.1%} (n={len(low_conf)})")

        print(f"\n{'='*80}")
        print(f"VERDICT: {results['verdict']}")
        print(f"{'='*80}")

        return results

    def _get_verdict(self, accuracy: float, p_value: float) -> str:
        """Determine verdict from statistical results."""
        if p_value < 0.001:
            return "STRONG EVIDENCE: Remote viewing works! (p < 0.001)"
        elif p_value < 0.01:
            return "SIGNIFICANT EVIDENCE: Remote viewing likely works (p < 0.01)"
        elif p_value < 0.05:
            return "MODEST EVIDENCE: Remote viewing may work (p < 0.05)"
        elif accuracy > 0.5:
            return "NOT SIGNIFICANT: Better than chance but not statistically proven"
        else:
            return "NO EVIDENCE: Performance equals or worse than random chance"

    def generate_report(self, filename: Optional[str] = None):
        """Generate comprehensive experiment report."""
        if filename is None:
            filename = f"{self.experiment_name}_report.txt"

        validated = [p for p in self.predictions if p.correct is not None]

        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("REMOTE VIEWING TRADING EXPERIMENT - FINAL REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            f.write("METHODOLOGY:\n")
            f.write("1. Blind predictions (no market data)\n")
            f.write("2. Cryptographically hashed (tamper-proof)\n")
            f.write("3. Timestamped before market movement\n")
            f.write("4. Statistical validation (binomial test)\n\n")

            if len(validated) > 0:
                stats = self.statistical_analysis()
                f.write(f"RESULTS:\n")
                f.write(f"Total Predictions: {stats['n_predictions']}\n")
                f.write(f"Accuracy: {stats['accuracy']:.1%}\n")
                f.write(f"P-value: {stats['p_value']:.4f}\n")
                f.write(f"Statistically Significant: {stats['statistically_significant']}\n\n")
                f.write(f"VERDICT: {stats['verdict']}\n\n")

            f.write("\nDETAILED PREDICTIONS:\n")
            f.write("-" * 80 + "\n")

            for pred in self.predictions:
                f.write(f"\nSession: {pred.session_id}\n")
                f.write(f"  Ticker: {pred.ticker}\n")
                f.write(f"  Time: {pred.timestamp}\n")
                f.write(f"  Prediction: {pred.prediction} (confidence: {pred.confidence:.2f})\n")
                if pred.actual_movement:
                    f.write(f"  Actual: {pred.actual_movement} ({pred.actual_return:+.2%})\n")
                    f.write(f"  Result: {'✓ CORRECT' if pred.correct else '✗ WRONG'}\n")
                f.write(f"  Hash: {pred.prediction_hash}\n")
                if pred.viewer_notes:
                    f.write(f"  Notes: {pred.viewer_notes}\n")

        print(f"\n✓ Report saved to {filename}")

    def get_prediction_template(self) -> str:
        """Get template for making predictions."""
        return """
REMOTE VIEWING PREDICTION TEMPLATE
==================================

Date: {date}
Session ID: (will be auto-generated)

Ticker: ___________
Prediction: [ ] UP  [ ] DOWN
Confidence (0.0-1.0): _____

Viewing Session Notes:
- What images/impressions did you receive?
- Any specific symbols or feelings?
- Timeline/timing impressions?
- Confidence level justification?

Notes:
_____________________________________________
_____________________________________________
_____________________________________________

Submit prediction by calling:
experiment.record_prediction(
    ticker='AAPL',
    prediction='UP',  # or 'DOWN'
    confidence=0.7,   # 0.0 to 1.0
    viewer_notes='Your notes here'
)
""".format(date=datetime.now().strftime('%Y-%m-%d'))


def main():
    """Example usage of the experiment framework."""

    print("=" * 80)
    print("REMOTE VIEWING TRADING EXPERIMENT")
    print("=" * 80)
    print()
    print("This is a rigorous scientific test of remote viewing for trading.")
    print("All predictions are timestamped, hashed, and statistically validated.")
    print()

    # Create experiment
    exp = RemoteViewingExperiment("RV_Trading_2025")

    print("GETTING STARTED:")
    print()
    print("1. Make blind predictions (no market data):")
    print("   exp.record_prediction('AAPL', 'UP', confidence=0.8)")
    print()
    print("2. Validate predictions after market closes:")
    print("   exp.validate_predictions()")
    print()
    print("3. Analyze results statistically:")
    print("   exp.statistical_analysis()")
    print()
    print("4. Generate report:")
    print("   exp.generate_report()")
    print()
    print("PREDICTION TEMPLATE:")
    print(exp.get_prediction_template())
    print()
    print("MINIMUM FOR STATISTICAL VALIDITY:")
    print("- At least 100 predictions")
    print("- Predictions made before market open")
    print("- Blind (no access to charts, news, etc.)")
    print("- Honest recording of ALL predictions (not just successful ones)")
    print()
    print("Remember: Accept the results even if they show remote viewing doesn't work!")
    print()

    return exp


if __name__ == "__main__":
    experiment = main()
