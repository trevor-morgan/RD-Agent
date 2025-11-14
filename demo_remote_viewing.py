#!/usr/bin/env python3
"""
Demonstration of Remote Viewing Experiment Framework

This script shows the complete workflow:
1. Recording blind predictions
2. Validating against real market data
3. Statistical analysis
4. Report generation
"""

from remote_viewing_experiment import RemoteViewingExperiment
from datetime import datetime, timedelta
import random

def demo_experiment():
    """Run a demonstration of the remote viewing experiment."""

    print("\n" + "="*80)
    print("REMOTE VIEWING EXPERIMENT DEMONSTRATION")
    print("="*80)

    # Create experiment
    exp = RemoteViewingExperiment("RV_Demo_2025")
    print(f"\n✓ Created experiment: {exp.experiment_name}")
    print(f"  Results file: {exp.results_file}")

    # Simulate making 10 sample predictions
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    predictions_list = ['UP', 'DOWN']

    print(f"\n" + "-"*80)
    print("STEP 1: Recording Blind Predictions")
    print("-"*80)

    for i, ticker in enumerate(tickers, 1):
        # Simulate random predictions (in real use, these would be from remote viewing sessions)
        prediction = random.choice(predictions_list)
        confidence = round(random.uniform(0.6, 0.95), 2)

        viewer_notes = f"Session {i}: Saw {'green/upward imagery' if prediction == 'UP' else 'red/downward imagery'}"

        pred = exp.record_prediction(
            ticker=ticker,
            prediction=prediction,
            confidence=confidence,
            viewer_notes=viewer_notes
        )

        print(f"  {i}. {ticker:6s} → {prediction:4s} (confidence: {confidence:.2f}) [Session: {pred.session_id}]")

    print(f"\n✓ Recorded {len(tickers)} blind predictions")

    # Show what happens next
    print(f"\n" + "-"*80)
    print("STEP 2: Validation (would download real market data)")
    print("-"*80)
    print("  Note: This requires historical data for the prediction dates.")
    print("  For demo purposes, we're showing the process flow.")
    print("\n  To validate in real use:")
    print("    exp.validate_predictions()")

    print(f"\n" + "-"*80)
    print("STEP 3: Statistical Analysis (after validation)")
    print("-"*80)
    print("  After validating predictions against actual market data:")
    print("    results = exp.statistical_analysis()")
    print("\n  Results would include:")
    print("    - Total predictions")
    print("    - Accuracy percentage")
    print("    - P-value (binomial test)")
    print("    - 95% confidence interval")
    print("    - Statistical significance verdict")

    print(f"\n" + "-"*80)
    print("STEP 4: Generate Report")
    print("-"*80)
    print("  Generate comprehensive report:")
    print("    exp.generate_report()")
    print("\n  Report includes:")
    print("    - Full prediction log with hashes")
    print("    - Validation results")
    print("    - Statistical analysis")
    print("    - Verdict on remote viewing effectiveness")

    # Show example of what a completed experiment would look like
    print(f"\n" + "="*80)
    print("EXAMPLE: What Statistical Results Look Like")
    print("="*80)

    example_results = {
        'total_predictions': 100,
        'correct_predictions': 62,
        'accuracy': 0.62,
        'p_value': 0.0023,
        'confidence_interval': (0.52, 0.72),
        'expected_random': 0.50,
        'statistically_significant': True,
        'verdict': 'SIGNIFICANT EVIDENCE'
    }

    print(f"\n  Total Predictions: {example_results['total_predictions']}")
    print(f"  Correct: {example_results['correct_predictions']}")
    print(f"  Accuracy: {example_results['accuracy']:.1%}")
    print(f"  Expected by Random Chance: {example_results['expected_random']:.1%}")
    print(f"  95% Confidence Interval: [{example_results['confidence_interval'][0]:.1%}, {example_results['confidence_interval'][1]:.1%}]")
    print(f"  P-value: {example_results['p_value']:.4f}")
    print(f"  Statistically Significant: {example_results['statistically_significant']}")
    print(f"  Verdict: {example_results['verdict']}")

    print(f"\n  Interpretation:")
    print(f"    With 62% accuracy (p=0.0023), there is SIGNIFICANT EVIDENCE")
    print(f"    that the predictions are better than random chance (50%).")

    # Show prediction file location
    print(f"\n" + "="*80)
    print("FILES CREATED")
    print("="*80)
    print(f"\n  Predictions saved to: {exp.results_file}")
    print(f"  Each prediction includes:")
    print(f"    - SHA-256 hash (tamper-proof)")
    print(f"    - Timestamp")
    print(f"    - Ticker, prediction, confidence")
    print(f"    - Viewer notes")

    print(f"\n" + "="*80)
    print("READY FOR REAL EXPERIMENT")
    print("="*80)
    print(f"\n  The framework is ready to use for real remote viewing testing.")
    print(f"  Minimum 100 predictions needed for statistical validity.")
    print(f"\n  Remember:")
    print(f"    ✓ Make predictions BEFORE market open (blind)")
    print(f"    ✓ Record ALL predictions (not just successful ones)")
    print(f"    ✓ No access to charts/news during remote viewing")
    print(f"    ✓ Accept results even if negative")
    print(f"\n")

if __name__ == '__main__':
    demo_experiment()
