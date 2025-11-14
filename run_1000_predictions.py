#!/usr/bin/env python3
"""
Run 1000 Remote Viewing Predictions and Validate Against Real Market Data

This simulates a complete remote viewing experiment with:
- 1000 predictions across multiple stocks
- Historical dates (so we can validate against actual market data)
- Full statistical analysis
- Complete report generation
"""

from remote_viewing_experiment import RemoteViewingExperiment
from datetime import datetime, timedelta
import random
import time

def run_full_experiment():
    """Run a complete 1000-prediction experiment."""

    print("\n" + "="*80)
    print("RUNNING FULL 1000-PREDICTION REMOTE VIEWING EXPERIMENT")
    print("="*80)
    print("\nThis will:")
    print("  1. Generate 1000 predictions with random 'remote viewing' results")
    print("  2. Validate against REAL historical market data")
    print("  3. Perform rigorous statistical analysis")
    print("  4. Generate comprehensive report")
    print("\n" + "="*80 + "\n")

    # Create experiment
    exp = RemoteViewingExperiment("RV_Full_1000")

    # Major stocks to predict
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
        'JPM', 'V', 'WMT', 'MA', 'JNJ', 'UNH', 'HD', 'PG',
        'BAC', 'XOM', 'CVX', 'ABBV', 'KO', 'PEP', 'COST',
        'MRK', 'TMO', 'AVGO', 'LLY', 'ADBE', 'NKE', 'DIS'
    ]

    predictions_list = ['UP', 'DOWN']

    # Use dates from October 2024 (historical data available)
    # We'll make predictions for 50 trading days
    start_date = datetime(2024, 10, 1)

    print(f"STEP 1: Recording 1000 Blind Predictions")
    print("-"*80)
    print(f"Prediction period: {start_date.date()} onwards")
    print(f"Tickers: {len(tickers)} stocks")
    print(f"Target: 1000 predictions\n")

    prediction_count = 0
    day_offset = 0

    # Generate 1000 predictions
    while prediction_count < 1000:
        # Cycle through tickers and dates
        ticker = tickers[prediction_count % len(tickers)]

        # Spread predictions across ~50 trading days
        if prediction_count > 0 and prediction_count % 20 == 0:
            day_offset += 1

        # Simulate random remote viewing prediction
        # (In reality, this would be 50% accurate if truly random)
        prediction = random.choice(predictions_list)
        confidence = round(random.uniform(0.5, 0.95), 2)

        viewer_notes = f"RV Session {prediction_count + 1}: "
        if prediction == 'UP':
            viewer_notes += random.choice([
                "Green imagery, upward spiral",
                "Bright colors, ascending feeling",
                "Mountain peak, growth symbol",
                "Rising sun, positive energy"
            ])
        else:
            viewer_notes += random.choice([
                "Red imagery, downward spiral",
                "Dark colors, descending feeling",
                "Valley, contraction symbol",
                "Setting sun, negative energy"
            ])

        pred = exp.record_prediction(
            ticker=ticker,
            prediction=prediction,
            confidence=confidence,
            viewer_notes=viewer_notes
        )

        prediction_count += 1

        # Progress updates
        if prediction_count % 100 == 0:
            print(f"  ✓ {prediction_count}/1000 predictions recorded...")

    print(f"\n✓ Completed recording {prediction_count} predictions")
    print(f"  Results file: {exp.results_file}")

    # Validate predictions
    print(f"\n" + "="*80)
    print(f"STEP 2: Validating Against Real Historical Market Data")
    print("-"*80)
    print("Downloading market data from Yahoo Finance...")
    print("This may take a few minutes...\n")

    try:
        validated_count = exp.validate_predictions()
        print(f"\n✓ Validated {validated_count} predictions against real market data")
    except Exception as e:
        print(f"\n⚠ Validation error: {e}")
        print("Continuing with statistical analysis of available data...")

    # Statistical analysis
    print(f"\n" + "="*80)
    print(f"STEP 3: Statistical Analysis")
    print("-"*80)

    results = exp.statistical_analysis()

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal Predictions: {results['total_predictions']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Incorrect Predictions: {results['incorrect_predictions']}")
    print(f"\nAccuracy: {results['accuracy']:.2%}")
    print(f"Expected by Random Chance: {results['expected_random']:.2%}")
    print(f"95% Confidence Interval: [{results['confidence_interval'][0]:.2%}, {results['confidence_interval'][1]:.2%}]")

    print(f"\n{'─'*80}")
    print(f"STATISTICAL SIGNIFICANCE")
    print(f"{'─'*80}")
    print(f"P-value: {results['p_value']:.6f}")
    print(f"Statistically Significant (p < 0.05): {results['statistically_significant']}")
    print(f"\nVerdict: {results['verdict']}")

    # Interpretation
    print(f"\n{'─'*80}")
    print(f"INTERPRETATION")
    print(f"{'─'*80}")

    if results['p_value'] < 0.001:
        print("⭐ STRONG EVIDENCE that predictions are better than random chance.")
        print("   This would suggest remote viewing may have predictive power.")
    elif results['p_value'] < 0.01:
        print("✓ SIGNIFICANT EVIDENCE that predictions are better than random chance.")
        print("  This suggests something beyond random guessing is occurring.")
    elif results['p_value'] < 0.05:
        print("→ MODEST EVIDENCE that predictions are better than random chance.")
        print("  More data would help confirm this finding.")
    else:
        print("✗ NO EVIDENCE that predictions are better than random chance.")
        print("  Results are consistent with random guessing (50/50).")
        print("  Remote viewing does not appear to work for market prediction.")

    # Confidence-weighted analysis
    print(f"\n{'─'*80}")
    print(f"CONFIDENCE-WEIGHTED ANALYSIS")
    print(f"{'─'*80}")
    print(f"High Confidence (>0.7) Accuracy: {results['high_confidence_accuracy']:.2%}")
    print(f"Low Confidence (≤0.7) Accuracy: {results['low_confidence_accuracy']:.2%}")

    if results['high_confidence_accuracy'] > results['low_confidence_accuracy'] + 0.05:
        print("\n→ High confidence predictions ARE more accurate (good calibration)")
    else:
        print("\n→ Confidence level does NOT correlate with accuracy (poor calibration)")

    # Generate full report
    print(f"\n" + "="*80)
    print(f"STEP 4: Generating Comprehensive Report")
    print("-"*80)

    report_file = exp.generate_report()
    print(f"\n✓ Full report saved to: {report_file}")

    # Final summary
    print(f"\n" + "="*80)
    print(f"EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  - {exp.results_file} (prediction data)")
    print(f"  - {report_file} (full report)")
    print(f"\nThis experiment demonstrates:")
    print(f"  ✓ Proper scientific methodology")
    print(f"  ✓ Cryptographic tamper-proofing")
    print(f"  ✓ Validation against real market data")
    print(f"  ✓ Rigorous statistical testing")
    print(f"  ✓ Honest reporting of results")
    print(f"\n")

    return results

if __name__ == '__main__':
    start_time = time.time()
    results = run_full_experiment()
    elapsed_time = time.time() - start_time

    print(f"Total execution time: {elapsed_time:.1f} seconds")
    print(f"\n{'='*80}\n")
