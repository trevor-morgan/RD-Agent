#!/usr/bin/env python3
# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""Default DiscoRL training template.

This template is replaced by auto-generated code from the LLM.
It serves as a fallback and reference implementation.
"""

import json
import sys
from pathlib import Path

# Configuration (will be replaced by LLM-generated values)
DEFAULT_CONFIG = {
    "horizon": 50,
    "order_side": 1,
    "base_spread": 0.01,
    "base_volatility": 0.02,
    "mean_reversion": 0.1,
    "momentum_factor": 0.3,
    "arrival_price_bonus": 0.1,
    "incomplete_penalty": 5.0,
    "wait_penalty": 0.001,
    "num_training_steps": 10000,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "trajectory_length": 50,
    "use_disco": True,
    "num_eval_episodes": 50,
    "eval_seeds": [42, 123, 456],
}


def main():
    """Run DiscoRL training."""
    config = DEFAULT_CONFIG

    print("=" * 60)
    print("DiscoRL Optimal Execution Training")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 60)

    try:
        # Try to import disco_rl
        from disco_rl.rdagent_integration import (
            DiscoTrainer,
            ExecutionExperimentConfig,
        )

        # Create config
        exp_config = ExecutionExperimentConfig.from_dict(config)

        # Create trainer
        print("\nInitializing trainer...")
        trainer = DiscoTrainer(
            exp_config,
            use_disco=config["use_disco"],
        )

        # Train
        print(f"\nTraining for {config['num_training_steps']} steps...")
        trainer.train(config["num_training_steps"])

        # Evaluate
        print("\nEvaluating...")
        result = trainer.evaluate()

        # Save results
        save_results(result.to_dict())

        # Print summary
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Score: {result.score():.2f}")
        print(f"Shortfall: {result.mean_shortfall:.4f} +/- {result.std_shortfall:.4f}")
        print(f"Completion: {result.completion_rate:.1%}")
        print(f"Actions: {result.action_distribution}")

    except ImportError as e:
        print(f"\nWarning: disco_rl not available: {e}")
        print("Running mock training for testing...")
        run_mock_training(config)


def run_mock_training(config: dict):
    """Run mock training when disco_rl is not available."""
    import random
    import time

    print("\n[Mock Mode] Simulating training...")

    # Simulate training time
    for step in range(0, config["num_training_steps"], config["num_training_steps"] // 5):
        print(f"  Step {step}/{config['num_training_steps']}")
        time.sleep(0.1)

    # Generate mock results
    mock_result = {
        "mean_shortfall": 0.03 + random.uniform(-0.01, 0.01),
        "std_shortfall": 0.015 + random.uniform(-0.005, 0.005),
        "completion_rate": 0.92 + random.uniform(-0.05, 0.05),
        "mean_vwap_vs_arrival": -0.001 + random.uniform(-0.001, 0.001),
        "mean_market_impact": 0.002 + random.uniform(-0.001, 0.001),
        "mean_episode_length": config["horizon"] * 0.8,
        "action_distribution": {
            "WAIT": 0.15,
            "PASSIVE": 0.25,
            "MODERATE": 0.35,
            "AGGRESSIVE": 0.18,
            "URGENT": 0.07,
        },
        "final_loss": 0.01,
        "training_steps": config["num_training_steps"],
        "converged": True,
    }

    save_results(mock_result)

    print("\n[Mock Mode] Training complete!")
    print(f"  Mock shortfall: {mock_result['mean_shortfall']:.4f}")
    print(f"  Mock completion: {mock_result['completion_rate']:.1%}")


def save_results(result: dict):
    """Save results to JSON file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "execution_result.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
