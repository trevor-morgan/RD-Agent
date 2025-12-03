#!/usr/bin/env python3
# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""DiscoRL RD-Agent Loop.

This module implements the main research loop for DiscoRL optimal execution:

1. Generate hypothesis (what might improve execution?)
2. Convert to experiment (generate config)
3. Execute training (run DiscoRL with Disco103)
4. Evaluate results (generate feedback)
5. Update trace and iterate

Usage:
    python -m rdagent.app.discorl.loop --max_iterations 10

Or via the CLI:
    rdagent discorl --max_iterations 10
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from rdagent.core.proposal import Trace
from rdagent.log import rdagent_logger as logger

from rdagent.scenarios.discorl import (
    DiscoRLScenario,
    DiscoRLExperiment,
    DiscoRLTask,
    ExecutionConfig,
)
from rdagent.scenarios.discorl.proposal import (
    DiscoRLHypothesisGen,
    DiscoRLHypothesis2Experiment,
    DiscoRLExperiment2Feedback,
)
from rdagent.scenarios.discorl.developer import DiscoRLRunner


def run_discorl_loop(
    max_iterations: int = 10,
    output_dir: str | Path = "./discorl_output",
    disco_rl_path: str | Path | None = None,
) -> Trace:
    """Run the DiscoRL research loop.

    Args:
        max_iterations: Maximum number of iterations
        output_dir: Directory for outputs
        disco_rl_path: Path to disco_rl repo

    Returns:
        Trace with experiment history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set disco_rl path in environment
    if disco_rl_path:
        os.environ["DISCO_RL_PATH"] = str(disco_rl_path)

    # Initialize components
    logger.info("Initializing DiscoRL scenario...")
    scenario = DiscoRLScenario()

    hypothesis_gen = DiscoRLHypothesisGen(scenario)
    hypothesis_2_exp = DiscoRLHypothesis2Experiment(scenario)
    runner = DiscoRLRunner(scenario)
    feedback_gen = DiscoRLExperiment2Feedback(scenario)

    # Initialize trace
    trace = Trace(scen=scenario)

    # Run initial experiment with default config
    logger.info("Running initial experiment with default config...")
    initial_task = DiscoRLTask(
        name="initial_baseline",
        config=ExecutionConfig(),
        description="Baseline experiment with default configuration",
    )
    initial_exp = DiscoRLExperiment(sub_tasks=[initial_task])

    # Execute initial experiment
    runner.develop(initial_exp)

    if initial_exp.execution_result:
        initial_feedback = feedback_gen.generate_feedback(initial_exp, trace)
        trace.hist.append((initial_exp, initial_feedback))
        logger.info(f"Initial score: {initial_exp.execution_result.score():.2f}")

        # Save initial results
        save_iteration(output_dir, 0, initial_exp, initial_feedback)
    else:
        logger.error("Initial experiment failed!")
        return trace

    # Main loop
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Iteration {iteration}/{max_iterations}")
        logger.info(f"{'=' * 60}")

        # 1. Generate hypothesis
        logger.info("Generating hypothesis...")
        hypothesis = hypothesis_gen.gen(trace)
        logger.info(f"Hypothesis: {hypothesis.hypothesis[:100]}...")

        # 2. Convert to experiment
        logger.info("Converting hypothesis to experiment...")
        experiment = hypothesis_2_exp.convert(hypothesis, trace)
        logger.info(f"Config: {experiment.get_config()}")

        # 3. Execute
        logger.info("Executing experiment...")
        runner.develop(experiment)

        if experiment.execution_result is None:
            logger.warning("Experiment failed, skipping...")
            continue

        # 4. Generate feedback
        logger.info("Generating feedback...")
        feedback = feedback_gen.generate_feedback(experiment, trace)
        logger.info(f"Decision: {'ACCEPTED' if feedback.decision else 'REJECTED'}")
        logger.info(f"Score: {experiment.execution_result.score():.2f}")

        # 5. Update trace
        trace.hist.append((experiment, feedback))

        # Save iteration
        save_iteration(output_dir, iteration, experiment, feedback)

        # Check for convergence or early stopping
        if should_stop(trace):
            logger.info("Stopping early - convergence detected")
            break

    # Final summary
    print_summary(trace)
    save_summary(output_dir, trace)

    return trace


def save_iteration(
    output_dir: Path,
    iteration: int,
    exp: DiscoRLExperiment,
    feedback,
):
    """Save iteration results."""
    iter_dir = output_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(exist_ok=True)

    # Save config
    with open(iter_dir / "config.json", "w") as f:
        json.dump(exp.get_config().to_dict(), f, indent=2)

    # Save results
    if exp.execution_result:
        with open(iter_dir / "results.json", "w") as f:
            json.dump(exp.execution_result.to_dict(), f, indent=2)

    # Save hypothesis
    if exp.hypothesis:
        with open(iter_dir / "hypothesis.txt", "w") as f:
            f.write(f"Hypothesis: {exp.hypothesis.hypothesis}\n")
            f.write(f"Reason: {exp.hypothesis.reason}\n")

    # Save feedback
    with open(iter_dir / "feedback.json", "w") as f:
        json.dump({
            "decision": feedback.decision,
            "reason": feedback.reason,
        }, f, indent=2)


def should_stop(trace: Trace) -> bool:
    """Check if we should stop early."""
    if len(trace.hist) < 3:
        return False

    # Check if last 3 iterations showed no improvement
    recent_scores = []
    for exp, _ in trace.hist[-3:]:
        if hasattr(exp, "execution_result") and exp.execution_result:
            recent_scores.append(exp.execution_result.score())

    if len(recent_scores) < 3:
        return False

    # Stop if variance is very low (converged)
    import statistics
    if statistics.stdev(recent_scores) < 0.1:
        return True

    return False


def print_summary(trace: Trace):
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    best_score = float("-inf")
    best_exp = None
    best_idx = -1

    for i, (exp, feedback) in enumerate(trace.hist):
        if hasattr(exp, "execution_result") and exp.execution_result:
            score = exp.execution_result.score()
            decision = "+" if feedback.decision else "-"
            print(f"[{decision}] Iteration {i}: score={score:.2f}")

            if score > best_score:
                best_score = score
                best_exp = exp
                best_idx = i

    if best_exp and best_exp.execution_result:
        print("\n" + "-" * 60)
        print(f"BEST RESULT (Iteration {best_idx})")
        print("-" * 60)
        print(f"Score: {best_score:.2f}")
        print(f"Shortfall: {best_exp.execution_result.mean_shortfall:.4f}")
        print(f"Completion: {best_exp.execution_result.completion_rate:.1%}")
        print(f"Config: {best_exp.get_config()}")


def save_summary(output_dir: Path, trace: Trace):
    """Save experiment summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_iterations": len(trace.hist),
        "iterations": [],
    }

    best_score = float("-inf")
    best_idx = -1

    for i, (exp, feedback) in enumerate(trace.hist):
        iter_summary = {
            "iteration": i,
            "decision": feedback.decision,
            "reason": feedback.reason[:200] if feedback.reason else "",
        }

        if hasattr(exp, "execution_result") and exp.execution_result:
            score = exp.execution_result.score()
            iter_summary["score"] = score
            iter_summary["shortfall"] = exp.execution_result.mean_shortfall
            iter_summary["completion_rate"] = exp.execution_result.completion_rate

            if score > best_score:
                best_score = score
                best_idx = i

        if exp.hypothesis:
            iter_summary["hypothesis"] = exp.hypothesis.hypothesis[:200]

        summary["iterations"].append(iter_summary)

    summary["best_iteration"] = best_idx
    summary["best_score"] = best_score

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run DiscoRL optimal execution research loop"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of iterations (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./discorl_output",
        help="Output directory (default: ./discorl_output)",
    )
    parser.add_argument(
        "--disco_rl_path",
        type=str,
        default=None,
        help="Path to disco_rl repository",
    )

    args = parser.parse_args()

    trace = run_discorl_loop(
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        disco_rl_path=args.disco_rl_path,
    )

    print(f"\nExperiment complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
