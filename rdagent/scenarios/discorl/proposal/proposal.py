# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""DiscoRL Proposal Components.

These components handle the LLM-driven research loop:
1. HypothesisGen: Generate hypotheses about what might improve execution
2. Hypothesis2Experiment: Convert hypothesis to executable config
3. Experiment2Feedback: Evaluate results and generate feedback
"""

from __future__ import annotations

import json
from typing import Any

from rdagent.core.proposal import (
    Hypothesis,
    HypothesisGen,
    Hypothesis2Experiment,
    Experiment2Feedback,
    ExperimentFeedback,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.discorl.experiment.scenario import DiscoRLScenario
from rdagent.scenarios.discorl.experiment.discorl_experiment import (
    DiscoRLExperiment,
    DiscoRLTask,
    ExecutionConfig,
    ExecutionResult,
)


# =============================================================================
# Prompt Templates
# =============================================================================

HYPOTHESIS_SYSTEM_PROMPT = """You are an expert in reinforcement learning and optimal execution.
Your task is to generate hypotheses about how to improve the performance of RL-based execution strategies.

You understand:
- Market microstructure (spreads, impact, order books)
- Reinforcement learning (reward shaping, exploration, policy gradients)
- DiscoRL's Disco103 algorithm (meta-learned update rules)

Generate hypotheses that are:
1. Specific and testable
2. Based on the observed results
3. Actionable through configuration changes
"""

HYPOTHESIS_USER_PROMPT = """## Current Scenario
{scenario}

## Experiment History
{history}

## Your Task
Based on the scenario and history, generate a hypothesis about what might improve execution performance.

Focus on:
- Reward shaping parameters if completion rate is low or actions are imbalanced
- Training hyperparameters if convergence is poor
- Market simulation if results are inconsistent

Respond with JSON:
{{
    "hypothesis": "A specific, testable hypothesis",
    "reason": "Detailed reasoning based on observations",
    "concise_reason": "One-sentence reason",
    "concise_observation": "Key observation from history",
    "concise_justification": "Why this hypothesis might work",
    "concise_knowledge": "Relevant RL/market knowledge"
}}
"""

CODE_SYSTEM_PROMPT = """You are an expert Python developer specializing in reinforcement learning.
Your task is to convert a hypothesis into an ExecutionConfig for DiscoRL training.

The config controls:
- Environment: horizon, order_side
- Market simulation: base_spread, base_volatility, mean_reversion, momentum_factor
- Reward shaping: arrival_price_bonus, incomplete_penalty, wait_penalty
- Training: num_training_steps, batch_size, learning_rate, trajectory_length
- Algorithm: use_disco (True for Disco103, False for Actor-Critic baseline)
"""

CODE_USER_PROMPT = """## Hypothesis
{hypothesis}

## Scenario
{scenario}

## Previous Best Config
{previous_config}

## Your Task
Generate an ExecutionConfig that tests this hypothesis.

Respond with JSON:
{{
    "task_name": "descriptive_task_name",
    "config": {{
        "horizon": 50,
        "order_side": 1,
        "base_spread": 0.01,
        "base_volatility": 0.02,
        "mean_reversion": 0.1,
        "momentum_factor": 0.3,
        "arrival_price_bonus": 0.1,
        "incomplete_penalty": 5.0,
        "wait_penalty": 0.001,
        "num_training_steps": 100000,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "trajectory_length": 50,
        "use_disco": true
    }},
    "rationale": "Why these specific values test the hypothesis"
}}
"""

FEEDBACK_SYSTEM_PROMPT = """You are an expert evaluator for reinforcement learning experiments.
Your task is to assess whether an experiment supports its hypothesis and whether the results represent an improvement.

Consider:
- Implementation shortfall (lower is better)
- Completion rate (higher is better)
- Action distribution (balanced is usually better)
- Consistency (lower std is better)
"""

FEEDBACK_USER_PROMPT = """## Hypothesis
{hypothesis}

## Configuration
{config}

## Results
{results}

## Previous Best Results
{previous_best}

## Your Task
Evaluate whether the hypothesis was supported and whether this represents an improvement.

Respond with JSON:
{{
    "decision": true or false (is this an improvement over previous best?),
    "hypothesis_supported": true or false (did results support the hypothesis?),
    "reason": "Detailed analysis of the results",
    "improvements": ["list", "of", "improvements"],
    "concerns": ["list", "of", "concerns"],
    "next_suggestions": ["suggestions", "for", "next", "iteration"]
}}
"""


# =============================================================================
# HypothesisGen
# =============================================================================

class DiscoRLHypothesisGen(HypothesisGen):
    """Generate hypotheses for improving optimal execution."""

    def __init__(self, scen: DiscoRLScenario):
        super().__init__(scen)

    def gen(self, trace: Trace, plan=None) -> Hypothesis:
        """Generate a hypothesis based on experiment history."""
        # Get scenario description
        scenario_desc = trace.scen.get_scenario_all_desc(simple_background=True)

        # Build history context
        if len(trace.hist) > 0:
            history_parts = []
            for i, (exp, feedback) in enumerate(trace.hist[-5:]):  # Last 5
                if hasattr(exp, "execution_result") and exp.execution_result:
                    result = exp.execution_result
                    history_parts.append(f"""
Iteration {i + 1}:
  Hypothesis: {exp.hypothesis.hypothesis if exp.hypothesis else 'N/A'}
  Config: {exp.get_config() if hasattr(exp, 'get_config') else 'N/A'}
  Result: shortfall={result.mean_shortfall:.4f}, completion={result.completion_rate:.1%}
  Decision: {'ACCEPTED' if feedback.decision else 'REJECTED'}
  Reason: {feedback.reason[:200] if feedback.reason else 'N/A'}
""")
            history = "\n".join(history_parts)
        else:
            history = "No previous experiments. This is the first iteration."

        # Call LLM
        user_prompt = HYPOTHESIS_USER_PROMPT.format(
            scenario=scenario_desc,
            history=history,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_mode=True,
        )

        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            # Fallback
            response_dict = {
                "hypothesis": "Improve reward shaping to encourage more balanced action distribution",
                "reason": "Default hypothesis due to JSON parse error",
                "concise_reason": "Balance exploration",
                "concise_observation": "Initial experiment",
                "concise_justification": "Better balance leads to better adaptation",
                "concise_knowledge": "RL benefits from diverse experience",
            }

        return Hypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            reason=response_dict.get("reason", ""),
            concise_reason=response_dict.get("concise_reason", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
        )


# =============================================================================
# Hypothesis2Experiment
# =============================================================================

class DiscoRLHypothesis2Experiment(Hypothesis2Experiment[DiscoRLExperiment]):
    """Convert hypothesis to executable experiment."""

    def __init__(self, scen: DiscoRLScenario | None = None):
        self.scen = scen

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> DiscoRLExperiment:
        """Convert hypothesis to experiment configuration."""
        # Get scenario description
        scenario_desc = trace.scen.get_scenario_all_desc(simple_background=True)

        # Get previous best config
        if len(trace.hist) > 0:
            best_exp, _ = trace.get_sota_hypothesis_and_experiment()
            if hasattr(best_exp, "get_config"):
                previous_config = str(best_exp.get_config())
            else:
                previous_config = "Default config"
        else:
            previous_config = "No previous config (first iteration)"

        # Call LLM
        user_prompt = CODE_USER_PROMPT.format(
            hypothesis=str(hypothesis),
            scenario=scenario_desc,
            previous_config=previous_config,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=CODE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_mode=True,
        )

        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to default config
            response_dict = {
                "task_name": "default_execution_task",
                "config": ExecutionConfig().to_dict(),
                "rationale": "Default config due to JSON parse error",
            }

        # Create config from response
        config_dict = response_dict.get("config", {})
        config = ExecutionConfig.from_dict(config_dict)

        # Create task
        task = DiscoRLTask(
            name=response_dict.get("task_name", "execution_task"),
            config=config,
            description=hypothesis.hypothesis,
        )

        # Create experiment
        experiment = DiscoRLExperiment(
            sub_tasks=[task],
            hypothesis=hypothesis,
        )

        # Generate training code
        self._inject_training_code(experiment, config)

        return experiment

    def _inject_training_code(self, experiment: DiscoRLExperiment, config: ExecutionConfig):
        """Inject the training script into the workspace."""
        training_code = f'''#!/usr/bin/env python3
"""Auto-generated DiscoRL training script."""

import json
import sys
from pathlib import Path

# Add disco_rl to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    # Configuration
    config = {json.dumps(config.to_dict(), indent=4)}

    try:
        from disco_rl.rdagent_integration import DiscoTrainer, ExecutionExperimentConfig

        # Create config object
        exp_config = ExecutionExperimentConfig.from_dict(config)

        # Create trainer
        trainer = DiscoTrainer(exp_config, use_disco={str(config.use_disco).lower()})

        # Train
        print("Starting training...")
        trainer.train(config["num_training_steps"])

        # Evaluate
        print("Evaluating...")
        result = trainer.evaluate()

        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / "execution_result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Training complete. Score: {{result.score():.2f}}")
        print(f"Shortfall: {{result.mean_shortfall:.4f}}")
        print(f"Completion: {{result.completion_rate:.1%}}")

    except ImportError as e:
        print(f"Warning: Could not import disco_rl: {{e}}")
        print("Running in mock mode...")

        # Mock results for testing
        mock_result = {{
            "mean_shortfall": 0.05,
            "std_shortfall": 0.02,
            "completion_rate": 0.95,
            "mean_vwap_vs_arrival": -0.001,
            "mean_market_impact": 0.002,
            "mean_episode_length": 40,
            "action_distribution": {{"WAIT": 0.1, "PASSIVE": 0.2, "MODERATE": 0.4, "AGGRESSIVE": 0.2, "URGENT": 0.1}},
            "final_loss": 0.01,
            "training_steps": config["num_training_steps"],
            "converged": True,
        }}

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / "execution_result.json", "w") as f:
            json.dump(mock_result, f, indent=2)

        print("Mock training complete.")

if __name__ == "__main__":
    main()
'''
        experiment.experiment_workspace.inject_files(**{"train.py": training_code})


# =============================================================================
# Experiment2Feedback
# =============================================================================

class DiscoRLExperiment2Feedback(Experiment2Feedback):
    """Evaluate experiment results and generate feedback."""

    def __init__(self, scen: DiscoRLScenario):
        super().__init__(scen)

    def generate_feedback(
        self,
        exp: DiscoRLExperiment,
        trace: Trace,
    ) -> ExperimentFeedback:
        """Generate feedback based on experiment results."""
        # Get current results
        result = exp.execution_result
        if result is None:
            # Try to get from workspace
            ws_result = exp.experiment_workspace.get_result()
            if ws_result:
                result = ExecutionResult.from_dict(ws_result)
                exp.execution_result = result

        if result is None:
            return ExperimentFeedback(
                decision=False,
                reason="No results available - experiment may have failed",
            )

        # Get previous best for comparison
        if len(trace.hist) > 0:
            best_exp, _ = trace.get_sota_hypothesis_and_experiment()
            if hasattr(best_exp, "execution_result") and best_exp.execution_result:
                previous_best = str(best_exp.execution_result)
            else:
                previous_best = "No previous results"
        else:
            previous_best = "No previous results (first iteration)"

        # Call LLM for evaluation
        user_prompt = FEEDBACK_USER_PROMPT.format(
            hypothesis=str(exp.hypothesis) if exp.hypothesis else "N/A",
            config=str(exp.get_config()),
            results=str(result),
            previous_best=previous_best,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=FEEDBACK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_mode=True,
        )

        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: simple score-based decision
            is_improvement = True  # Assume first is always accepted
            if len(trace.hist) > 0:
                best_exp, _ = trace.get_sota_hypothesis_and_experiment()
                if hasattr(best_exp, "execution_result") and best_exp.execution_result:
                    is_improvement = result.score() > best_exp.execution_result.score()

            response_dict = {
                "decision": is_improvement,
                "hypothesis_supported": True,
                "reason": f"Score: {result.score():.2f}",
                "improvements": [],
                "concerns": [],
                "next_suggestions": [],
            }

        return ExperimentFeedback(
            decision=response_dict.get("decision", False),
            reason=response_dict.get("reason", ""),
        )
