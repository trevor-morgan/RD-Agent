# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""DiscoRL Experiment and Task classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rdagent.core.experiment import Experiment, Task
from rdagent.core.proposal import Hypothesis
from rdagent.scenarios.discorl.experiment.workspace import DiscoRLFBWorkspace


@dataclass
class ExecutionConfig:
    """Configuration for optimal execution experiments.

    This is the main interface that the LLM evolves. Each field represents
    a design decision that affects training and evaluation.
    """

    # Environment configuration
    horizon: int = 50
    order_side: int = 1  # 1=buy, -1=sell

    # Market simulation parameters
    base_spread: float = 0.01
    base_volatility: float = 0.02
    mean_reversion: float = 0.1
    momentum_factor: float = 0.3

    # Reward shaping (critical for RL performance)
    arrival_price_bonus: float = 0.1
    incomplete_penalty: float = 5.0
    wait_penalty: float = 0.001

    # Training hyperparameters
    num_training_steps: int = 100_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    trajectory_length: int = 50

    # Algorithm selection
    use_disco: bool = True  # If False, use Actor-Critic baseline

    # Evaluation
    num_eval_episodes: int = 100
    eval_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "horizon": self.horizon,
            "order_side": self.order_side,
            "base_spread": self.base_spread,
            "base_volatility": self.base_volatility,
            "mean_reversion": self.mean_reversion,
            "momentum_factor": self.momentum_factor,
            "arrival_price_bonus": self.arrival_price_bonus,
            "incomplete_penalty": self.incomplete_penalty,
            "wait_penalty": self.wait_penalty,
            "num_training_steps": self.num_training_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "trajectory_length": self.trajectory_length,
            "use_disco": self.use_disco,
            "num_eval_episodes": self.num_eval_episodes,
            "eval_seeds": self.eval_seeds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionConfig:
        """Deserialize from storage."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"""ExecutionConfig(
    horizon={self.horizon}, order_side={'buy' if self.order_side == 1 else 'sell'},
    reward_shaping=(arrival_bonus={self.arrival_price_bonus}, incomplete_penalty={self.incomplete_penalty}),
    training=(steps={self.num_training_steps}, batch={self.batch_size}, lr={self.learning_rate}),
    algorithm={'Disco103' if self.use_disco else 'Actor-Critic'}
)"""


@dataclass
class ExecutionResult:
    """Results from a DiscoRL training run."""

    # Primary metrics
    mean_shortfall: float
    std_shortfall: float
    completion_rate: float

    # Detailed metrics
    mean_vwap_vs_arrival: float = 0.0
    mean_market_impact: float = 0.0
    mean_episode_length: float = 0.0

    # Per-action statistics
    action_distribution: dict[str, float] = field(default_factory=dict)

    # Training metrics
    final_loss: float = 0.0
    training_steps: int = 0
    converged: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "mean_shortfall": self.mean_shortfall,
            "std_shortfall": self.std_shortfall,
            "completion_rate": self.completion_rate,
            "mean_vwap_vs_arrival": self.mean_vwap_vs_arrival,
            "mean_market_impact": self.mean_market_impact,
            "mean_episode_length": self.mean_episode_length,
            "action_distribution": self.action_distribution,
            "final_loss": self.final_loss,
            "training_steps": self.training_steps,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionResult:
        """Deserialize from storage."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def score(self) -> float:
        """Single scalar score for optimization (higher is better)."""
        # Invert shortfall (we want to minimize it)
        shortfall_score = -self.mean_shortfall * 100

        # Completion bonus
        completion_score = self.completion_rate * 10

        # Consistency bonus (low variance)
        consistency_score = -self.std_shortfall * 50

        return shortfall_score + completion_score + consistency_score

    def __str__(self) -> str:
        """Human-readable representation."""
        action_str = ", ".join(f"{k}:{v:.1%}" for k, v in self.action_distribution.items())
        return f"""ExecutionResult(
    shortfall={self.mean_shortfall:.4f} +/- {self.std_shortfall:.4f},
    completion={self.completion_rate:.1%},
    actions=[{action_str}],
    score={self.score():.2f}
)"""


class DiscoRLTask(Task):
    """A single optimal execution task.

    Each task represents one configuration to evaluate.
    """

    def __init__(
        self,
        name: str,
        config: ExecutionConfig | None = None,
        description: str = "",
    ):
        """Initialize task.

        Args:
            name: Task identifier
            config: Execution configuration
            description: Human-readable description
        """
        super().__init__(name=name, version=1, description=description)
        self.config = config or ExecutionConfig()

    def get_task_information(self) -> str:
        """Get task information for LLM prompts."""
        return f"""Task: {self.name}
Description: {self.description}
Configuration:
{self.config}
"""


class DiscoRLExperiment(Experiment[DiscoRLTask, DiscoRLFBWorkspace, DiscoRLFBWorkspace]):
    """Container for DiscoRL optimal execution experiments.

    An experiment consists of:
    - One or more tasks (configurations to evaluate)
    - A hypothesis being tested
    - A workspace for execution
    - Results from training/evaluation
    """

    def __init__(
        self,
        sub_tasks: Sequence[DiscoRLTask],
        based_experiments: Sequence[Experiment] = (),
        hypothesis: Hypothesis | None = None,
    ):
        """Initialize experiment.

        Args:
            sub_tasks: Tasks to execute
            based_experiments: Previous experiments this builds on
            hypothesis: Hypothesis being tested
        """
        super().__init__(
            sub_tasks=sub_tasks,
            based_experiments=list(based_experiments),
            hypothesis=hypothesis,
        )

        # Initialize workspace with default template
        template_path = Path(__file__).parent / "templates" / "optimal_execution"
        if template_path.exists():
            self.experiment_workspace = DiscoRLFBWorkspace(
                template_folder_path=template_path
            )
        else:
            self.experiment_workspace = DiscoRLFBWorkspace()

        # Store results
        self._execution_result: ExecutionResult | None = None

    @property
    def execution_result(self) -> ExecutionResult | None:
        """Get execution result."""
        return self._execution_result

    @execution_result.setter
    def execution_result(self, value: ExecutionResult | dict[str, Any] | None):
        """Set execution result."""
        if value is None:
            self._execution_result = None
        elif isinstance(value, ExecutionResult):
            self._execution_result = value
        elif isinstance(value, dict):
            self._execution_result = ExecutionResult.from_dict(value)
        else:
            raise ValueError(f"Invalid result type: {type(value)}")

    def get_config(self) -> ExecutionConfig:
        """Get configuration from first task."""
        if self.sub_tasks:
            return self.sub_tasks[0].config
        return ExecutionConfig()

    def set_config(self, config: ExecutionConfig):
        """Set configuration on first task."""
        if self.sub_tasks:
            self.sub_tasks[0].config = config
