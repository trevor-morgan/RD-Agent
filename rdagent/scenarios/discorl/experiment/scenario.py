# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""DiscoRL Scenario for Optimal Execution.

This scenario uses DiscoRL's meta-learned RL algorithms (Disco103) to train
optimal execution policies. The LLM evolves environment configurations and
training hyperparameters while DiscoRL handles policy learning.
"""

from rdagent.core.scenario import Scenario
from rdagent.core.experiment import Task


class DiscoRLScenario(Scenario):
    """Scenario for training optimal execution policies using DiscoRL.

    Optimal execution is the problem of executing a large order over a fixed
    time horizon while minimizing market impact and slippage. This is an
    ideal application for DiscoRL because:

    1. Dense feedback: Every execution decision has immediate P&L feedback
    2. Discrete actions: Wait/passive/moderate/aggressive/urgent
    3. Game-like dynamics: Adversarial (other traders, market makers)
    4. Short episodes: Minutes to hours

    The LLM's role is to evolve:
    - Reward shaping parameters
    - State feature engineering
    - Training hyperparameters
    - Market simulation parameters

    DiscoRL's Disco103 algorithm handles the actual policy learning.
    """

    def __init__(self):
        super().__init__()

    @property
    def background(self) -> str:
        return """## Optimal Execution with DiscoRL

### Problem Statement
Execute a large order (buy or sell) over a fixed time horizon while minimizing:
- Implementation shortfall (execution price vs. arrival price)
- Market impact (permanent price movement caused by trading)
- Execution risk (failing to complete the order)

### Why DiscoRL?
DiscoRL uses Disco103, a meta-learned reinforcement learning algorithm that:
- Was trained on 103 diverse environments (Atari, ProcGen, DMLab)
- Discovers novel update rules that generalize to unseen domains
- Outperforms hand-designed algorithms like Actor-Critic

Optimal execution is ideal for DiscoRL because it resembles a game:
- Dense rewards (every trade has immediate P&L)
- Discrete action space (execution urgency levels)
- Adversarial dynamics (market participants)
- Short episodes (single order execution)

### Architecture
```
LLM (RD-Agent)                    DiscoRL
     |                                |
     | generates ExecutionConfig      |
     |------------------------------->|
     |                                |
     |    trains policy using         |
     |    Disco103 update rule        |
     |                                |
     |<-------------------------------|
     | receives ExecutionResult       |
     |                                |
     | evolves config based on        |
     | performance metrics            |
```
"""

    @property
    def source_data(self) -> str:
        return """## State Space (Observation)

The agent observes a 7-dimensional state vector at each timestep:

| Feature | Range | Description |
|---------|-------|-------------|
| remaining_qty | [0, 1] | Fraction of order remaining to execute |
| time_remaining | [0, 1] | Fraction of time horizon remaining |
| spread | [-1, 1] | Normalized bid-ask spread (vs baseline) |
| volatility | [-1, 1] | Normalized recent volatility (vs baseline) |
| momentum | [-1, 1] | Recent price momentum |
| imbalance | [-1, 1] | Order book imbalance (buy vs sell pressure) |
| vwap_deviation | [-1, 1] | Current VWAP vs arrival price |

## Action Space

Discrete actions representing execution urgency:

| Action | Name | Execution | Impact |
|--------|------|-----------|--------|
| 0 | WAIT | 0% | None |
| 1 | PASSIVE | 5% of remaining | Low (limit orders) |
| 2 | MODERATE | 10% of remaining | Medium |
| 3 | AGGRESSIVE | 25% of remaining | High |
| 4 | URGENT | 50% of remaining | Very high (cross spread) |

## Reward Signal

Per-step reward based on execution quality:
- **Base**: Implementation shortfall (arrival_price - exec_price) * qty
- **Bonus**: Beat arrival price bonus
- **Penalty**: Incomplete execution at deadline
- **Penalty**: Small cost for waiting (opportunity cost)
"""

    @property
    def interface(self) -> str:
        return """## ExecutionConfig Interface

The LLM generates configurations with these parameters:

```python
@dataclass
class ExecutionConfig:
    # Environment
    horizon: int = 50              # Steps to complete execution
    order_side: int = 1            # 1=buy, -1=sell

    # Market Simulation
    base_spread: float = 0.01      # Baseline bid-ask spread
    base_volatility: float = 0.02  # Baseline price volatility
    mean_reversion: float = 0.1    # Price mean reversion strength
    momentum_factor: float = 0.3   # Price momentum strength

    # Reward Shaping (CRITICAL for RL performance)
    arrival_price_bonus: float = 0.1   # Bonus for beating arrival
    incomplete_penalty: float = 5.0    # Penalty per unit unexecuted
    wait_penalty: float = 0.001        # Cost of waiting

    # Training
    num_training_steps: int = 100000
    batch_size: int = 64
    learning_rate: float = 3e-4
```

## ExecutionResult Interface

Results returned after training:

```python
@dataclass
class ExecutionResult:
    mean_shortfall: float      # Average implementation shortfall
    std_shortfall: float       # Shortfall standard deviation
    completion_rate: float     # Fraction of orders fully executed
    mean_vwap_vs_arrival: float
    action_distribution: dict  # How often each action was taken
    score: float              # Combined metric for optimization
```
"""

    @property
    def rich_style_description(self) -> str:
        return """### DiscoRL Optimal Execution

**Goal**: Train RL policies that execute large orders with minimal market impact.

**Method**: Use Disco103 (meta-learned RL algorithm) to learn execution strategies.

**LLM Role**: Evolve environment configs and reward shaping based on results.

**Key Metrics**:
- Implementation shortfall (lower is better)
- Completion rate (higher is better)
- Action distribution (balanced is usually better)
"""

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        """Get complete scenario description for LLM prompts."""
        parts = [self.background]

        if not simple_background:
            parts.append(self.source_data)
            parts.append(self.interface)

        if task is not None:
            parts.append(f"\n## Current Task\n{task.get_task_information()}")

        return "\n\n".join(parts)

    def get_runtime_environment(self) -> str:
        return """Runtime Requirements:
- Python >= 3.11
- JAX with GPU support (recommended)
- disco_rl package (from disco_rl repo)
- Dependencies: haiku, optax, distrax, rlax, ml_collections
"""

    @property
    def experiment_setting(self) -> str:
        return """## Experiment Settings

### Training
- Use Disco103 pre-trained weights (meta-learned on 103 environments)
- Train for 50,000-200,000 steps depending on complexity
- Batch size 32-128 (higher for stability)
- Learning rate 1e-4 to 5e-4

### Evaluation
- Run 100+ episodes with different random seeds
- Measure mean and std of implementation shortfall
- Track completion rate and action distribution
- Compare against TWAP/VWAP baselines

### Iteration Strategy
1. Start with default reward shaping
2. Analyze action distribution - adjust if too passive/aggressive
3. Tune incomplete_penalty based on completion rate
4. Adjust market simulation for robustness
"""
