# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""DiscoRL Scenario for RD-Agent.

This scenario integrates DiscoRL (meta-learned RL algorithms) with RD-Agent
for automated discovery of optimal execution strategies in quantitative finance.

The LLM evolves:
- Environment configurations (features, reward shaping)
- Training hyperparameters
- Market simulation parameters

DiscoRL handles the actual policy learning using Disco103 (the meta-learned update rule).
"""

from rdagent.scenarios.discorl.experiment.scenario import DiscoRLScenario
from rdagent.scenarios.discorl.experiment.discorl_experiment import (
    DiscoRLExperiment,
    DiscoRLTask,
    ExecutionConfig,
)
from rdagent.scenarios.discorl.experiment.workspace import DiscoRLFBWorkspace

__all__ = [
    "DiscoRLScenario",
    "DiscoRLExperiment",
    "DiscoRLTask",
    "DiscoRLFBWorkspace",
    "ExecutionConfig",
]
