# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""DiscoRL Runner for executing experiments."""

from __future__ import annotations

from rdagent.core.developer import Developer
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.discorl.experiment.discorl_experiment import (
    DiscoRLExperiment,
    ExecutionResult,
)
from rdagent.scenarios.discorl.experiment.scenario import DiscoRLScenario


class DiscoRLRunner(Developer[DiscoRLExperiment]):
    """Executes DiscoRL optimal execution experiments."""

    def __init__(self, scen: DiscoRLScenario):
        """Initialize runner.

        Args:
            scen: The DiscoRL scenario
        """
        super().__init__(scen)

    def develop(self, exp: DiscoRLExperiment) -> DiscoRLExperiment:
        """Execute the experiment.

        Args:
            exp: Experiment to execute

        Returns:
            The experiment with results populated
        """
        logger.info(f"Starting DiscoRL experiment: {exp.sub_tasks[0].name if exp.sub_tasks else 'unnamed'}")

        try:
            # Execute workspace
            output = exp.experiment_workspace.execute(
                entry="train.py",
                timeout=7200,  # 2 hour timeout for training
            )

            logger.info(f"Execution output:\n{output[:1000]}...")

            # Get results from workspace
            result_dict = exp.experiment_workspace.get_result()

            if result_dict:
                exp.execution_result = ExecutionResult.from_dict(result_dict)
                logger.info(f"Experiment completed: score={exp.execution_result.score():.2f}")
            else:
                logger.warning("No results found after execution")
                exp.execution_result = None

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            exp.execution_result = None

        return exp
