"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`

Supports seed models - provide your own model as a starting point for evolution.
"""

import asyncio
from pathlib import Path
from typing import Any

from rdagent.components.poetiq.conf import POETIQ_SETTINGS
from rdagent.components.poetiq.early_exit import EarlyExitChecker
from rdagent.components.poetiq.exploration import ParallelHypothesisGen, select_diverse_hypotheses
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.utils.workflow import LoopBase, LoopMeta


class RDLoop(LoopBase, metaclass=LoopMeta):

    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")
        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)

        # Enable Poetiq parallel hypothesis generation when configured
        self._parallel_enabled = POETIQ_SETTINGS.enabled and POETIQ_SETTINGS.parallel_experts > 1
        if self._parallel_enabled:
            self.hypothesis_gen = ParallelHypothesisGen(self.hypothesis_gen)

        self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()

        self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
        self.runner: Developer = import_class(PROP_SETTING.runner)(scen)

        self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
        self.trace = Trace(scen=scen)
        self.scen = scen  # Store scenario reference for seed model loading

        # Buffer for extra hypotheses generated in parallel mode
        self._hypothesis_buffer: list[Hypothesis] = []

        # Initialize Poetiq early exit checker if enabled
        self._early_exit_checker = EarlyExitChecker() if POETIQ_SETTINGS.enabled else None

        super().__init__()

    def _load_seed_model(self, model_path: str, hypothesis_text: str) -> None:
        """
        Load a user-provided model as the initial SOTA in the trace.

        This allows users to evolve from their own model architecture rather than
        starting from scratch. The seed model will be:
        1. Loaded and validated
        2. Run through the evaluation pipeline to get actual metrics
        3. Added to trace.hist as SOTA (decision=True)

        The LLM will then see this as the baseline to beat.

        Args:
            model_path: Path to model.py file containing a Net class
            hypothesis_text: Description of the model architecture for LLM context
        """
        from rdagent.components.coder.model_coder.model import (
            ModelFBWorkspace,
            ModelTask,
        )
        from rdagent.scenarios.qlib.experiment.model_experiment import (
            QlibModelExperiment,
        )

        logger.info(f"Loading seed model from: {model_path}")

        # Read model code
        model_code = Path(model_path).read_text()

        # Detect model type from code
        model_type = "TimeSeries" if "num_timesteps" in model_code else "Tabular"

        # Create a ModelTask for the seed model
        seed_task = ModelTask(
            name="SeedModel",
            description=hypothesis_text,
            architecture=f"User-provided seed model from {Path(model_path).name}",
            hyperparameters={},
            training_hyperparameters={
                "n_epochs": "100",
                "lr": "1e-3",
                "early_stop": "10",
                "batch_size": "256",
                "weight_decay": "1e-4",
            },
            formulation="User-defined architecture",
            variables={},
            model_type=model_type,
        )

        # Create hypothesis for the seed model
        seed_hypothesis = Hypothesis(
            hypothesis=hypothesis_text,
            reason="User-provided seed model for evolution baseline",
            concise_reason="Seed model baseline",
            concise_observation="Starting point for model evolution",
            concise_justification="User believes this architecture has potential",
            concise_knowledge="Domain expertise encoded in architecture",
        )

        # Create experiment
        seed_exp = QlibModelExperiment(sub_tasks=[seed_task], hypothesis=seed_hypothesis)

        # Create workspace and inject code
        workspace = ModelFBWorkspace(target_task=seed_task)
        workspace.inject_files(**{"model.py": model_code})
        seed_exp.sub_workspace_list = [workspace]

        # Inject the model code into experiment workspace
        seed_exp.experiment_workspace.inject_files(**{"model.py": model_code})

        logger.info(f"Running seed model evaluation (model_type={model_type})...")

        try:
            # Run through the runner to get actual metrics
            seed_exp = self.runner.develop(seed_exp)

            # Check if we got results
            if seed_exp.result is not None:
                logger.info(f"Seed model evaluation complete. Results:\n{seed_exp.result}")

                # Create positive feedback marking this as SOTA
                seed_feedback = HypothesisFeedback(
                    observations=f"Seed model successfully trained and evaluated. Results: {seed_exp.result}",
                    hypothesis_evaluation="Seed model serves as the baseline for evolution.",
                    new_hypothesis="The LLM should analyze this architecture and propose improvements.",
                    reason="User-provided seed model accepted as initial SOTA.",
                    decision=True,  # Mark as SOTA so LLM tries to beat it
                )

                # Add to trace history
                self.trace.hist.append((seed_exp, seed_feedback))
                logger.info("Seed model added to trace as initial SOTA. LLM will try to improve upon it.")

            else:
                logger.warning(f"Seed model evaluation returned no results. Stdout: {seed_exp.stdout}")
                # Still add to trace but with negative feedback
                seed_feedback = HypothesisFeedback(
                    observations=f"Seed model failed to produce results. Output: {seed_exp.stdout}",
                    hypothesis_evaluation="Seed model evaluation failed.",
                    new_hypothesis="Start with a simpler architecture.",
                    reason="Seed model did not produce valid results.",
                    decision=False,
                )
                self.trace.hist.append((seed_exp, seed_feedback))

        except Exception as e:
            logger.error(f"Error evaluating seed model: {e}")
            # Add failed attempt to trace
            seed_feedback = HypothesisFeedback(
                observations=f"Seed model evaluation error: {e!s}",
                hypothesis_evaluation="Seed model could not be evaluated.",
                new_hypothesis="Fix the model implementation or start fresh.",
                reason=f"Evaluation error: {e!s}",
                decision=False,
            )
            self.trace.hist.append((seed_exp, seed_feedback))
            raise

    # excluded steps
    def _propose(self) -> Hypothesis:
        hypothesis = self.hypothesis_gen.gen(self.trace)
        logger.log_object(hypothesis, tag="hypothesis generation")
        return hypothesis

    def _next_hypothesis(self) -> Hypothesis:
        """Get the next hypothesis, using parallel generation buffer when enabled."""
        if self._hypothesis_buffer:
            return self._hypothesis_buffer.pop(0)

        if self._parallel_enabled and hasattr(self.hypothesis_gen, "gen_parallel"):
            hypotheses = self.hypothesis_gen.gen_parallel(self.trace)
            hypotheses = select_diverse_hypotheses(hypotheses, max_select=POETIQ_SETTINGS.parallel_experts)
            if hypotheses:
                primary = hypotheses[0]
                self._hypothesis_buffer.extend(hypotheses[1:])
                logger.log_object(
                    {"primary": primary, "buffered": len(self._hypothesis_buffer)},
                    tag="poetiq_parallel_hypothesis_generation",
                )
                return primary

        return self._propose()

    def _exp_gen(self, hypothesis: Hypothesis):
        exp = self.hypothesis2experiment.convert(hypothesis, self.trace)
        logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    # included steps
    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._next_hypothesis()
                exp = self._exp_gen(hypo)
                return {"propose": hypo, "exp_gen": exp}
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        exp = self.coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["running"], feedback))

        # Check for Poetiq early exit condition
        if self._early_exit_checker is not None and POETIQ_SETTINGS.enabled:
            should_exit, reason = self._early_exit_checker.should_exit(self.trace)
            if should_exit:
                logger.info(f"Poetiq early exit triggered: {reason}")
                from rdagent.utils.workflow.loop import LoopTerminationError

                raise LoopTerminationError(reason)

    # TODO: `def record(self, prev_out: dict[str, Any]):` has already been hard coded into LoopBase
    # So we should add it into RDLoop class to make sure every RDLoop Sub Class be aware of it.
