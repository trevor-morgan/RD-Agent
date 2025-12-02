import json
from pathlib import Path

import pandas as pd
from rdagent.components.poetiq.conf import POETIQ_SETTINGS
from rdagent.components.poetiq.feedback import ScoredHypothesisFeedback, SoftScore
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]


def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Filter the combined DataFrame to retain only the important metrics
    filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        # Use Poetiq scoring when enabled
        if POETIQ_SETTINGS.enabled:
            return self._generate_poetiq_feedback(exp, trace)
        return self._generate_standard_feedback(exp, trace)

    def _generate_poetiq_feedback(self, exp: Experiment, trace: Trace) -> ScoredHypothesisFeedback:
        """Generate feedback using Poetiq soft scoring paradigm."""
        hypothesis = exp.hypothesis
        logger.info("Generating Poetiq scored feedback...")

        # Calculate metrics-based score components
        score_components = self._compute_score_components(exp)
        computed_score = self._aggregate_score(score_components)

        # Get top experiments for context
        top_experiments = self._get_top_experiments(trace, k=3)
        successful_count = sum(1 for _, fb in trace.hist if fb.decision)
        scores = [getattr(fb, "soft_score", None) for _, fb in trace.hist]
        avg_score = sum(s.value for s in scores if s) / max(len([s for s in scores if s]), 1)

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.poetiq_prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.poetiq_prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt with trajectory context
        user_prompt = T("scenarios.qlib.poetiq_prompts:model_feedback_generation.user").r(
            trace=trace,
            successful_count=successful_count,
            avg_score=f"{avg_score:.2f}",
            top_experiments=top_experiments,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "execution failed",
        )

        # Call the APIBackend to generate the response
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=dict[str, str | bool | int | float],
        )

        response_json = json.loads(response)

        # Extract LLM-provided score or use computed score
        llm_score = response_json.get("score")
        if llm_score is not None and isinstance(llm_score, (int, float)):
            final_score = float(llm_score)
        else:
            final_score = computed_score

        # Create soft score with components
        soft_score = SoftScore(
            value=final_score,
            confidence=0.8,  # Could be refined based on result quality
            components=score_components,
        )

        # Determine decision based on threshold
        threshold = POETIQ_SETTINGS.score_threshold
        decision = final_score >= threshold

        return ScoredHypothesisFeedback(
            observations=response_json.get("observations", "No observations provided"),
            hypothesis_evaluation=response_json.get("hypothesis_evaluation", "No feedback provided"),
            new_hypothesis=response_json.get("new_hypothesis", "No new hypothesis provided"),
            reason=response_json.get("reason", "No reasoning provided"),
            decision=decision,
            soft_score=soft_score,
        )

    def _generate_standard_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """Generate feedback using standard SOTA paradigm."""
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result.loc[IMPORTANT_METRICS] if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "execution failed",
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=dict[str, str | bool | int],
        )

        response_json = json.loads(response)
        return HypothesisFeedback(
            observations=response_json.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json.get("Decision", "false")),
        )

    def _compute_score_components(self, exp: Experiment) -> dict[str, float]:
        """Compute score components from experiment metrics."""
        components = {}

        if exp.result is None:
            return {"ic_score": 0.0, "return_score": 0.0, "training_quality": 0.3}

        try:
            # IC score: normalize from [-0.1, 0.1] to [0, 1]
            ic = exp.result.get("IC", 0.0)
            if hasattr(ic, "iloc"):
                ic = float(ic.iloc[0])
            components["ic_score"] = max(0.0, min(1.0, (float(ic) + 0.1) / 0.2))

            # Return score: normalize based on annualized return
            ann_return_key = "1day.excess_return_with_cost.annualized_return"
            ann_return = exp.result.get(ann_return_key, 0.0)
            if hasattr(ann_return, "iloc"):
                ann_return = float(ann_return.iloc[0])
            # Normalize: -0.5 to 0.5 -> 0 to 1
            components["return_score"] = max(0.0, min(1.0, (float(ann_return) + 0.5) / 1.0))

            # Training quality: based on whether training completed
            components["training_quality"] = 0.8 if exp.result is not None else 0.3

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Error computing score components: {e}")
            components = {"ic_score": 0.0, "return_score": 0.0, "training_quality": 0.3}

        return components

    def _aggregate_score(self, components: dict[str, float]) -> float:
        """Aggregate score components into final score."""
        # Weighted average: IC is most important
        weights = {"ic_score": 0.5, "return_score": 0.3, "training_quality": 0.2}
        total = sum(components.get(k, 0.0) * w for k, w in weights.items())
        return max(0.0, min(1.0, total))

    def _get_top_experiments(self, trace: Trace, k: int = 3) -> list[tuple]:
        """Get top-K experiments by score."""
        scored = []
        for exp, fb in trace.hist:
            if not fb.decision:
                continue
            soft_score = getattr(fb, "soft_score", None)
            score = soft_score.value if soft_score else 1.0
            scored.append((exp, fb, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]
