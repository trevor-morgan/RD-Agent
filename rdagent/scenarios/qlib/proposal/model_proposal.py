import json

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.components.poetiq.conf import POETIQ_SETTINGS
from rdagent.components.poetiq.trajectory import TrajectoryFormatter
from rdagent.components.proposal import ModelHypothesis2Experiment, ModelHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils.agent.tpl import T

QlibModelHypothesis = Hypothesis


class QlibModelHypothesisGen(ModelHypothesisGen):
    def __init__(self, scen: Scenario) -> tuple[dict, bool]:
        super().__init__(scen)
        self._trajectory_formatter = TrajectoryFormatter() if POETIQ_SETTINGS.enabled else None

    def prepare_context(self, trace: Trace) -> tuple[dict, bool]:
        # Use Poetiq exploration-based prompts when enabled
        if POETIQ_SETTINGS.enabled:
            return self._prepare_poetiq_context(trace)
        return self._prepare_standard_context(trace)

    def _prepare_poetiq_context(self, trace: Trace) -> tuple[dict, bool]:
        """Prepare context using Poetiq exploration paradigm."""
        trajectory_context = (
            T("scenarios.qlib.poetiq_prompts:trajectory_context").r(trace=trace)
            if len(trace.hist) > 0
            else "No experiments recorded yet. This is the first exploration."
        )

        last_experiment_context = (
            T("scenarios.qlib.poetiq_prompts:last_experiment_context").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "No previous experiment. Start with a simple, testable architecture."
        )

        # Get top experiments by score
        top_experiments = self._get_top_experiments(trace, k=3)
        top_experiments_summary = (
            T("scenarios.qlib.poetiq_prompts:top_experiments_summary").r(
                top_experiments=top_experiments
            )
            if top_experiments
            else "No successful experiments yet."
        )

        context_dict = {
            "trajectory_context": trajectory_context,
            "last_experiment_context": last_experiment_context,
            "top_experiments_summary": top_experiments_summary,
            "RAG": "1. In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. Do not generate GNN model as for now.\n2. The training data consists of less than 1 million samples for the training set and approximately 250,000 samples for the validation set. Please design the hyperparameters accordingly and control the model size.",
            "hypothesis_output_format": T("scenarios.qlib.poetiq_prompts:hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.poetiq_prompts:exploration_hypothesis_spec").r(),
        }
        return context_dict, True

    def _prepare_standard_context(self, trace: Trace) -> tuple[dict, bool]:
        """Prepare context using standard SOTA paradigm."""
        hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                trace=trace,
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        sota_hypothesis_and_feedback = ""
        if len(trace.hist) == 0:
            sota_hypothesis_and_feedback = "No SOTA hypothesis and feedback available since it is the first round."
        else:
            for i in range(len(trace.hist) - 1, -1, -1):
                if trace.hist[i][1].decision:
                    sota_hypothesis_and_feedback = T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                        experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                    )
                    break
            else:
                sota_hypothesis_and_feedback = (
                    "No SOTA hypothesis and feedback available since previous experiments were not accepted."
                )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            "RAG": "1. In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. Do not generate GNN model as for now.\n2. The training data consists of less than 1 million samples for the training set and approximately 250,000 samples for the validation set. Please design the hyperparameters accordingly and control the model size. This has a significant impact on the training results. If you believe that the previous model itself is good but the training hyperparameters or model hyperparameters are not optimal, you can return the same model and adjust these parameters instead.",
            "hypothesis_output_format": T("scenarios.qlib.prompts:hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.prompts:model_hypothesis_specification").r(),
        }
        return context_dict, True

    def _get_top_experiments(self, trace: Trace, k: int = 3) -> list[tuple]:
        """Get top-K experiments by score.

        Returns list of (experiment, feedback, score) tuples.
        """
        scored = []
        for exp, fb in trace.hist:
            if not fb.decision:
                continue
            soft_score = getattr(fb, "soft_score", None)
            score = soft_score.value if soft_score else 1.0
            scored.append((exp, fb, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibModelHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )
        return hypothesis


class QlibModelHypothesis2Experiment(ModelHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> tuple[dict, bool]:
        if isinstance(trace.scen, QlibQuantScenario):
            scenario = trace.scen.get_scenario_all_desc(action="model")
        else:
            scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = T("scenarios.qlib.prompts:model_experiment_output_format").r()

        last_experiment = None
        last_feedback = None
        sota_experiment = None
        sota_feedback = None

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            specific_trace = Trace(trace.scen)
            for i in range(len(trace.hist) - 1, -1, -1):
                if not hasattr(trace.hist[i][0].hypothesis, "action") or trace.hist[i][0].hypothesis.action == "model":
                    if last_experiment is None:
                        last_experiment = trace.hist[i][0]
                        last_feedback = trace.hist[i][1]
                    if trace.hist[i][1].decision is True and sota_experiment is None:
                        sota_experiment = trace.hist[i][0]
                        sota_feedback = trace.hist[i][1]
                    specific_trace.hist.insert(0, trace.hist[i])
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=last_experiment, feedback=last_feedback
            )
            if last_experiment is not None
            else "No previous hypothesis and feedback available since it's the first round."
        )

        sota_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                experiment=sota_experiment, feedback=sota_feedback
            )
            if sota_experiment is not None
            else "No SOTA hypothesis and feedback available since previous experiments were not accepted."
        )

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": "Note, the training data consists of less than 1 million samples for the training set and approximately 250,000 samples for the validation set. Please design the hyperparameters accordingly and control the model size. This has a significant impact on the training results. If you believe that the previous model itself is good but the training hyperparameters or model hyperparameters are not optimal, you can return the same model and adjust these parameters instead.",
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        for model_name in response_dict:
            description = response_dict[model_name]["description"]
            formulation = response_dict[model_name]["formulation"]
            architecture = response_dict[model_name]["architecture"]
            variables = response_dict[model_name]["variables"]
            hyperparameters = response_dict[model_name]["hyperparameters"]
            training_hyperparameters = response_dict[model_name]["training_hyperparameters"]
            model_type = response_dict[model_name]["model_type"]
            tasks.append(
                ModelTask(
                    name=model_name,
                    description=description,
                    formulation=formulation,
                    architecture=architecture,
                    variables=variables,
                    hyperparameters=hyperparameters,
                    training_hyperparameters=training_hyperparameters,
                    model_type=model_type,
                )
            )
        exp = QlibModelExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [t[0] for t in trace.hist if t[1] and isinstance(t[0], ModelExperiment)]
        return exp
