import asyncio
import json
from pathlib import Path
from typing import Any

import fire
from rdagent.app.qlib_rd_loop.conf import FACTOR_FROM_REPORT_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
    load_and_process_pdfs_by_langchain,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import LoopMeta


def generate_hypothesis(factor_result: dict, report_content: str) -> Hypothesis:
    """
    Generate a hypothesis based on factor results and report content.

    Args:
        factor_result (dict): The results of the factor analysis.
        report_content (str): The content of the report.

    Returns:
        Hypothesis: The generated hypothesis.
    """
    system_prompt = T(".prompts:hypothesis_generation.system").r()
    user_prompt = T(".prompts:hypothesis_generation.user").r(
        factor_descriptions=json.dumps(factor_result), report_content=report_content
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        json_target_type=dict[str, str],
    )

    response_json = json.loads(response)

    return Hypothesis(
        hypothesis=response_json.get("hypothesis", "No hypothesis provided"),
        reason=response_json.get("reason", "No reason provided"),
        concise_reason=response_json.get("concise_reason", "No concise reason provided"),
        concise_observation=response_json.get("concise_observation", "No concise observation provided"),
        concise_justification=response_json.get("concise_justification", "No concise justification provided"),
        concise_knowledge=response_json.get("concise_knowledge", "No concise knowledge provided"),
    )


def extract_hypothesis_and_exp_from_reports(report_file_path: str) -> QlibFactorExperiment | None:
    """
    Extract hypothesis and experiment details from report files.

    Args:
        report_file_path (str): Path to the report file.

    Returns:
        QlibFactorExperiment: An instance of QlibFactorExperiment containing the extracted details.
        None: If no valid experiment is found in the report.
    """
    exp = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
    if exp is None or exp.sub_tasks == []:
        return None

    pdf_screenshot = extract_first_page_screenshot_from_pdf(report_file_path)
    logger.log_object(pdf_screenshot, tag="load_pdf_screenshot")

    docs_dict = load_and_process_pdfs_by_langchain(report_file_path)

    factor_result = {
        task.factor_name: {
            "description": task.factor_description,
            "formulation": task.factor_formulation,
            "variables": task.variables,
            "resources": task.factor_resources,
        }
        for task in exp.sub_tasks
    }

    report_content = "\n".join(docs_dict.values())
    hypothesis = generate_hypothesis(factor_result, report_content)
    exp.hypothesis = hypothesis
    return exp


class FactorReportLoop(FactorRDLoop, metaclass=LoopMeta):
    def __init__(self, report_folder: str | None = None) -> None:
        super().__init__(PROP_SETTING=FACTOR_FROM_REPORT_PROP_SETTING)
        if report_folder is None:
            with Path(FACTOR_FROM_REPORT_PROP_SETTING.report_result_json_file_path).open() as f:
                self.judge_pdf_data_items = json.load(f)
        else:
            self.judge_pdf_data_items = list(Path(report_folder).rglob("*.pdf"))

        self.loop_n = min(len(self.judge_pdf_data_items), FACTOR_FROM_REPORT_PROP_SETTING.report_limit)
        self.shift_report = (
            0  # some reports does not contain viable factor, so we skip some of them to avoid infinite loop
        )

    async def direct_exp_gen(self, prev_out: dict[str, Any]) -> Any:
        del prev_out  # unused but required by parent class signature
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                report_file_path = self.judge_pdf_data_items[self.loop_idx + self.shift_report]
                logger.info(f"Processing number {self.loop_idx} report: {report_file_path}")
                exp = extract_hypothesis_and_exp_from_reports(str(report_file_path))
                if exp is None:
                    self.shift_report += 1
                    if self.loop_n is not None:
                        self.loop_n -= 1
                        if self.loop_n < 0:  # NOTE: on every step, we self.loop_n -= 1 at first.
                            msg = "Reach stop criterion and stop loop"
                            raise self.LoopTerminationError(msg)
                    continue
                exp.based_experiments = [  # type: ignore[assignment]
                    QlibFactorExperiment(sub_tasks=[], hypothesis=exp.hypothesis)
                ] + [t[0] for t in self.trace.hist if t[1]]
                exp.sub_workspace_list = exp.sub_workspace_list[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                exp.sub_tasks = exp.sub_tasks[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                logger.log_object(exp.hypothesis, tag="hypothesis generation")
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                return exp
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]) -> Any:
        exp = self.coder.develop(prev_out["direct_exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def feedback(self, prev_out: dict[str, Any]) -> None:
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


def main(
    report_folder: str | None = None,
    path: str | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
) -> None:
    """
    Auto R&D Evolving loop for fintech factors (extracted from finance reports).

    Args:
        report_folder: The folder containing report PDF files to load.
        path: The path for loading an existing session.
        all_duration: Duration limit for the loop.
        checkout: Whether to checkout when loading a session.
    """
    if path is None and report_folder is None:
        model_loop = FactorReportLoop()
    elif path is not None:
        model_loop = FactorReportLoop.load(path, checkout=checkout)
    else:
        model_loop = FactorReportLoop(report_folder=report_folder)

    asyncio.run(model_loop.run(all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
