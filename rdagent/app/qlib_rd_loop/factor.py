"""
Factor workflow with session control
"""

import asyncio
from pathlib import Path
from typing import Annotated, Any

import fire
import typer
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger


class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,)

    def running(self, prev_out: dict[str, Any]) -> Any:
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            msg = "Factor extraction failed."
            logger.error(msg)
            raise FactorEmptyError(msg)
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: str | None = None,
) -> None:
    """
    Auto R&D Evolving loop for fintech factors.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor.py \\
            $LOG_PATH/__session__/1/0_propose --step_n 1  # `step_n` is optional

    """
    checkout_value: bool | Path = Path(checkout_path) if checkout_path is not None else checkout

    model_loop = FactorRDLoop(FACTOR_PROP_SETTING) if path is None else FactorRDLoop.load(path, checkout=checkout_value)
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
