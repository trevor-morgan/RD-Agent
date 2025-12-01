"""
Model workflow with session control

Supports seed models for evolution - provide your own model as a starting point.
"""

import asyncio
import os
from pathlib import Path

import fire

from rdagent.app.qlib_rd_loop.conf import MODEL_PROP_SETTING, ModelBasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import ModelEmptyError
from rdagent.log import rdagent_logger as logger


class ModelRDLoop(RDLoop):
    skip_loop_error = (ModelEmptyError,)


def main(
    path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
    seed_model: str | None = None,
    seed_hypothesis: str | None = None,
    data_region: str | None = None,
):
    """
    Auto R&D Evolving loop for fintech models

    Args:
        path: Path to resume a previous session
        step_n: Number of steps to run
        loop_n: Number of loops to run
        all_duration: Total duration to run (e.g., "1h", "30m")
        checkout: Whether to checkout session state
        seed_model: Path to a model.py file to use as the seed model for evolution.
                   The model will be evaluated first and used as the SOTA baseline.
                   LLM will then try to improve upon this architecture.
        seed_hypothesis: Description of the seed model's architecture and approach.
                        This helps the LLM understand what it's trying to improve.
                        Example: "Symplectic physics-informed model with Hurst exponent features"
        data_region: Data region to use: cn_data (default), us_data, or alpaca_us

    Example with seed model:

    .. code-block:: bash

        rdagent fin_model \\
            --seed-model ./my_model.py \\
            --seed-hypothesis "Transformer with multi-head attention for time series" \\
            --data-region alpaca_us \\
            --loop-n 5

    """
    # Handle data region environment variable
    if data_region is not None:
        os.environ["QLIB_DATA_REGION"] = data_region
        logger.info(f"Using data region: {data_region}")

    # Create settings with seed model if provided
    settings = MODEL_PROP_SETTING

    if seed_model is not None:
        # Validate seed model path
        seed_path = Path(seed_model)
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed model not found: {seed_model}")

        # Update settings with seed model info
        settings = ModelBasePropSetting(
            seed_model_path=str(seed_path.absolute()),
            seed_hypothesis=seed_hypothesis or f"User-provided model from {seed_path.name}",
            data_region=data_region or os.environ.get("QLIB_DATA_REGION", "cn_data"),
        )
        logger.info(f"Using seed model: {seed_model}")
        if seed_hypothesis:
            logger.info(f"Seed hypothesis: {seed_hypothesis}")

    if path is None:
        model_loop = ModelRDLoop(settings)

        # Load seed model if provided
        if seed_model is not None and hasattr(model_loop, "_load_seed_model"):
            model_loop._load_seed_model(
                model_path=settings.seed_model_path,
                hypothesis_text=settings.seed_hypothesis,
            )
    else:
        model_loop = ModelRDLoop.load(path, checkout=checkout)

    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
