"""Thin adapter to call RD-Agent quant scenarios from within the monorepo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class QuantRunResult:
    success: bool
    scenario: str
    iterations_completed: int | None = None
    output_dir: str | None = None
    error: str | None = None
    raw: Any = None


@dataclass
class QuantRunConfig:
    scenario: str = "fin_quant"
    iterations: int = 10
    output_dir: str = "rdagent_output"
    env: dict[str, str] = field(default_factory=dict)
    path: str | None = None
    step_n: int | None = None
    loop_n: int | None = None
    all_duration: str | None = None
    checkout: bool = True


class RDAgentAdapter:
    """Directly invoke RD-Agent quant workflows without subprocesses."""

    def __init__(self, base_env: dict[str, str] | None = None) -> None:
        self.base_env = base_env or {}

    def run_quant(self, config: QuantRunConfig | None = None) -> QuantRunResult:
        from rdagent.app.qlib_rd_loop import quant

        cfg = config or QuantRunConfig()
        merged_env = os.environ.copy()
        merged_env.update(self.base_env)
        merged_env.update(cfg.env)

        output_dir = Path(cfg.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_env["RDAGENT_OUTPUT_DIR"] = str(output_dir)

        try:
            quant.main(
                path=cfg.path,
                step_n=cfg.step_n,
                loop_n=cfg.loop_n or cfg.iterations,
                all_duration=cfg.all_duration,
                checkout=cfg.checkout,
            )
            return QuantRunResult(
                success=True,
                scenario=cfg.scenario,
                iterations_completed=cfg.loop_n or cfg.iterations,
                output_dir=str(output_dir),
                raw=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"RD-Agent run failed: {exc}")
            return QuantRunResult(
                success=False,
                scenario=cfg.scenario,
                iterations_completed=None,
                output_dir=str(output_dir),
                error=str(exc),
            )
