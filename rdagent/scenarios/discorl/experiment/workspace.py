# Copyright 2025 Trevor Morgan
# SPDX-License-Identifier: Apache-2.0

"""DiscoRL Workspace for executing optimal execution experiments."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger


class DiscoRLFBWorkspace(FBWorkspace):
    """File-based workspace for DiscoRL optimal execution experiments.

    This workspace:
    1. Injects generated code (config, environment modifications)
    2. Executes training using DiscoRL's Disco103 algorithm
    3. Parses and returns execution results
    """

    def __init__(
        self,
        template_folder_path: Path | None = None,
        disco_rl_path: Path | None = None,
        *args,
        **kwargs,
    ):
        """Initialize DiscoRL workspace.

        Args:
            template_folder_path: Path to template code directory
            disco_rl_path: Path to disco_rl installation (for imports)
        """
        super().__init__(*args, **kwargs)

        # Path to disco_rl repo
        self.disco_rl_path = disco_rl_path or Path(
            os.environ.get("DISCO_RL_PATH", "~/repos/research/disco_rl")
        ).expanduser()

        # Inject template code if provided
        if template_folder_path and template_folder_path.exists():
            self.inject_code_from_folder(template_folder_path)

        # Results storage
        self.execution_result: dict[str, Any] | None = None

    def prepare(self) -> None:
        """Prepare workspace directory and files."""
        super().prepare()

        # Create results directory
        results_dir = self.workspace_path / "results"
        results_dir.mkdir(exist_ok=True)

        # Create symlink to disco_rl if needed
        disco_link = self.workspace_path / "disco_rl"
        if not disco_link.exists() and self.disco_rl_path.exists():
            disco_src = self.disco_rl_path / "disco_rl"
            if disco_src.exists():
                disco_link.symlink_to(disco_src)

    def execute(
        self,
        entry: str = "train.py",
        timeout: int = 3600,
        run_env: dict | None = None,
        **kwargs,
    ) -> str:
        """Execute the DiscoRL training experiment.

        Args:
            entry: Entry point script (default: train.py)
            timeout: Execution timeout in seconds
            run_env: Additional environment variables

        Returns:
            Execution output as string
        """
        self.prepare()

        # Inject all code files
        for filename, content in self.file_dict.items():
            file_path = self.workspace_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(content, str):
                file_path.write_text(content)
            elif isinstance(content, bytes):
                file_path.write_bytes(content)

        # Build execution environment
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.workspace_path}:{self.disco_rl_path}:{env.get('PYTHONPATH', '')}"

        if run_env:
            env.update(run_env)

        # Execute training script
        cmd = [sys.executable, str(self.workspace_path / entry)]
        logger.info(f"Executing DiscoRL: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n\nSTDERR:\n{result.stderr}"

            # Parse results if available
            results_file = self.workspace_path / "results" / "execution_result.json"
            if results_file.exists():
                with open(results_file) as f:
                    self.execution_result = json.load(f)
                output += f"\n\nParsed Results:\n{json.dumps(self.execution_result, indent=2)}"

            return output

        except subprocess.TimeoutExpired:
            return f"ERROR: Execution timed out after {timeout} seconds"
        except Exception as e:
            return f"ERROR: Execution failed: {e}"

    def get_result(self) -> dict[str, Any] | None:
        """Get parsed execution result."""
        if self.execution_result:
            return self.execution_result

        # Try to load from file
        results_file = self.workspace_path / "results" / "execution_result.json"
        if results_file.exists():
            with open(results_file) as f:
                self.execution_result = json.load(f)
                return self.execution_result

        return None

    @property
    def all_codes(self) -> str:
        """Return all code files as a single string."""
        parts = []
        for filename, content in sorted(self.file_dict.items()):
            if isinstance(content, str) and filename.endswith(".py"):
                parts.append(f"# === {filename} ===\n{content}")
        return "\n\n".join(parts)

    def copy(self) -> DiscoRLFBWorkspace:
        """Create a copy of this workspace."""
        new_ws = DiscoRLFBWorkspace(disco_rl_path=self.disco_rl_path)
        new_ws.file_dict = self.file_dict.copy()
        new_ws.execution_result = self.execution_result
        return new_ws
