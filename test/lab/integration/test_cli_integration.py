"""Integration tests for CLI commands.

These tests verify CLI commands work end-to-end.
Some tests require Qlib data, others can run without.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# --- Tests that don't require Qlib data ---


def test_main_cli_help() -> None:
    """Main CLI should show help without error."""
    from rdagent.app.cli import app

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "fin_factor" in result.output or "Usage" in result.output


def test_lab_cli_help() -> None:
    """Lab CLI should show help without error."""
    from rdagent_lab.cli import app

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "backtest" in result.output
    assert "research" in result.output


def test_lab_train_help() -> None:
    """Lab train subcommand should show help."""
    from rdagent_lab.cli import app

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "model" in result.output.lower()


def test_lab_backtest_help() -> None:
    """Lab backtest subcommand should show help."""
    from rdagent_lab.cli import app

    result = runner.invoke(app, ["backtest", "--help"])

    assert result.exit_code == 0


def test_lab_research_help() -> None:
    """Lab research subcommand should show help."""
    from rdagent_lab.cli import app

    result = runner.invoke(app, ["research", "--help"])

    assert result.exit_code == 0
    assert "quant" in result.output.lower()


def test_lab_live_help() -> None:
    """Lab live subcommand should show help."""
    from rdagent_lab.cli import app

    result = runner.invoke(app, ["live", "--help"])

    assert result.exit_code == 0


# --- Tests that require Qlib data (skipped if not available) ---


@pytest.mark.skipif(
    not pytest.importorskip("qlib", reason="Qlib not installed"),
    reason="Qlib required",
)
def test_lab_train_model_missing_data_error() -> None:
    """Should show helpful error when Qlib data is missing."""
    from rdagent_lab.cli import app

    # Use a non-existent path to trigger error
    result = runner.invoke(
        app,
        [
            "train",
            "model",
            "--model",
            "lgbm",
            "--qlib-uri",
            "/nonexistent/path/to/qlib/data",
        ],
    )

    # Should fail but with a meaningful error
    assert result.exit_code != 0 or "not found" in result.output.lower() or "error" in result.output.lower()
