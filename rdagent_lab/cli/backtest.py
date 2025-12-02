"""Backtest commands for lab."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rdagent_lab.services.backtest import BacktestService, BacktestConfig
from rdagent_lab.services.training import TrainingService

app = typer.Typer(help="Run backtests")
console = Console()


@app.command("qlib-workflow")
def backtest_workflow(config: Path = typer.Argument(..., help="Path to Qlib workflow YAML config")) -> None:
    """Run a Qlib workflow via RD-Agent lab."""
    service = TrainingService()
    try:
        with console.status("[bold green]Running Qlib workflow..."):
            result = service.train_with_qlib_workflow(config)
        console.print(f"[bold green]✓[/bold green] Workflow complete (recorder: {result['recorder_id']})")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]✗[/bold red] Workflow failed: {exc}")
        raise typer.Exit(1)


@app.command("vectorbt")
def backtest_vectorbt(
    predictions_path: Path = typer.Argument(..., help="Pickle/Parquet predictions with multi-index"),
    prices_path: Path = typer.Argument(..., help="CSV/Parquet prices aligned to predictions"),
    topk: int = typer.Option(30, "--topk", "-k", help="Top-k signals to hold"),
    threshold: float = typer.Option(0.0, "--threshold", "-t", help="Signal threshold"),
    report: Optional[Path] = typer.Option(None, "--report", "-r", help="Optional HTML report path"),
) -> None:
    """Quick VectorBT backtest for rapid iteration."""
    import pandas as pd

    preds = _load_series(predictions_path)
    prices = _load_frame(prices_path)
    service = BacktestService()
    config = BacktestConfig(engine="vectorbt", topk=topk, threshold=threshold, report_output=str(report) if report else None)
    try:
        with console.status("[bold green]Running VectorBT backtest..."):
            result = service.run_vectorbt(preds, prices, config)
        console.print(f"[bold green]✓[/bold green] Backtest complete. Sharpe: {result.metrics.get('sharpe', 0):.2f}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]✗[/bold red] Backtest failed: {exc}")
        raise typer.Exit(1)


def _load_series(path: Path):
    import pandas as pd

    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".parquet"}:
        return pd.read_parquet(path).squeeze()
    return pd.read_csv(path, index_col=[0, 1], parse_dates=True).squeeze()


def _load_frame(path: Path):
    import pandas as pd

    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)
