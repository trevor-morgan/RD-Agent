"""Training commands for lab."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rdagent_lab.services.training import TrainingService, TrainingConfig

app = typer.Typer(help="Train models with Qlib workflows")
console = Console()


@app.command("model")
def train_model(
    config: Optional[Path] = typer.Argument(None, help="Path to Qlib workflow YAML config"),
    model_type: str = typer.Option("lgbm", "--model", "-m", help="Model type (lgbm, transformer)"),
    features: str = typer.Option("Alpha158", "--features", "-f", help="Feature set (Alpha158, Alpha360)"),
    instruments: str = typer.Option("csi300", "--instruments", "-i", help="Instrument pool"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for model"),
    experiment_name: Optional[str] = typer.Option(None, "--name", "-n", help="Experiment name for tracking"),
) -> None:
    """Train a model using Qlib."""
    service = TrainingService()
    if config and config.exists():
        console.print(f"[bold blue]Training model from config:[/bold blue] {config}")
        try:
            with console.status("[bold green]Training in progress..."):
                result = service.train_from_yaml(config)
            _display_training_result(result)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]✗[/bold red] Training failed: {exc}")
            raise typer.Exit(1)
        return

    console.print(f"[bold blue]Training {model_type} model[/bold blue]")
    console.print(f"  Features: {features}")
    console.print(f"  Instruments: {instruments}")
    training_config = TrainingConfig(
        model_type=model_type,
        feature_config=features,
        instruments=instruments,
        experiment_name=experiment_name or f"rdagent_lab_{model_type}",
        save_path=str(output_dir / f"{model_type}_model.pkl") if output_dir else None,
    )
    try:
        with console.status("[bold green]Training in progress..."):
            result = service.train(training_config)
        _display_training_result(result)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]✗[/bold red] Training failed: {exc}")
        console.print("[dim]Ensure Qlib data is available at the configured path[/dim]")
        raise typer.Exit(1)


def _display_training_result(result) -> None:
    console.print("\n[bold green]✓[/bold green] Training complete!")
    table = Table(title="Training Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for metric, value in result.metrics.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))
    console.print(table)
    if result.save_path:
        console.print(f"\n[dim]Model saved to: {result.save_path}[/dim]")
    if result.feature_importance is not None and len(result.feature_importance) > 0:
        console.print("\n[bold]Top 10 Feature Importances:[/bold]")
        for feat, imp in result.feature_importance.head(10).items():
            console.print(f"  {feat}: {imp:.4f}")
