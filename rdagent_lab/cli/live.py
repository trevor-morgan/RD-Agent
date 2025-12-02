"""Placeholder live/paper trading commands."""

import typer
from rich.console import Console

app = typer.Typer(help="Live trading commands (stub)")
console = Console()


@app.command("status")
def status() -> None:
    """Show live trading status."""
    console.print("[yellow]Live trading orchestration not yet implemented in the lab monorepo.[/yellow]")


@app.command("start")
def start(
    broker: str = typer.Option("paper", "--broker", "-b"),
    strategy: str = typer.Option("topk", "--strategy", "-s"),
) -> None:
    """Start live/paper trading."""
    console.print(
        "[yellow]Live trading start stub.[/yellow] "
        "Wire this to your execution stack after consolidating broker adapters."
    )
