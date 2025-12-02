"""Typer-based CLI for lab commands."""

import typer

from rdagent_lab.cli import train, backtest, research, live

app = typer.Typer(help="RD-Agent Lab CLI")
app.add_typer(train.app, name="train", help="Train models")
app.add_typer(backtest.app, name="backtest", help="Run backtests")
app.add_typer(research.app, name="research", help="Run RD-Agent quant flows")
app.add_typer(live.app, name="live", help="Live/paper trading stubs")


__all__ = ["app"]
