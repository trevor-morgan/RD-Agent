"""Research commands that call RD-Agent quant scenarios directly."""

from __future__ import annotations

import typer
from rich.console import Console

from rdagent_lab.research.rdagent_adapter import RDAgentAdapter, QuantRunConfig

app = typer.Typer(help="Invoke RD-Agent quant flows from the lab CLI")
console = Console()


@app.command("quant")
def run_quant(
    iterations: int = typer.Option(10, "--iterations", "-n", help="Number of evolution loops"),
    output_dir: str = typer.Option("rdagent_output", "--output-dir", "-o", help="Output directory"),
    path: str | None = typer.Option(None, "--path", help="Resume from saved loop path"),
    step_n: int | None = typer.Option(None, "--step-n", help="Number of steps to run per loop"),
    loop_n: int | None = typer.Option(None, "--loop-n", help="Override loop count"),
    all_duration: str | None = typer.Option(None, "--all-duration", help="Wall time budget"),
    checkout: bool = typer.Option(True, "--checkout/--no-checkout", help="Reload git state during resume"),
) -> None:
    """Run RD-Agent fin_quant loop directly as a library call."""
    adapter = RDAgentAdapter()
    cfg = QuantRunConfig(
        iterations=iterations,
        output_dir=output_dir,
        path=path,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
    )
    with console.status("[bold green]Running RD-Agent quant loop..."):
        result = adapter.run_quant(cfg)
    if result.success:
        console.print(f"[bold green]✓[/bold green] RD-Agent quant finished. Output: {result.output_dir}")
    else:
        console.print(f"[bold red]✗[/bold red] RD-Agent quant failed: {result.error}")
        raise typer.Exit(1)
