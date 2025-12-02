"""CLI commands for Docker cleanup operations."""

from typing import Annotated

import typer
from rdagent.utils.docker_cleanup import DockerCleanupManager
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Docker resource cleanup commands")
console = Console()


@app.command(name="all")
def cleanup_all(
    dangling: Annotated[
        bool, typer.Option("--dangling/--no-dangling", "-d/-D", help="Clean dangling images")
    ] = True,
    containers: Annotated[
        bool, typer.Option("--containers/--no-containers", "-c/-C", help="Clean stopped containers")
    ] = True,
    cache: Annotated[bool, typer.Option("--cache/--no-cache", "-b/-B", help="Clean build cache")] = False,
    images: Annotated[
        bool, typer.Option("--images/--no-images", "-i/-I", help="Clean RD-Agent images (local_*)")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Show what would be cleaned without actually cleaning")
    ] = False,
) -> None:
    """Clean up all Docker resources used by RD-Agent."""
    manager = DockerCleanupManager()

    if dry_run:
        console.print("[yellow]Dry run mode - showing current disk usage:[/yellow]")
        usage = manager.get_disk_usage()
        _print_disk_usage(usage)
        return

    results = manager.full_cleanup(
        dangling_images=dangling,
        stopped_containers=containers,
        build_cache=cache,
        rdagent_images=images,
    )

    _print_cleanup_results(results)
    console.print("[green]Cleanup complete![/green]")


@app.command(name="dangling")
def cleanup_dangling() -> None:
    """Clean up dangling (untagged) Docker images."""
    manager = DockerCleanupManager()
    result = manager.cleanup_dangling_images()
    space = result.get("SpaceReclaimed", 0)
    count = len(result.get("ImagesDeleted", []) or [])
    console.print(f"[green]Cleaned {count} dangling images, reclaimed {space / 1024 / 1024:.2f} MB[/green]")


@app.command(name="containers")
def cleanup_containers() -> None:
    """Clean up stopped Docker containers."""
    manager = DockerCleanupManager()
    result = manager.cleanup_stopped_containers()
    space = result.get("SpaceReclaimed", 0)
    count = len(result.get("ContainersDeleted", []) or [])
    console.print(f"[green]Cleaned {count} stopped containers, reclaimed {space / 1024 / 1024:.2f} MB[/green]")


@app.command(name="cache")
def cleanup_cache() -> None:
    """Clean up Docker build cache."""
    manager = DockerCleanupManager()
    result = manager.cleanup_build_cache()
    space = result.get("SpaceReclaimed", 0)
    console.print(f"[green]Cleaned build cache, reclaimed {space / 1024 / 1024:.2f} MB[/green]")


@app.command(name="images")
def cleanup_images(
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="Image name prefix to match")] = "local_",
) -> None:
    """Clean up RD-Agent Docker images (default: local_* images)."""
    manager = DockerCleanupManager()
    removed = manager.cleanup_rdagent_images(prefix=prefix)
    if removed:
        console.print(f"[green]Removed images: {', '.join(removed)}[/green]")
    else:
        console.print("[yellow]No matching images found[/yellow]")


@app.command(name="status")
def status() -> None:
    """Show Docker disk usage summary."""
    manager = DockerCleanupManager()
    usage = manager.get_disk_usage()
    _print_disk_usage(usage)


def _print_disk_usage(usage: dict) -> None:
    """Print disk usage in a formatted table."""
    table = Table(title="Docker Disk Usage")
    table.add_column("Type", style="cyan")
    table.add_column("Total", style="green")
    table.add_column("Active", style="yellow")
    table.add_column("Size", style="magenta")
    table.add_column("Reclaimable", style="red")

    for category in ["Images", "Containers", "Volumes", "BuildCache"]:
        data = usage.get(category, {})
        if isinstance(data, list):
            total = len(data)
            active = sum(1 for item in data if item.get("Active", False))
            size = sum(item.get("Size", 0) for item in data)
            reclaimable = sum(item.get("Size", 0) for item in data if not item.get("Active", False))
        elif isinstance(data, dict):
            total = data.get("TotalCount", 0)
            active = data.get("Active", 0)
            size = data.get("Size", 0)
            reclaimable = data.get("Reclaimable", 0)
        else:
            continue

        table.add_row(
            category,
            str(total),
            str(active),
            f"{size / 1024 / 1024:.2f} MB",
            f"{reclaimable / 1024 / 1024:.2f} MB",
        )

    console.print(table)


def _print_cleanup_results(results: dict) -> None:
    """Print cleanup results in a formatted table."""
    table = Table(title="Cleanup Results")
    table.add_column("Operation", style="cyan")
    table.add_column("Items Cleaned", style="green")
    table.add_column("Space Reclaimed", style="magenta")

    for op, result in results.items():
        if isinstance(result, dict):
            items = len(result.get("ImagesDeleted", []) or result.get("ContainersDeleted", []) or [])
            space = result.get("SpaceReclaimed", 0)
            table.add_row(op.replace("_", " ").title(), str(items), f"{space / 1024 / 1024:.2f} MB")
        elif isinstance(result, list):
            table.add_row(op.replace("_", " ").title(), str(len(result)), "-")

    console.print(table)
