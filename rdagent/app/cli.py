"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv

NOTE: We call load_dotenv() before other imports to ensure environment variables
are loaded before BaseSettings classes initialize.
"""

from __future__ import annotations

import subprocess
from importlib.resources import path as rpath

from dotenv import load_dotenv

# Load .env BEFORE importing any rdagent modules that use BaseSettings.
# The ".env" argument ensures it loads from the current working directory.
load_dotenv(".env")

import typer  # noqa: E402
from rdagent.app.data_science.loop import main as data_science  # noqa: E402
from rdagent.app.general_model.general_model import (  # noqa: E402
    extract_models_and_implement as general_model,
)
from rdagent.app.qlib_rd_loop.factor import main as fin_factor  # noqa: E402
from rdagent.app.qlib_rd_loop.factor_from_report import (  # noqa: E402
    main as fin_factor_report,
)
from rdagent.app.qlib_rd_loop.model import main as fin_model  # noqa: E402
from rdagent.app.qlib_rd_loop.quant import main as fin_quant  # noqa: E402
from rdagent.app.utils.docker_cleanup_cli import app as cleanup_app  # noqa: E402
from rdagent.app.utils.health_check import health_check  # noqa: E402
from rdagent.app.utils.info import collect_info  # noqa: E402
from rdagent.log.mle_summary import grade_summary  # noqa: E402
from rdagent_lab.cli import app as lab_app  # noqa: E402

app = typer.Typer()


def ui(port: int = 19899, log_dir: str = "", debug: bool = False, data_science: bool = False) -> None:
    """
    start web app to show the log traces.
    """
    if data_science:
        with rpath("rdagent.log.ui", "dsapp.py") as app_path:
            cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
            subprocess.run(cmds, check=False)  # noqa: S603 - CLI entry point with trusted package paths
        return
    with rpath("rdagent.log.ui", "app.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
        if log_dir or debug:
            cmds.append("--")
        if log_dir:
            cmds.append(f"--log_dir={log_dir}")
        if debug:
            cmds.append("--debug")
        subprocess.run(cmds, check=False)  # noqa: S603 - CLI entry point with trusted package paths


def server_ui(port: int = 19899) -> None:
    """
    start web app to show the log traces in real time
    """
    import sys  # noqa: PLC0415 - lazy import to avoid sys overhead

    subprocess.run([sys.executable, "rdagent/log/server/app.py", f"--port={port}"], check=False)  # noqa: S603


def ds_user_interact(port: int = 19900) -> None:
    """
    start web app to show the log traces in real time
    """
    commands = ["streamlit", "run", "rdagent/log/ui/ds_user_interact.py", f"--server.port={port}"]
    subprocess.run(commands, check=False)  # noqa: S603


app.command(name="fin_factor")(fin_factor)
app.command(name="fin_model")(fin_model)
app.command(name="fin_quant")(fin_quant)
app.command(name="fin_factor_report")(fin_factor_report)
app.command(name="general_model")(general_model)
app.add_typer(lab_app, name="lab", help="RD-Agent Lab commands (Qlib workflows, backtests, research)")
app.add_typer(cleanup_app, name="cleanup", help="Docker resource cleanup commands")
app.command(name="data_science")(data_science)
app.command(name="grade_summary")(grade_summary)
app.command(name="ui")(ui)
app.command(name="server_ui")(server_ui)
app.command(name="health_check")(health_check)
app.command(name="collect_info")(collect_info)
app.command(name="ds_user_interact")(ds_user_interact)


if __name__ == "__main__":
    app()
