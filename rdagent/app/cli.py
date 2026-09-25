"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import sys
import os
import re
import shutil
import socket
from pathlib import Path

def _check_required_env_files() -> None:
    in_docker = Path("/.dockerenv").exists()
    config_home = Path("/root") if in_docker else Path.home()
    project_dir = Path(__file__).parent.parent.parent

    missing = []
    if not (config_home / ".config.env").exists() and not Path(".config.env").exists() and not (project_dir / "config.env").exists():
        missing.append(f"{config_home / '.config.env'} or config.env in project directory ({project_dir})")
    if not (config_home / ".secrets.env").exists() and not Path(".secrets.env").exists() and not (project_dir / "secrets.env").exists():
        missing.append(f"{config_home / '.secrets.env'} or secrets.env in project directory ({project_dir})")
    if missing:
        print("ERROR: Required configuration files not found:")
        for m in missing:
            print(f"  - {m}")
        print("Please mount your .config.env and .secrets.env files before starting the agent.")
        sys.exit(1)

_check_required_env_files()

from dotenv import load_dotenv

def _find_env_file(name: str, dotname: str) -> Path:
    """Find env file checking home, cwd, and project root."""
    in_docker = Path("/.dockerenv").exists()
    config_home = Path("/root") if in_docker else Path.home()
    project_dir = Path(__file__).parent.parent.parent

    if (config_home / dotname).exists():
        return config_home / dotname
    if Path(dotname).exists():
        return Path(dotname)
    return project_dir / name  # no-dot variant in project root

load_dotenv(_find_env_file("config.env", ".config.env"))
load_dotenv(_find_env_file("secrets.env", ".secrets.env"))

import subprocess
from importlib.resources import path as rpath
from typing import Optional

import typer
from typing_extensions import Annotated

from rdagent.app.data_science.loop import main as data_science
from rdagent.app.utils.health_check import health_check
from rdagent.app.utils.info import collect_info
from rdagent.core.conf import RD_AGENT_SETTINGS, get_run_timestamp, get_safe_cwd
from rdagent.log.submission_summary import grade_summary as grade_summary

app = typer.Typer()

CheckoutOption = Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")]
CheckEnvOption = Annotated[bool, typer.Option("--check-env/--no-check-env", "-e/-E")]
CheckDockerOption = Annotated[bool, typer.Option("--check-docker/--no-check-docker", "-d/-D")]
CheckPortsOption = Annotated[bool, typer.Option("--check-ports/--no-check-ports", "-p/-P")]

_TIMESTAMPED_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+$")
_UI_RUN_MARKER_NAME = ".rdagent_ui_run_owner"


def _ensure_ui_run_marker(ui_run_dir: Path, run_dir_existed_before_launch: bool) -> bool:
    """Create an ownership marker for folders created by this UI invocation."""
    if run_dir_existed_before_launch:
        return False
    try:
        ui_run_dir.mkdir(parents=True, exist_ok=True)
        marker_path = ui_run_dir / _UI_RUN_MARKER_NAME
        marker_path.write_text("ui-owner\n", encoding="utf-8")
        return True
    except Exception:
        return False


def _cleanup_ui_run_folders_after_close(
    ui_run_id: str,
    log_run_dir_existed_before_launch: bool,
    workspace_run_dir_existed_before_launch: bool,
    marker_created: bool,
) -> None:
    """Delete only this UI invocation's owned run folders on close."""
    enabled = os.environ.get("RD_AGENT_UI_CLEAR_LOGS_ON_CLOSE", "false").lower() == "true"
    if not enabled:
        return

    if log_run_dir_existed_before_launch and workspace_run_dir_existed_before_launch:
        return

    if not marker_created:
        return

    log_root = get_safe_cwd() / "log"
    workspace_root = Path(RD_AGENT_SETTINGS.workspace_path)

    log_run_dir = log_root / ui_run_id
    marker_path = log_run_dir / _UI_RUN_MARKER_NAME
    if marker_path.exists() and log_run_dir.is_dir() and _TIMESTAMPED_RUN_DIR_RE.match(log_run_dir.name):
        try:
            shutil.rmtree(log_run_dir, ignore_errors=True)
        except Exception:
            pass

    workspace_run_dir = workspace_root / ui_run_id
    if workspace_run_dir.is_dir() and _TIMESTAMPED_RUN_DIR_RE.match(workspace_run_dir.name):
        try:
            shutil.rmtree(workspace_run_dir, ignore_errors=True)
        except Exception:
            pass


def _cleanup_empty_timestamp_folders():
    """Clean up empty timestamped dump folders in log and workspace directories."""
    import atexit
    import signal
    
    # System/metadata folders that don't contain real results
    IGNORE_SYSTEM_DIRS = {
        "GOOGLE_AI_STUDIO_SETTINGS",
        "RD_AGENT_SETTINGS",
        "RDLOOP_SETTINGS",
        "token_cost",
        "debug_llm",
    }
    
    def _do_cleanup():
        try:
            log_root = get_safe_cwd() / "log"
            workspace_root = Path(RD_AGENT_SETTINGS.workspace_path)
            
            for root_dir in [log_root, workspace_root]:
                if not root_dir.exists():
                    continue
                try:
                    for run_dir in list(root_dir.iterdir()):
                        if not (run_dir.is_dir() and _TIMESTAMPED_RUN_DIR_RE.match(run_dir.name)):
                            continue
                        # Check if folder only contains hidden files/folders or ignored system dirs
                        try:
                            contents = list(run_dir.iterdir())
                            if not contents:  # Completely empty
                                shutil.rmtree(run_dir, ignore_errors=True)
                            else:
                                has_real_content = any(
                                    not item.name.startswith(".") and item.name not in IGNORE_SYSTEM_DIRS
                                    for item in contents
                                )
                                if not has_real_content:  # Only hidden files/folders or system dirs
                                    shutil.rmtree(run_dir, ignore_errors=True)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
    
    # Register with atexit for normal process exit
    atexit.register(_do_cleanup)
    
    # Register signal handlers for termination signals
    def _signal_handler(signum, frame):
        _do_cleanup()
        sys.exit(1)
    
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass


_cleanup_empty_timestamp_folders()


def ui(port=19899, log_dir="", data_science: bool = True):
    """
    start the data-science web app to show the log traces.
    `--data-science` is kept for compatibility; the data-science app is the only UI.
    """
    # UI-only command should not trigger unrelated run-folder cleanup at process exit.
    os.environ["RD_AGENT_CLEANUP_DEBUG_RUNS_ON_EXIT"] = "false"
    os.environ["RD_AGENT_CLEANUP_DEBUG_RUNS_ON_DOCKER_STOP"] = "false"
    os.environ["RD_AGENT_ENABLE_RUN_FOLDER_CLEANUP"] = "false"

    selected_port = int(port)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", selected_port))
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            selected_port = s.getsockname()[1]
        typer.echo(f"Port {port} is not available, using {selected_port}.")

    ui_run_id = get_run_timestamp()
    ui_run_dir = get_safe_cwd() / "log" / ui_run_id
    workspace_run_dir = Path(RD_AGENT_SETTINGS.workspace_path) / ui_run_id
    log_run_dir_existed_before_launch = ui_run_dir.exists()
    workspace_run_dir_existed_before_launch = workspace_run_dir.exists()
    marker_created = _ensure_ui_run_marker(ui_run_dir, log_run_dir_existed_before_launch)

    with rpath("rdagent.log.ui", "dsapp.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={selected_port}"]
        if log_dir:
            cmds.append("--")
            cmds.append(f"--log_dir={log_dir}")
        try:
            subprocess.run(cmds)
        finally:
            _cleanup_ui_run_folders_after_close(
                ui_run_id,
                log_run_dir_existed_before_launch,
                workspace_run_dir_existed_before_launch,
                marker_created,
            )


@app.command(name="data_science")
def data_science_cli(
    path: Optional[str] = None,
    checkout: CheckoutOption = True,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
    competition: Optional[str] = None,
):
    data_science(
        path=path,
        checkout=checkout,
        step_n=step_n,
        loop_n=loop_n,
        timeout=timeout,
        competition=competition,
    )


@app.command(name="grade_summary")
def grade_summary_cli(log_folder: str):
    grade_summary(log_folder)


app.command(name="ui")(ui)


@app.command(name="health_check")
def health_check_cli(
    check_env: CheckEnvOption = True,
    check_docker: CheckDockerOption = True,
    check_ports: CheckPortsOption = True,
):
    health_check(check_env=check_env, check_docker=check_docker, check_ports=check_ports)


@app.command(name="collect_info")
def collect_info_cli():
    collect_info()


if __name__ == "__main__":
    app()
