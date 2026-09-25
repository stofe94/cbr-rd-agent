import asyncio
import os
from pathlib import Path
from typing import Optional

import fire
from loguru import logger as loguru_logger

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


DEFAULT_DS_RD_LOOP = "rdagent.scenarios.data_science.loop.DataScienceRDLoop"

# Keep a plain terminal log for the full data-science run so CBR and proposal
# logs remain available even when the start script does not tee stdout/stderr.
loguru_logger.add("terminal.log", level="DEBUG")


def _ensure_runtime_dirs() -> None:
    """Create required runtime directories if they are missing."""
    # Default log root used by RD-Agent logging/session dumps.
    (Path.cwd() / "log").mkdir(parents=True, exist_ok=True)

    # Data science workspace root configured from env (DS_LOCAL_DATA_PATH).
    local_data_path = getattr(DS_RD_SETTING, "local_data_path", None)
    if local_data_path:
        Path(local_data_path).mkdir(parents=True, exist_ok=True)

    # Only create workspace if DS_LOCAL_DATA_PATH points to it
    if local_data_path and "workspace" in Path(local_data_path).parts:
        Path.cwd().joinpath("workspace").mkdir(parents=True, exist_ok=True)


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _env_str(name: str) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def main(
    path: Optional[str] = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
    competition="bms-molecular-translation",
    replace_timer=True,
    exp_gen_cls: Optional[str] = None,
):
    """

    Parameters
    ----------
    path :
        A path like `$LOG_PATH/__session__/1/0_propose`. This indicates that we restore the state after finishing step 0 in loop 1.
    checkout :
        Used to control the log session path. Boolean type, default is True.
        - If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
        - If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
    checkout_path:
        If a checkout_path (or a str like Path) is provided, the new loop will be saved to that path, leaving the original path unchanged.
    step_n :
        Number of steps to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs.
    loop_n :
        Number of loops to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs.
        - If the current loop is incomplete, it will be counted as the first loop for completion.
        - If both step_n and loop_n are provided, the process will stop as soon as either condition is met.
    timeout :
        Maximum duration to run the loop. Accepts a string format recognized by the internal timer.
        - If None, the loop will run until completion, error, or KeyboardInterrupt.
    competition :
        Competition name.
    replace_timer :
        If a session is loaded, determines whether to replace the timer with session.timer.
    exp_gen_cls :
        When there are different stages, the exp_gen can be replaced with the new proposal.


    Auto R&D Evolving loop for models in a Kaggle scenario.
    You can continue running a session by using the command:

    .. code-block:: bash

      dotenv run -- python rdagent/app/data_science/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is an optional parameter
      rdagent kaggle --competition playground-series-s4e8  # This command is recommended.
    """
    if not checkout_path is None:
        checkout = Path(checkout_path)

    _ensure_runtime_dirs()

    if competition is not None:
        DS_RD_SETTING.competition = competition

    if not DS_RD_SETTING.competition:
        logger.error("Please specify competition name.")

    rd_loop_cls = import_class(os.environ.get("DS_RD_LOOP", DEFAULT_DS_RD_LOOP))

    if path is None:
        kaggle_loop = rd_loop_cls(DS_RD_SETTING)
    else:
        kaggle_loop: DataScienceRDLoop = rd_loop_cls.load(path, checkout=checkout, replace_timer=replace_timer)

    # replace exp_gen if we have new class
    if exp_gen_cls is not None:
        kaggle_loop.exp_gen = import_class(exp_gen_cls)(kaggle_loop.exp_gen.scen)

    # Allow .env defaults for run limits when CLI flags are omitted.
    if step_n is None:
        step_n = _env_int("DS_STEP_N")
    if loop_n is None:
        loop_n = _env_int("DS_LOOP_N")
    if timeout is None:
        timeout = _env_str("DS_TIMEOUT")

    try:
        asyncio.run(kaggle_loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user (Ctrl+C). Exiting safely...")
        try:
            kaggle_loop.close_pbar()
        except Exception:
            pass
        # Cancel all running asyncio tasks to suppress noisy warnings
        try:
            loop = asyncio.get_event_loop()
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()
        except Exception:
            pass
        return


if __name__ == "__main__":
    fire.Fire(main)
