"""
The motivation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder
"""

# TODO: move the scenario specific docker env into other folders.

import contextlib
import json
import os
import pickle
import re
import select
import shutil
import subprocess
import time
import uuid
import zipfile
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Generic,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    cast,
)

import docker  # type: ignore[import-untyped]
import docker.models  # type: ignore[import-untyped]
import docker.models.containers  # type: ignore[import-untyped]
import docker.types  # type: ignore[import-untyped]
from pydantic import BaseModel, model_validator
from pydantic_settings import SettingsConfigDict
from rich import print
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from rdagent.core.conf import ExtendedBaseSettings, get_run_timestamp, get_safe_cwd
from rdagent.core.experiment import RD_AGENT_SETTINGS, cleanup_timestamped_run_folders_on_docker_stop
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils import filter_redundant_text
from rdagent.utils.agent.tpl import T
from rdagent.utils.fmt import shrink_text
from rdagent.utils.workflow import wait_retry

CacheKeyFunc = Callable[[str | Path], list[list[str]]]
_PIPELINE_WARNING_EMITTED = False


def _has_shell_pipeline(entry: str) -> bool:
    """Return True when `entry` contains a shell pipeline operator (`|`).

    This intentionally excludes logical OR (`||`), which is not a pipeline.
    """
    return re.search(r"(?<!\|)\|(?!\|)", entry) is not None


def _is_internal_entry_wrapper(entry: str | None) -> bool:
    """Return True for internal shell wrapper commands that should not be shown."""
    if not entry:
        return False
    return (
        entry.startswith("/bin/sh -c '")
        and "entry_exit_code=$?" in entry
        and "exit $entry_exit_code" in entry
    )


def _print_entry_prompt(entry: str | None) -> None:
    """Print user-facing command prompt unless it is an internal wrapper."""
    if entry and not _is_internal_entry_wrapper(entry):
        Console().print(f"[bold yellow]$ {entry}[/bold yellow]", markup=True)


def _docker_client_from_env() -> docker.DockerClient:
    """Create a Docker client, falling back to a local config when credential helper fails.

    In WSL setups, docker-py can fail while invoking docker-credential-desktop.exe.
    This fallback bypasses external credential helpers for public/local images.
    """
    try:
        return docker.from_env()
    except Exception as e:
        err = str(e)
        cred_helper_error = (
            "docker-credential-desktop" in err
            or "docker-credential-desktop.exe" in err
            or "Credentials store" in err
            or "StoreError" in err
        )
        if not cred_helper_error:
            raise

        fallback_dir = get_safe_cwd() / ".rdagent_docker_config"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_cfg = fallback_dir / "config.json"
        if not fallback_cfg.exists():
            fallback_cfg.write_text(json.dumps({"auths": {}}, indent=2), encoding="utf-8")

        os.environ["DOCKER_CONFIG"] = str(fallback_dir)
        # Ignore any injected auth payload that points to an unavailable credential store.
        os.environ.pop("DOCKER_AUTH_CONFIG", None)

        logger.warning(
            "Docker credential helper failed. Falling back to local DOCKER_CONFIG at %s",
            fallback_dir,
        )
        return docker.from_env()


def extract_dir_name_from_path_config(path_str: str) -> str:
    """
    Extract the first directory component from a relative path string.

    This is used to get the basename from path configurations like "./workspace_input/"
    to use in chmod exclusion patterns.

    Args:
        path_str: A path string, typically from T() template configuration

    Returns:
        The first directory component, or empty string if not a relative path

    Examples:
        "./workspace_input/" -> "workspace_input"
        "./assets/" -> "assets"
        "/absolute/path" -> ""
    """
    p = Path(path_str)
    if not p.is_absolute() and p.parts:
        return p.parts[0]
    return ""


def cleanup_container(container: docker.models.containers.Container | None, context: str = "") -> None:  # type: ignore[no-any-unimported]
    """
    Shared helper function to clean up a Docker container.
    Always stops the container before removing it.

    Parameters
    ----------
    container : docker container object or None
        The container to clean up, or None if no container to clean up
    context : str
        Additional context for logging (e.g., "health check", "GPU test")
    """
    if container is not None:
        try:
            # Always stop first - stop() doesn't raise error if already stopped
            container.stop(timeout=5)
            container.remove(force=True)  # force=True ensures it's removed even if stop fails
        except Exception as cleanup_error:
            # Log cleanup error but don't mask the original exception
            context_str = f" {context}" if context else ""
            logger.warning(f"Failed to cleanup{context_str} container {container.id}: {cleanup_error}")


# Normalize all bind paths in volumes to absolute paths using the workspace (working_dir).
def normalize_volumes(vols: dict[str, str | dict[str, str]], working_dir: str) -> dict:
    abs_vols: dict[str, str | dict[str, str]] = {}

    def to_abs(path: str) -> str:
        # Converts a relative path to an absolute path using the workspace (working_dir).
        return os.path.abspath(os.path.join(working_dir, path)) if not os.path.isabs(path) else path

    for lp, vinfo in vols.items():
        # Support both:
        # 1. {'host_path': {'bind': 'container_path', ...}}
        # 2. {'host_path': 'container_path'}
        if isinstance(vinfo, dict):
            # abs_vols = cast(dict[str, dict[str, str]], abs_vols)
            vinfo = vinfo.copy()
            vinfo["bind"] = to_abs(vinfo["bind"])
            abs_vols[lp] = vinfo
        else:
            # abs_vols = cast(dict[str, str], abs_vols)
            abs_vols[lp] = to_abs(vinfo)
    return abs_vols


def pull_image_with_progress(image: str) -> None:
    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    pull_logs = client.pull(image, stream=True, decode=True)
    progress_bars = {}

    for log in pull_logs:
        if "id" in log and log.get("progressDetail"):
            layer_id = log["id"]
            progress_detail = log["progressDetail"]
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)

            if total:
                if layer_id not in progress_bars:
                    progress_bars[layer_id] = tqdm(total=total, desc=f"Layer {layer_id}", unit="B", unit_scale=True)
                progress_bars[layer_id].n = current
                progress_bars[layer_id].refresh()

        elif "status" in log:
            print(log["status"])

    for pb in progress_bars.values():
        pb.close()


class EnvConf(ExtendedBaseSettings):
    default_entry: str
    env_dict: dict = {}
    extra_volumes: dict = {}
    running_timeout_period: int | None = 3600  # 10 minutes

    """it is a function to calculating hash keys"""

    def get_workspace_content_for_hash(self, local_path: str | Path) -> list[list[str]]:
        """Get content of key files in workspace for cache hash calculation.

        Scans .py, .csv, and .yaml files.
        """
        # we must add the information of data (beyond code) into the key.
        # Otherwise, all commands operating on data will become invalid (e.g. rm -r submission.csv)
        # So we recursively walk in the folder and add the sorted relative filename list as part of the key.
        # data_key = []
        # for path in Path(local_path).rglob("*"):
        #     p = str(path.relative_to(Path(local_path)))
        #     if p.startswith("__pycache__"):
        #         continue
        #     data_key.append(p)
        # data_key = sorted(data_key)
        local_path = Path(local_path)
        return [
            [str(path.relative_to(local_path)), path.read_text()]
            for path in sorted(
                list(local_path.rglob("*.py")) + list(local_path.rglob("*.csv")) + list(local_path.rglob("*.yaml"))
            )
        ]

    redirect_stdout_to_file: bool = False
    # helper settings to support transparent;
    enable_cache: bool = True
    retry_count: int = 5  # retry count for the docker run
    retry_wait_seconds: int = 10  # retry wait seconds for the docker run
    exclude_chmod_paths: list[str] = []  # List of directory names to exclude from chmod operation

    model_config = SettingsConfigDict(
        # TODO: add prefix ....
        env_parse_none_str="None",  # Nthis is the key to accept `RUNNING_TIMEOUT_PERIOD=None`
    )


ASpecificEnvConf = TypeVar("ASpecificEnvConf", bound=EnvConf)


@dataclass
class EnvResult:
    """
    The result of running the environment.
    It contains the stdout, the exit code, and the running time in seconds.
    """

    full_stdout: str
    exit_code: int
    running_time: float
    stored_full_stdout_to_truncated_stdout: Dict[str, str]

    def __init__(self, stdout: str, exit_code: int, running_time: float):
        self.full_stdout = stdout
        self.exit_code = exit_code
        self.running_time = running_time
        self.stored_full_stdout_to_truncated_stdout = {}

    def update_stdout(self, stdout: str) -> None:
        self.full_stdout = stdout

    @property
    def stdout(self) -> str:
        if self.full_stdout not in self.stored_full_stdout_to_truncated_stdout:
            truncated: str = self._get_truncated_stdout(self.full_stdout)
            self.stored_full_stdout_to_truncated_stdout[self.full_stdout] = truncated
        return self.stored_full_stdout_to_truncated_stdout[self.full_stdout]

    def hash_full_stdout(self, full_stdout: str) -> str:
        return md5_hash(full_stdout)

    @cache_with_pickle(hash_full_stdout)
    def _get_truncated_stdout(self, full_stdout: str) -> str:
        return shrink_text(
            filter_redundant_text(full_stdout),
            context_lines=RD_AGENT_SETTINGS.stdout_context_len,
            line_len=RD_AGENT_SETTINGS.stdout_line_len,
        )


class Env(Generic[ASpecificEnvConf]):
    """
    We use BaseModel as the setting due to the features it provides
    - It provides base typing and checking features.
    - loading and dumping the information will be easier: for example, we can use package like `pydantic-yaml`
    """

    conf: ASpecificEnvConf  # different env have different conf.

    def __init__(self, conf: ASpecificEnvConf):
        self.conf = conf

    def zip_a_folder_into_a_file(self, folder_path: str, zip_file_path: str) -> None:
        """
        Zip a folder into a file, use zipfile instead of subprocess
        """
        with zipfile.ZipFile(zip_file_path, "w") as z:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    z.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), folder_path),
                    )

    def unzip_a_file_into_a_folder(
        self, zip_file_path: str, folder_path: str, files_to_extract: list[str] | None = None
    ) -> None:
        """
        Unzip a file into a folder, use zipfile instead of subprocess
        """
        if files_to_extract is None:
            # Clear folder_path before extracting
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, "r") as z:
            if files_to_extract is not None:
                for file_name in files_to_extract:
                    try:
                        z.extract(file_name, folder_path)
                    except KeyError:
                        logger.warning(f"File {file_name} not found in cache zip.")
            else:
                z.extractall(folder_path)

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Prepare for the environment based on it's configure
        """

    def check_output(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> str:
        result = self.run(
            entry=entry,
            local_path=local_path,
            env=env,
            running_extra_volume=running_extra_volume,
            cache_key_extra_func=cache_key_extra_func,
            cache_files_to_extract=cache_files_to_extract,
        )
        return result.stdout

    def __run_with_retry(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        for retry_index in range(self.conf.retry_count + 1):
            try:
                start = time.time()
                log_output, return_code = self._run(
                    entry,
                    local_path,
                    env,
                    running_extra_volume=running_extra_volume,
                )
                end = time.time()
                logger.debug(f"Running time: {end - start} seconds")
                if self.conf.running_timeout_period is not None and end - start + 1 >= self.conf.running_timeout_period:
                    logger.warning(
                        f"The running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."
                    )
                    log_output += f"\n\nThe running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."
                return EnvResult(log_output, return_code, end - start)
            except Exception as e:
                if retry_index == self.conf.retry_count:
                    raise
                logger.warning(
                    f"Error while running the container: {e}, current try index: {retry_index + 1}, {self.conf.retry_count - retry_index - 1} retries left."
                )
                time.sleep(self.conf.retry_wait_seconds)
        raise RuntimeError  # for passing CI

    def run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> EnvResult:
        """
        Run the folder under the environment and return the stdout, exit code, and running time.

        Parameters
        ----------
        entry : str | None
            We may we the entry point when we run it.
            For example, we may have different entries when we run and summarize the project.
        local_path : str | None
            the local path (to project, mainly for code) will be mounted into the docker
            Here are some examples for a None local path
            - for example, run docker for updating the data in the extra_volumes.
            - simply run the image. The results are produced by output or network
        env : dict | None
            Run the code with your specific environment.
        running_extra_volume : Mapping
            Extra volumes to mount during execution.
        cache_key_extra_func : CacheKeyFunc | None
            Optional function to calculate extra information for cache key calculation
        cache_files_to_extract : list[str] | None
            Optional list of files to extract from cache zip. If None, extract all.

        Returns
        -------
            EnvResult: An object containing the stdout, the exit code, and the running time in seconds.
        """
        _env = self.conf.env_dict.copy()
        if env:
            _env.update(env)
        env = _env

        if entry is None:
            entry = self.conf.default_entry

        global _PIPELINE_WARNING_EMITTED
        if _has_shell_pipeline(entry) and not _PIPELINE_WARNING_EMITTED:
            # logger.warning(
            #     "You are using a command with a shell pipeline (i.e., '|'). "
            #     "The exit code ($exit_code) will reflect the result of "
            #     "the last command in the pipeline.",
            # )
            _PIPELINE_WARNING_EMITTED = True

        # Exclude configured directories from chmod operation to prevent modifying
        # read-only or specially configured directories that may produce warnings.
        def _get_chmod_cmd(workspace_path: str) -> str:
            find_cmd = f"find {workspace_path} -mindepth 1 -maxdepth 1"

            # Use configurable exclude paths from DockerConf
            for name in self.conf.exclude_chmod_paths:
                if name:  # Skip empty names
                    find_cmd += f" ! -name {name}"

            chmod_cmd = f"{find_cmd} -exec chmod -R 777 {{}} +"
            return chmod_cmd

        if self.conf.redirect_stdout_to_file:
            log_file_name = md5_hash(entry)[:8] + ".log"
            log_file = Path(local_path) / f"{log_file_name}"
            log_file_relative_path = log_file.relative_to(Path(local_path))
            entry = f"{entry} > {log_file_relative_path} 2>&1"

        if self.conf.running_timeout_period is None:
            timeout_cmd = entry
        else:
            timeout_cmd = f"timeout --kill-after=10 {self.conf.running_timeout_period} {entry}"

        def _escape_for_sh_single_quotes(script: str) -> str:
            # The command is executed as `/bin/sh -c '<script>'`. Any single quote inside
            # <script> would terminate the quoted string and break parsing.
            # This is the standard safe escape sequence for embedding a literal `'`.
            return script.replace("'", "'\\''")

        script_body = (
            f"{timeout_cmd}\nentry_exit_code=$?; "
            + (
                f"{_get_chmod_cmd(self.conf.mount_path)}; "
                if isinstance(self.conf, DockerConf)
                else ""
            )
            + "exit $entry_exit_code"
        )
        entry_add_timeout = (
            f"/bin/sh -c '"  # start of the sh command
            + _escape_for_sh_single_quotes(script_body)
            + "'"  # end of the sh command
        )

        if self.conf.enable_cache:
            result = self.cached_run(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
                cache_key_extra_func,
                cache_files_to_extract,
            )
        else:
            result = self.__run_with_retry(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
            )
        if self.conf.redirect_stdout_to_file:
            stdout = log_file.read_text(errors="replace")
            log_file.unlink(missing_ok=True)
            result.update_stdout(stdout)
        if str(Path(local_path).resolve()) in result.stdout:
            result.update_stdout(result.stdout.replace(str(Path(local_path).resolve()), "<WORKSPACE_PATH>"))

        return result

    def cached_run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> EnvResult:
        """
        Run the folder under the environment.
        Will cache the output and the folder diff for next round of running.
        Use the python codes and the parameters(entry, running_extra_volume) as key to hash the input.
        """
        target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"utils.env.run"
        target_folder.mkdir(parents=True, exist_ok=True)

        if cache_key_extra_func is not None:
            cache_key_extra = cache_key_extra_func(local_path)
        else:
            cache_key_extra = self.conf.get_workspace_content_for_hash(local_path)

        key = md5_hash(
            json.dumps(cache_key_extra)
            + json.dumps({"entry": entry, "running_extra_volume": dict(running_extra_volume)})
            + json.dumps({"extra_volumes": self.conf.extra_volumes})
            # + json.dumps(data_key)
        )
        if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
            with open(target_folder / f"{key}.pkl", "rb") as f:
                ret = pickle.load(f)
            self.unzip_a_file_into_a_folder(str(target_folder / f"{key}.zip"), local_path, cache_files_to_extract)
        else:
            ret = self.__run_with_retry(entry, local_path, env, running_extra_volume)
            with open(target_folder / f"{key}.pkl", "wb") as f:
                pickle.dump(ret, f)
            self.zip_a_folder_into_a_file(local_path, str(target_folder / f"{key}.zip"))
        return cast(EnvResult, ret)

    @abstractmethod
    def _run(
        self,
        entry: str | None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        Execute the specified entry point within the given environment and local path.

        Parameters
        ----------
        entry : str | None
            The entry point to execute. If None, defaults to the configured entry.
        local_path : str
            The local directory path where the execution should occur.
        env : dict | None
            Environment variables to set during execution.
        kwargs : dict
            Additional keyword arguments for execution customization.

        Returns
        -------
        tuple[str, int]
            A tuple containing the standard output and the exit code.
        """
        pass

    def dump_python_code_run_and_get_results(
        self,
        code: str,
        dump_file_names: list[str],
        local_path: str,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        code_dump_file_py_name: Optional[str] = None,
    ) -> tuple[str, list]:
        """
        Dump the code into the local path and run the code.
        """
        random_file_name = f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
        with open(os.path.join(local_path, random_file_name), "w") as f:
            f.write(code)
        entry = f"python {random_file_name}"
        log_output = self.check_output(entry, local_path, env, running_extra_volume=dict(running_extra_volume))
        results = []
        os.remove(os.path.join(local_path, random_file_name))
        for name in dump_file_names:
            if os.path.exists(os.path.join(local_path, f"{name}")):
                results.append(pickle.load(open(os.path.join(local_path, f"{name}"), "rb")))
                os.remove(os.path.join(local_path, f"{name}"))
            else:
                return log_output, []
        return log_output, results

    def refresh_env(self) -> None:
        """Refresh the environment, e.g., pull the latest docker image. rebuild the conda env."""
        pass


# class EnvWithCache
#

## Local Environment -----


class LocalConf(EnvConf):
    bin_path: str = ""
    """path like <path1>:<path2>:<path3>, which will be prepend to bin path."""

    retry_count: int = 0  # retry count for; run `retry_count + 1` times
    live_output: bool = True


ASpecificLocalConf = TypeVar("ASpecificLocalConf", bound=LocalConf)


class LocalEnv(Env[ASpecificLocalConf]):
    """
    Sometimes local environment may be more convenient for testing
    """

    def prepare(self) -> None: ...

    def _run(
        self,
        entry: str | None = None,
        local_path: str | None = None,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: dict,
    ) -> tuple[str, int]:

        # Handle volume links
        volumes = {}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp["bind"] if isinstance(rp, dict) else rp
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = T("scenarios.data_science.share:scen.cache_path").r()
        host_workspace = os.environ.get("HOST_WORKSPACE")
        container_workspace = os.environ.get("DS_LOCAL_DATA_PATH")
        for lp, rp in running_extra_volume.items():
            if host_workspace and container_workspace and lp.startswith(container_workspace):
                lp = lp.replace(container_workspace, host_workspace, 1)
            volumes[lp] = rp

        assert local_path is not None, "local_path should not be None"
        volumes = normalize_volumes(volumes, local_path)

        @contextlib.contextmanager
        def _symlink_ctx(vol_map: Mapping[str, str]) -> Generator[None, None, None]:
            created_links: list[Path] = []
            try:
                for real, link in vol_map.items():
                    link_path = Path(link)
                    real_path = Path(real)
                    if not link_path.parent.exists():
                        link_path.parent.mkdir(parents=True, exist_ok=True)
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(real_path)
                    created_links.append(link_path)
                yield
            finally:
                for p in created_links:
                    try:
                        if p.is_symlink() or p.exists():
                            p.unlink()
                    except FileNotFoundError:
                        pass

        with _symlink_ctx(volumes):
            # Setup environment
            if env is None:
                env = {}

            # Auto-propagate CUDA_VISIBLE_DEVICES for proper GPU isolation
            if "CUDA_VISIBLE_DEVICES" in os.environ and "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

            path = [
                *self.conf.bin_path.split(":"),
                "/bin/",
                "/usr/bin/",
                *env.get("PATH", "").split(":"),
            ]
            env["PATH"] = ":".join(path)

            if entry is None:
                entry = self.conf.default_entry

            print(Rule("[bold green]LocalEnv Logs Begin[/bold green]", style="dark_orange"))
            table = Table(title="Run Info", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("Entry", entry)
            table.add_row("Local Path", local_path or "")
            # Show code execution prompt in terminal output (hide internal wrappers).
            _print_entry_prompt(entry)
            table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("Volumes", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            cwd = Path(local_path).resolve() if local_path else None
            env = {k: str(v) if isinstance(v, int) else v for k, v in env.items()}

            process = subprocess.Popen(
                entry,
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Setup polling
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("The subprocess did not correctly create stdout/stderr pipes")

            if self.conf.live_output:
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()

                poller = select.poll()
                poller.register(stdout_fd, select.POLLIN)
                poller.register(stderr_fd, select.POLLIN)

                combined_output = ""
                while True:
                    if process.poll() is not None:
                        break
                    events = poller.poll(100)
                    for fd, event in events:
                        if event & select.POLLIN:
                            if fd == stdout_fd:
                                while True:
                                    output = process.stdout.readline()
                                    if output == "":
                                        break
                                    Console().print(output.strip(), markup=False)
                                    combined_output += output
                            elif fd == stderr_fd:
                                while True:
                                    error = process.stderr.readline()
                                    if error == "":
                                        break
                                    Console().print(error.strip(), markup=False)
                                    combined_output += error

                # Capture any final output
                remaining_output, remaining_error = process.communicate()
                if remaining_output:
                    Console().print(remaining_output.strip(), markup=False)
                    combined_output += remaining_output
                if remaining_error:
                    Console().print(remaining_error.strip(), markup=False)
                    combined_output += remaining_error
            else:
                # Sacrifice real-time output to avoid possible standard I/O hangs
                out, err = process.communicate()
                Console().print(out, end="", markup=False)
                Console().print(err, end="", markup=False)
                combined_output = out + err

            return_code = process.returncode
            print(Rule("[bold green]LocalEnv Logs End[/bold green]", style="dark_orange"))

            return combined_output, return_code


class CondaConf(LocalConf):
    conda_env_name: str
    default_entry: str = "python main.py"

    @model_validator(mode="after")
    def change_bin_path(self, **data: Any) -> "CondaConf":
        self._update_bin_path()
        return self

    def _update_bin_path(self) -> None:
        """Update bin_path by querying the conda environment's PATH.

        This is called during initialization and can be called again after prepare()
        to ensure bin_path is set correctly even if the conda env was just created.
        """
        conda_path_result = subprocess.run(
            f"conda run -n {self.conda_env_name} --no-capture-output env | grep '^PATH='",
            capture_output=True,
            text=True,
            shell=True,
        )
        self.bin_path = conda_path_result.stdout.strip().split("=")[1] if conda_path_result.returncode == 0 else ""


class MLECondaConf(CondaConf):
    enable_cache: bool = False  # aligning with the docker settings.


## Docker Environment -----
class DockerConf(EnvConf):
    build_from_dockerfile: bool = False
    dockerfile_folder_path: Optional[Path] = (
        None  # the path to the dockerfile optional path provided when build_from_dockerfile is False
    )
    image: str  # the image you want to build
    mount_path: str  # the path in the docker image to mount the folder
    default_entry: str  # the entry point of the image

    extra_volumes: dict = {}
    """It accept a dict of volumes, which can be either
    {<host_path>: <container_path>} or
    {<host_path>: {"bind": <container_path>, "mode": <mode, ro/rw/default is extra_volume_mode>}}
    """
    extra_volume_mode: str = "ro"  # by default. only the mount_path should be writable, others are changed to read-only

    exclude_chmod_paths: list[str] = []
    """List of directory names to exclude from chmod -R 777 operation.
    This prevents modifying permissions of read-only or specially configured directories."""

    # Declarative configuration for auto-populating exclude_chmod_paths from share.yaml
    # Subclasses can override these to specify which config keys to read
    _scenario_name: str | None = None  # e.g., "data_science"
    _exclude_path_keys: list[str] = []  # e.g., ["input_path", "cache_path"]

    # Sometime, we need maintain some extra data for the workspace.
    # And the extra data may be shared and the downloading can be time consuming.
    # So we just want to download it once.
    network: str | None = "bridge"  # the network mode for the docker
    shm_size: str | None = None
    enable_gpu: bool = True  # because we will automatically disable GPU if not available. So we enable it by default.
    mem_limit: str | None = "48g"  # Add memory limit attribute
    cpu_count: int | None = None  # Add CPU limit attribute

    running_timeout_period: int | None = 3600  # 1 hour

    enable_cache: bool = True  # enable the cache mechanism

    retry_count: int = 5  # retry count for the docker run
    retry_wait_seconds: int = 10  # retry wait seconds for the docker run
    save_logs_to_file: bool = True
    terminal_tail_lines: int = 20
    show_container_logs_in_terminal: bool = False
    show_docker_run_info_in_terminal: bool = False
    show_live_progress_in_terminal: bool = False

    @model_validator(mode="after")
    def populate_exclude_chmod_paths(self) -> "DockerConf":
        """
        Automatically populate exclude_chmod_paths from share.yaml configuration.

        This method reads path configurations from scenarios/<scenario_name>/share.yaml
        based on _scenario_name and _exclude_path_keys class attributes.
        """
        if not self.exclude_chmod_paths and self._scenario_name and self._exclude_path_keys:
            # Extract directory names from scenario configuration
            self.exclude_chmod_paths = [
                name
                for key in self._exclude_path_keys
                if (
                    name := extract_dir_name_from_path_config(
                        T(f"scenarios.{self._scenario_name}.share:scen.{key}").r()
                    )
                )
            ]
        return self


class DSDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="DS_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "DS_docker"
    image: str = "local_ds:latest"
    mount_path: str = "/kaggle/workspace"
    default_entry: str = "python main.py"

    running_timeout_period: int | None = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )

    # Declarative configuration: automatically loads from scenarios/data_science/share.yaml
    _scenario_name: str = "data_science"
    _exclude_path_keys: list[str] = ["input_path", "cache_path"]
    
    @model_validator(mode="after")
    def populate_data_science_volumes(self) -> "DSDockerConf":
        """
        Automatically populate extra_volumes for data science competitions by mounting
        the actual competition data directory to ./workspace_input/ in the container.
        
        This ensures that generated code which uses ./workspace_input/ can access
        the competition data that was downloaded and organized in DS_RD_SETTING.local_data_path.
        """
        if not self.extra_volumes and self._scenario_name == "data_science":
            try:
                from rdagent.app.data_science.conf import DS_RD_SETTING

                # Get the competition name - try from environment or settings
                competition = getattr(DS_RD_SETTING, 'competition', None)
                if competition:
                    # Resolve the actual data path for this competition
                    local_data_root = Path(DS_RD_SETTING.local_data_path).expanduser().resolve()
                    local_data_path = (local_data_root / competition).resolve()
                    if local_data_path.exists() and local_data_path.is_dir():
                        sample_data_path = (local_data_root / "sample" / competition).resolve()
                        sample_mount_enabled = True
                        try:
                            # Ensure the cache source directory exists before bind mount.
                            sample_data_path.mkdir(parents=True, exist_ok=True)
                        except Exception as sample_err:
                            sample_mount_enabled = False
                            logger.warning(
                                "Could not prepare DS sample cache mount folder %s: %s. "
                                "Skipping this mount and using runtime tmp cache instead.",
                                sample_data_path,
                                sample_err,
                            )

                        # Extract input and cache paths from share.yaml
                        input_path = T("scenarios.data_science.share:scen.input_path").r()
                        cache_path = T("scenarios.data_science.share:scen.cache_path").r()

                        # Create volume mappings for the container
                        # Map the competition data folder to ./workspace_input/ in the container
                        # Normalize bind paths to avoid Docker "Duplicate mount point" errors
                        normalized_input_bind = os.path.normpath(f"{self.mount_path}/{input_path}").rstrip("/")
                        normalized_cache_bind = os.path.normpath(f"{self.mount_path}/{cache_path}").rstrip("/")
                        
                        extra_volumes: dict[str, dict[str, str]] = {
                            str(local_data_path): {
                                "bind": normalized_input_bind,
                                "mode": "ro"
                            }
                        }

                        if sample_mount_enabled:
                            extra_volumes[str(sample_data_path)] = {
                                "bind": normalized_cache_bind,
                                "mode": "rw",
                            }

                        # Add kaggle credentials to inner container
                        kaggle_config = Path("~/.kaggle").expanduser().resolve()
                        if kaggle_config.exists():
                            extra_volumes[str(kaggle_config)] = {
                                "bind": "/root/.kaggle",
                                "mode": "ro"
                            }

                        self.extra_volumes = extra_volumes
                        logger.info(
                            f"Auto-configured DS Docker volumes: "
                            f"{local_data_path} -> {self.mount_path}/{input_path}"
                        )
            except Exception as e:
                logger.debug(f"Could not auto-populate DS Docker volumes: {e}")
        
        return self


class MLEBDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="MLEB_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "mle_bench_docker"
    image: str = "local_mle:latest"
    # image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/workspace/data_folder/"
    default_entry: str = "mlebench prepare --all"
    # extra_volumes: dict = {
    #     # TODO connect to the place where the data is stored
    #     Path("workspace/data").resolve(): "/root/.data/"
    # }
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )
    enable_cache: bool = False


# physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3
class DockerEnv(Env[DockerConf]):
    # TODO: Save the output into a specific file

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Download image if it doesn't exist
        """
        client = _docker_client_from_env()
        if (
            self.conf.build_from_dockerfile
            and self.conf.dockerfile_folder_path is not None
            and self.conf.dockerfile_folder_path.exists()
        ):
            resp_stream = client.api.build(
                path=str(self.conf.dockerfile_folder_path),
                tag=self.conf.image,
                network_mode=self.conf.network,
            )
            if isinstance(resp_stream, str):
                logger.debug(resp_stream)
            else:
                for part in resp_stream:
                    lines = part.decode("utf-8").split("\r\n")
                    for line in lines:
                        if not line.strip():
                            continue
                        status_dict = json.loads(line)
                        if "error" in status_dict:
                            raise docker.errors.BuildError(status_dict["error"], "")
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            image_pull = client.api.pull(self.conf.image, stream=True, decode=True)
            current_status = ""
            layer_set = set()
            completed_layers = 0
            with Progress(TextColumn("{task.description}"), TextColumn("{task.fields[progress]}")) as sp:
                main_task = sp.add_task("[cyan]Pulling image...", progress="")
                status_task = sp.add_task("[bright_magenta]layer status", progress="")
                for line in image_pull:
                    if "error" in line:
                        sp.update(
                            status_task,
                            description=f"[red]error",
                            progress=line["error"],
                        )
                        raise docker.errors.APIError(line["error"])

                    layer_id = line["id"]
                    status = line["status"]
                    p_text = line.get("progress", None)

                    if layer_id not in layer_set:
                        layer_set.add(layer_id)

                    if p_text:
                        current_status = p_text

                    if status == "Pull complete" or status == "Already exists":
                        completed_layers += 1

                    sp.update(
                        main_task,
                        progress=f"[green]{completed_layers}[white]/{len(layer_set)} layers completed",
                    )
                    sp.update(
                        status_task,
                        description=f"[bright_magenta]layer {layer_id} [yellow]{status}",
                        progress=current_status,
                    )
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while pulling the image: {e}")

    def _gpu_kwargs(self, client: docker.DockerClient) -> dict:  # type: ignore[no-any-unimported]
        """get gpu kwargs based on its availability.

        Supports GPU selection via CUDA_VISIBLE_DEVICES environment variable.
        If set, only the specified GPUs will be available in the container.
        Example: CUDA_VISIBLE_DEVICES=0,1 will only expose GPU 0 and 1.
        """
        if not self.conf.enable_gpu:
            return {}

        # Check if specific GPUs are requested via CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # Use device_ids to specify exact GPUs (cannot use count with device_ids)
            device_ids = [gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()]
            gpu_kwargs = {
                "device_requests": [docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])],
            }
            logger.info(f"GPU selection: using specific GPUs {device_ids}")
        else:
            # Default: use all available GPUs
            gpu_kwargs = {
                "device_requests": [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            }

        def get_image(image_name: str) -> None:
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                pull_image_with_progress(image_name)

        @wait_retry(5, 10)
        def _f() -> dict:
            container = None
            try:
                get_image(self.conf.image)
                container = client.containers.run(
                    self.conf.image, 
                    "nvidia-smi", 
                    detach=True, 
                    auto_remove=True,  # ADD THIS
                    **gpu_kwargs
                )
                container.wait()
                logger.info("GPU Devices are available.")
            except docker.errors.APIError:
                return {}
            return gpu_kwargs

        return _f()

    def _generate_log_header(self, entry: str | None = None) -> str:
        """
        Generate a header for log files with execution info.

        Args:
            entry: Command entry that was executed

        Returns:
            Formatted header string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = "=" * 80 + "\n"
        header += f"Docker Execution Log\n"
        header += f"Timestamp: {timestamp}\n"
        header += f"Image: {self.conf.image}\n"
        if entry and not _is_internal_entry_wrapper(entry):
            header += f"Command: {entry}\n"
        header += "=" * 80 + "\n\n"
        return header

    def _process_container_logs(self, logs: Iterable[bytes], local_path: str = ".", entry: str | None = None) -> str:
        """
        Process Docker container logs with optional tail mode.

        This method can be controlled via configuration:
        - save_logs_to_file: Save full logs to timestamped files in logs/ subdirectory
        - terminal_tail_lines: Show only last N lines in terminal (0 = show all)

        Args:
            logs: Docker container log stream
            local_path: Path to workspace for saving log files
            entry: Command entry that was executed (for logging header)

        Returns:
            Complete log output as string
        """
        log_output = ""

        # Determine if we should use tail mode
        use_tail_mode = self.conf.terminal_tail_lines > 0
        save_to_file = self.conf.save_logs_to_file
        show_in_terminal = self.conf.show_container_logs_in_terminal
        show_live_progress = False  # Force disable live progress print

        loop_step_pattern = re.compile(r"Start Loop\s+(\d+),\s*Step\s+(\d+):\s*([A-Za-z0-9_./-]+)")
        loop_pattern = re.compile(r"Start Loop\s+(\d+)")
        loop_tag_pattern = re.compile(r"Loop[_\s-]?(\d+)")
        step_tag_pattern = re.compile(r"(?:Step\s+(\d+)\s*:\s*([A-Za-z0-9_./-]+))|(?:/(\d+)_([A-Za-z0-9_./-]+))")

        current_loop: str | None = None
        current_step_idx: str | None = None
        current_step_name: str | None = None
        current_competition: str | None = None
        last_signal: str = "initializing"
        lines_seen = 0
        start_time = time.time()
        timeout_secs = self.conf.running_timeout_period

        competition_pattern = re.compile(r"(?:competition[_\s:-]*|Competition[_\s:-]*)([A-Za-z0-9_.-]+)")

        def _update_progress_state(line: str) -> None:
            nonlocal current_loop, current_step_idx, current_step_name, last_signal
            m_step = loop_step_pattern.search(line)
            if m_step:
                current_loop, current_step_idx, current_step_name = m_step.group(1), m_step.group(2), m_step.group(3)
                last_signal = f"loop={current_loop} step={current_step_idx}:{current_step_name}"
                return

            m_loop = loop_pattern.search(line)
            if m_loop:
                current_loop = m_loop.group(1)
                last_signal = f"loop={current_loop}"

            m_tag_loop = loop_tag_pattern.search(line)
            if m_tag_loop and current_loop is None:
                current_loop = m_tag_loop.group(1)

            m_tag_step = step_tag_pattern.search(line)
            if m_tag_step:
                if m_tag_step.group(1) and m_tag_step.group(2):
                    current_step_idx, current_step_name = m_tag_step.group(1), m_tag_step.group(2)
                elif m_tag_step.group(3) and m_tag_step.group(4):
                    current_step_idx, current_step_name = m_tag_step.group(3), m_tag_step.group(4)
                if current_loop is not None:
                    last_signal = f"loop={current_loop} step={current_step_idx}:{current_step_name}"

            comp_match = competition_pattern.search(line)
            if comp_match:
                current_competition = comp_match.group(1)

            if line:
                compact_line = line if len(line) <= 110 else (line[:107] + "...")
                last_signal = compact_line

        def _format_meta_text() -> Text:
            elapsed = int(time.time() - start_time)
            loop_text = current_loop if current_loop not in (None, "-") else None
            competition_text = current_competition if current_competition not in (None, "-") else None
            if current_step_idx is not None and current_step_name is not None and current_step_idx != "-" and current_step_name != "-":
                step_text = f"{current_step_idx}:{current_step_name}"
            else:
                step_text = None

            t = Text()
            t.append("Running", style="bold cyan")
            if loop_text:
                t.append(f" | loop={loop_text}")
            if step_text:
                t.append(f" | step={step_text}")
            if competition_text:
                t.append(f" | competition={competition_text}")
            t.append(f" | elapsed={elapsed}s")
            t.append(f" | lines={lines_seen}")
            t.append(f"\nimg={self.conf.image} | gpu={'on' if self.conf.enable_gpu else 'off'}")
            mode_text = "tail" if use_tail_mode else "stream"
            t.append(f" | mode={mode_text}")
            t.append(f" | timeout={timeout_secs if timeout_secs is not None else '-'}")

            if isinstance(timeout_secs, int) and timeout_secs > 0:
                ratio = min(max(elapsed / timeout_secs, 0.0), 1.0)
                bar_width = 20
                filled = int(ratio * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                t.append(f"\ntime: [{bar}] {elapsed}/{timeout_secs}s", style="dim")
            # Do not show cmd line
            t.append(f"\nlast: {last_signal}", style="dim")
            return t

        # Set up log file with timestamp if needed
        log_file_path = None
        if save_to_file and local_path:
            workspace = Path(local_path)

            # Create logs subdirectory
            logs_dir = workspace / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = logs_dir / f"docker_execution_{timestamp}.log"

            # Write header with execution info
            header = self._generate_log_header(entry)
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(header)

            # Also create/update a symlink to the latest log for convenience
            latest_link = logs_dir / "docker_execution_latest.log"

            if show_in_terminal:
                print(f"[cyan]Full logs will be saved to: {log_file_path.absolute()}[/cyan]")

        # Process logs with tail mode
        if use_tail_mode and show_in_terminal:

            log_buffer: Deque[str] = deque(maxlen=self.conf.terminal_tail_lines)

            def format_tail_display() -> Text:
                text = Text()
                text.append_text(_format_meta_text())
                text.append("\n")
                text.append(
                    f"[Showing last {len(log_buffer)}/{self.conf.terminal_tail_lines} lines",
                    style="dim",
                )
                if log_file_path:
                    text.append(f" | Full log: {log_file_path.name}]\n", style="dim cyan")
                else:
                    text.append("]\n", style="dim")
                text.append("-" * 80 + "\n", style="dim")
                for line in log_buffer:
                    text.append(line + "\n")
                return text

            with Live(format_tail_display(), refresh_per_second=2, console=Console()) as live:
                for log in logs:
                    decoded_log = log.strip().decode()
                    lines_seen += 1
                    _update_progress_state(decoded_log)
                    log_output += decoded_log + "\n"
                    log_buffer.append(decoded_log)

                    if log_file_path:
                        with open(log_file_path, "a", encoding="utf-8") as f:
                            f.write(decoded_log + "\n")

                    live.update(format_tail_display())
        else:
            if show_live_progress:
                with Live(_format_meta_text(), refresh_per_second=4, console=Console()) as live:
                    for log in logs:
                        decoded_log = log.strip().decode()
                        lines_seen += 1
                        _update_progress_state(decoded_log)
                        log_output += decoded_log + "\n"

                        if log_file_path:
                            with open(log_file_path, "a", encoding="utf-8") as f:
                                f.write(decoded_log + "\n")
                        live.update(_format_meta_text())
            else:
                for log in logs:
                    decoded_log = log.strip().decode()
                    lines_seen += 1
                    if show_in_terminal:
                        Console().print(decoded_log, markup=False)
                    log_output += decoded_log + "\n"

                    if log_file_path:
                        with open(log_file_path, "a", encoding="utf-8") as f:
                            f.write(decoded_log + "\n")

        # Show log file location and create latest symlink
        if log_file_path and log_file_path.exists():
            if show_in_terminal:
                print(f"[green]Full execution log saved to: {log_file_path.absolute()}[/green]")

            # Create or update symlink to latest log
            latest_link = log_file_path.parent / "docker_execution_latest.log"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            try:
                latest_link.symlink_to(log_file_path.name)
                if show_in_terminal:
                    print(f"[dim]Latest log symlink: logs/{latest_link.name} -> {log_file_path.name}[/dim]")
            except Exception:
                # Symlinks might not work on all systems (e.g., Windows without admin)
                pass

        return log_output

    def _run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        if env is None:
            env = {}
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer fork warning in multi-process training
        client = _docker_client_from_env()

        volumes = {}
        if local_path is not None:
            local_path = os.path.abspath(local_path)
            host_workspace = os.environ.get("HOST_WORKSPACE")
            container_workspace = os.environ.get("DS_LOCAL_DATA_PATH")
            container_workspace_abs = os.path.abspath(container_workspace) if container_workspace else None
            if host_workspace and container_workspace_abs and local_path.startswith(container_workspace_abs):
                host_local_path = local_path.replace(container_workspace_abs, host_workspace, 1)
            else:
                host_local_path = local_path
            volumes[host_local_path] = {"bind": self.conf.mount_path, "mode": "rw"}

        if self.conf.extra_volumes is not None:
            host_workspace = os.environ.get("HOST_WORKSPACE")
            container_workspace = os.environ.get("DS_LOCAL_DATA_PATH")
            container_workspace_abs = os.path.abspath(container_workspace) if container_workspace else None
            for lp, rp in self.conf.extra_volumes.items():
                # Remap extra_volumes host paths just like local_path
                if host_workspace and container_workspace_abs and lp.startswith(container_workspace_abs):
                    lp = lp.replace(container_workspace_abs, host_workspace, 1)
                volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            cache_bind = T("scenarios.data_science.share:scen.cache_path").r()
            already_mounted = any(
                (v.get("bind") if isinstance(v, dict) else v or "").rstrip("/").endswith(cache_bind.strip("./").rstrip("/"))
                for v in volumes.values()
            )
            if not already_mounted:
                volumes[cache_path] = {"bind": cache_bind, "mode": "rw"}
        host_workspace = os.environ.get("HOST_WORKSPACE")
        container_workspace = os.environ.get("DS_LOCAL_DATA_PATH")
        container_workspace_abs = os.path.abspath(container_workspace) if container_workspace else None
        for lp, rp in running_extra_volume.items():
            if host_workspace and container_workspace_abs and lp.startswith(container_workspace_abs):
                lp = lp.replace(container_workspace_abs, host_workspace, 1)
            volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}

        volumes = normalize_volumes(cast(dict[str, str | dict[str, str]], volumes), self.conf.mount_path)

        seen_binds: dict[str, str] = {}  # bind_target -> host_path
        deduped: dict[str, Any] = {}
        for host_path, vinfo in volumes.items():
            bind_target = vinfo["bind"] if isinstance(vinfo, dict) else vinfo
            # Normalize path to handle variations like /path/./subdir vs /path/subdir
            bind_target = os.path.normpath(bind_target).rstrip("/")
            if bind_target in seen_binds:
                logger.warning(
                    f"Duplicate Docker mount target '{bind_target}': "
                    f"dropping '{seen_binds[bind_target]}', keeping '{host_path}'."
                )
            seen_binds[bind_target] = host_path
            deduped[host_path] = vinfo
        volumes = deduped

        log_output = ""
        container: docker.models.containers.Container | None = None  # type: ignore[no-any-unimported]

        try:
            # Show code execution prompt in terminal output (hide internal wrappers).
            _print_entry_prompt(entry)
            container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes=volumes,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                # auto_remove=True, # remove too fast might cause the logs not to be get
                network=self.conf.network,
                shm_size=self.conf.shm_size,
                mem_limit=self.conf.mem_limit,  # Set memory limit
                cpu_count=self.conf.cpu_count,  # Set CPU limit
                **self._gpu_kwargs(client),
            )
            assert container is not None  # Ensure container was created successfully
            logs = container.logs(stream=True)
            if self.conf.show_docker_run_info_in_terminal:
                print(Rule("[bold green]Docker Logs Begin[/bold green]", style="dark_orange"))
                table = Table(title="Run Info", show_header=False)
                table.add_column("Key", style="bold cyan")
                table.add_column("Value", style="bold magenta")
                table.add_row("Image", self.conf.image)
                table.add_row("Container ID", container.id)
                table.add_row("Container Name", container.name)
                table.add_row("Entry", entry)
                table.add_row("Working Dir", str(self.conf.mount_path))
                table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
                vol_lines: list[str] = []
                for host_path, vinfo in volumes.items():
                    if isinstance(vinfo, dict):
                        bind_target = vinfo.get("bind", "")
                        mode = vinfo.get("mode", "")
                        vol_lines.append(f"{host_path} -> {bind_target} ({mode})")
                    else:
                        vol_lines.append(f"{host_path} -> {vinfo}")
                table.add_row("Volumes", "\n".join(vol_lines) if vol_lines else "(none)")
                Console().print(table)

            # Process logs (supports tail mode if configured)
            log_output = self._process_container_logs(logs, local_path, entry=entry)

            exit_status = container.wait()["StatusCode"]
            if self.conf.show_docker_run_info_in_terminal:
                print(Rule("[bold green]Docker Logs End[/bold green]", style="dark_orange"))
            return log_output, exit_status
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker image not found.")
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        finally:
            cleanup_container(container)
            try:
                client.containers.prune()  # removes all other stopped containers
            except Exception as e:
                logger.warning(f"Failed to prune stopped containers: {e}")
            try:
                protected_cleanup_paths: list[str] = []
                if local_path is not None:
                    protected_cleanup_paths.append(str(local_path))
                protected_cleanup_paths.extend(str(p) for p in self.conf.extra_volumes.keys())
                protected_cleanup_paths.extend(str(p) for p in running_extra_volume.keys())

                removed_log, removed_workspace = cleanup_timestamped_run_folders_on_docker_stop(
                    exclude_run_id=get_run_timestamp(),
                    protected_paths=protected_cleanup_paths,
                )
                if removed_log or removed_workspace:
                    logger.info(
                        "Post-Docker cleanup removed "
                        f"{removed_log} log run folder(s) and {removed_workspace} workspace run folder(s).",
                    )
            except Exception as e:
                logger.warning(f"Post-Docker run-folder cleanup failed: {e}")

    def refresh_env(self) -> None:
        """Remove the Docker image associated with this environment."""
        client = _docker_client_from_env()
        try:
            # Remove the specific image
            client.images.remove(image=self.conf.image, force=True)
            logger.info(f"Removed Docker image: {self.conf.image}")

            client.images.prune()
            client.api.prune_builds()
            logger.info(f"Successfully removed Docker image: {self.conf.image}")
        except docker.errors.ImageNotFound:
            logger.warning(f"Docker image not found, cannot remove: {self.conf.image}")
        except docker.errors.APIError as e:
            logger.error(f"Error while removing Docker image: {e}")
        self.prepare()


class MLEBDockerEnv(DockerEnv):
    """MLEBench Docker"""

    def __init__(self, conf: DockerConf = MLEBDockerConf()):
        super().__init__(conf)
