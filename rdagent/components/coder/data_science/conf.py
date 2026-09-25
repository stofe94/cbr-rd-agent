import os
from pathlib import Path
from typing import Literal

from pydantic_settings import SettingsConfigDict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import (
    CondaConf,
    DockerEnv,
    DSDockerConf,
    Env,
    LocalEnv,
    MLEBDockerConf,
    MLECondaConf,
)


class DSCoderCoSTEERSettings(CoSTEERSettings):
    """Data Science CoSTEER settings"""

    model_config = SettingsConfigDict(env_prefix="DS_Coder_CoSTEER_")

    max_seconds_multiplier: int = 4
    env_type: str = "docker"
    # TODO: extract a function for env and conf.
    extra_evaluator: list[str] = []
    """Extra evaluators to use"""

    extra_eval: list[str] = []
    """
    Extra evaluators

    The evaluator follows the following assumptions:
    - It runs after previous evaluator (So the running results are already there)

    It is not a complete feature due to it is only implemented in DS Pipeline & Coder.

    TODO: The complete version should be implemented in the CoSTEERSettings.
    """


def get_ds_env(
    conf_type: Literal["kaggle", "mlebench"] = "kaggle",
    extra_volumes: dict = {},
    running_timeout_period: int | None = DS_RD_SETTING.debug_timeout,
    enable_cache: bool | None = None,
) -> Env:
    conf = DSCoderCoSTEERSettings()
    assert conf_type in ["kaggle", "mlebench"], f"Unknown conf_type: {conf_type}"
    if conf.env_type == "docker":
        env_conf = DSDockerConf() if conf_type == "kaggle" else MLEBDockerConf()
        env = DockerEnv(conf=env_conf)
    elif conf.env_type == "conda":
        env = LocalEnv(
            conf=(
                CondaConf(conda_env_name=conf_type) if conf_type == "kaggle" else MLECondaConf(conda_env_name=conf_type)
            )
        )
    else:
        raise ValueError(f"Unknown env type: {conf.env_type}")

    merged_volumes = env.conf.extra_volumes.copy()
    merged_volumes.update(extra_volumes)

    # Deduplicate by bind target — DSDockerConf auto-mounts may overlap with
    # legacy extra_volumes passed by callers (e.g. eval.py mounting workspace_input).
    # Last writer wins; earlier entries with the same bind target are dropped.
    # Normalize paths to handle variations like /path/./subdir vs /path/subdir
    # Also convert relative host paths to absolute to ensure Docker can resolve them.
    seen_binds: dict[str, tuple[str, dict]] = {}
    deduped: dict = {}
    for host_path, vinfo in merged_volumes.items():
        # Convert host path to absolute for Docker compatibility
        abs_host_path = str(Path(host_path).resolve())
        raw_bind_target = vinfo["bind"] if isinstance(vinfo, dict) else vinfo
        # Normalize path: remove trailing slashes and resolve ./
        bind_target = os.path.normpath(raw_bind_target).rstrip("/")
        if bind_target in seen_binds:
            prev_abs_host, prev_vinfo = seen_binds[bind_target]
            logger.warning(
                f"get_ds_env: dropping duplicate bind target '{bind_target}' "
                f"(keeping '{abs_host_path}', dropping '{prev_abs_host}')."
            )
        seen_binds[bind_target] = (abs_host_path, vinfo)
        deduped[abs_host_path] = vinfo
    env.conf.extra_volumes = deduped

    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


def get_clear_ws_cmd(
    stage: Literal["before_training", "before_inference"] = "before_training",
    tolerate_missing: bool = False,
) -> str:
    """
    Clean the files in workspace to a specific stage
    """
    assert stage in ["before_training", "before_inference"], f"Unknown stage: {stage}"
    if stage == "before_training":
        # During code-testing, these artifacts may legitimately not exist if execution failed earlier.
        # Use force deletion to avoid noisy/expected warnings.
        if DS_RD_SETTING.enable_model_dump:
            cmd = "rm -f submission.csv scores.csv trace.log && rm -rf models"
        else:
            cmd = "rm -f submission.csv scores.csv trace.log"
    else:
        # Before inference we usually expect prior artifacts to exist.
        # In some failure paths (e.g., debug execution failed), callers can opt into tolerant cleanup.
        cmd = "rm -f submission.csv scores.csv trace.log" if tolerate_missing else "rm submission.csv scores.csv trace.log"
    return cmd
