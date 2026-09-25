import base64
import json
import re
from pathlib import Path

from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import Env

_RUNTIME_INFO_SRC = Path(__file__).parent / "runtime_info.py"


def get_runtime_environment_by_env(env: Env) -> str:
    from rdagent.core.experiment import _resolve_run_workspace_root
    from rdagent.app.data_science.conf import DS_RD_SETTING

    probe_dir = _resolve_run_workspace_root() / DS_RD_SETTING.competition / "runtime_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "runtime_info.py").write_text(_RUNTIME_INFO_SRC.read_text())

    # Prefer running the file from the mounted workspace, but this can fail on Windows
    # Docker-in-Docker setups when the bind mount isn't applied correctly.
    result = env.run(entry="python runtime_info.py", local_path=str(probe_dir))
    stdout = result.stdout

    json_match = re.search(r"\{.*\}", stdout, re.DOTALL)
    if json_match is None:
        # Fallback: execute the probe script inline.
        # NOTE: Env.run wraps entry in `/bin/sh -c '...'`, so the command must not
        # contain any single quotes or it'll break the wrapper quoting.
        payload = base64.b64encode(_RUNTIME_INFO_SRC.read_text(encoding="utf-8").encode("utf-8")).decode("ascii")
        entry = (
            "python - <<PY\n"
            "import base64\n"
            f"src = base64.b64decode(\"\"\"{payload}\"\"\").decode(\"utf-8\")\n"
            "exec(compile(src, \"runtime_info.py\", \"exec\"), {\"__name__\": \"__main__\"})\n"
            "PY"
        )
        stdout = env.check_output(entry=entry, local_path=str(probe_dir))

    json_match = re.search(r"\{.*\}", stdout, re.DOTALL)
    if json_match is None:
        raise RuntimeError(f"Runtime probe returned no JSON.\nstdout was:\n{stdout}")
    return json.dumps(json.loads(json_match.group()), indent=2)


def check_runtime_environment(env: Env) -> str:
    implementation = FBWorkspace()
    strace_check = implementation.execute(env=env, entry="which strace || echo MISSING").strip()
    if strace_check.endswith("MISSING"):
        raise RuntimeError("`strace` not found in the target environment.")
    coverage_check = implementation.execute(env=env, entry="python -m coverage --version || echo MISSING").strip()
    if coverage_check.endswith("MISSING"):
        raise RuntimeError("`coverage` module not found or not runnable in the target environment.")