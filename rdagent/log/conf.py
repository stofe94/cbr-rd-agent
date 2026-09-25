from pathlib import Path
from typing import Any

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings, get_run_timestamp, get_safe_cwd


class LogSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOG_", protected_namespaces=())

    trace_path: str = str(get_safe_cwd() / "log" / get_run_timestamp())

    format_console: str | None = None
    """"If it is None, leave it as the default"""

    ui_server_port: int | None = None

    storages: dict[str, list[int | str]] = {}

    def set_ui_server_port(self, port: int | None) -> None:
        self.ui_server_port = port
        if port is None:
            self.storages.pop("rdagent.log.ui.storage.WebStorage", None)
            return

        self.storages["rdagent.log.ui.storage.WebStorage"] = [port, self.trace_path]

    def model_post_init(self, _context: Any, /) -> None:
        self.set_ui_server_port(self.ui_server_port)


LOG_SETTINGS = LogSettings()
