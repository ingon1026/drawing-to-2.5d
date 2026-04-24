"""TorchServe lifecycle wrapper for AnimatedDrawings inference.

Invokes an externally-installed torchserve binary (living in the
`animated_drawings` conda env) via absolute path — stickerbook's own
python env does NOT install torchserve.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class EnvironmentCheckResult:
    ok: bool
    missing: List[str] = field(default_factory=list)
    install_hint: str = ""


def check_environment(torchserve_bin: Path) -> EnvironmentCheckResult:
    missing: List[str] = []
    if not Path(torchserve_bin).is_file():
        missing.append("torchserve")

    install_hint = ""
    if missing:
        install_hint = (
            f"torchserve not found at {torchserve_bin}. "
            f"Install into the animated_drawings conda env, or set "
            f"STICKERBOOK_TORCHSERVE_BIN to override."
        )

    return EnvironmentCheckResult(
        ok=not missing,
        missing=missing,
        install_hint=install_hint,
    )


class TorchServeNotReady(RuntimeError):
    pass


class TorchServeRuntime:
    def __init__(
        self,
        model_store: Path,
        config_path: Path,
        models: List[str],
        torchserve_bin: Path,
        health_url: str = "http://127.0.0.1:8080/ping",
        poll_interval_sec: float = 1.0,
        health_timeout_sec: float = 30.0,
    ) -> None:
        self._model_store = Path(model_store)
        self._config_path = Path(config_path)
        self._models = list(models)
        self._torchserve_bin = Path(torchserve_bin)
        self._health_url = health_url
        self._poll_interval_sec = poll_interval_sec
        self._health_timeout_sec = health_timeout_sec
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if not Path(self._torchserve_bin).is_file():
            raise TorchServeNotReady(
                f"torchserve binary not found at {self._torchserve_bin}. "
                f"Override with STICKERBOOK_TORCHSERVE_BIN env var."
            )

        cmd = [
            str(self._torchserve_bin),
            "--start",
            "--model-store", str(self._model_store),
            "--ts-config", str(self._config_path),
            "--models", *self._models,
            "--disable-token-auth",
            "--no-config-snapshots",
        ]
        self._proc = subprocess.Popen(cmd)
        self._wait_for_health()

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self._health_timeout_sec
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                with urlopen(self._health_url, timeout=2.0) as resp:
                    body = resp.read().decode()
                    if '"Healthy"' in body or '"status": "Healthy"' in body:
                        return
            except (URLError, ConnectionError, OSError) as e:
                last_err = e
            time.sleep(self._poll_interval_sec)
        raise TorchServeNotReady(
            f"health probe did not pass within {self._health_timeout_sec}s "
            f"(last error: {last_err})"
        )

    def stop(self) -> None:
        subprocess.run([str(self._torchserve_bin), "--stop"], check=False)
        self._proc = None
