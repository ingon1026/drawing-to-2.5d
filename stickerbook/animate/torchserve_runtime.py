"""TorchServe lifecycle wrapper for AnimatedDrawings inference.

Invokes an externally-installed torchserve binary (living in the
`animated_drawings` conda env) via absolute path — stickerbook's own
python env does NOT install torchserve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


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
