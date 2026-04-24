from pathlib import Path

import pytest

from animate.torchserve_runtime import (
    EnvironmentCheckResult,
    check_environment,
)


def test_check_environment_reports_ok_when_bin_exists(tmp_path: Path) -> None:
    fake_bin = tmp_path / "torchserve"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)

    result = check_environment(torchserve_bin=fake_bin)

    assert isinstance(result, EnvironmentCheckResult)
    assert result.ok is True
    assert result.missing == []


def test_check_environment_reports_missing_when_bin_not_found(tmp_path: Path) -> None:
    result = check_environment(torchserve_bin=tmp_path / "nope" / "torchserve")

    assert result.ok is False
    assert "torchserve" in result.missing
    assert "STICKERBOOK_TORCHSERVE_BIN" in result.install_hint
