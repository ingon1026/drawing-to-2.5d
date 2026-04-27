from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animate.torchserve_runtime import (
    EnvironmentCheckResult,
    TorchServeNotReady,
    TorchServeRuntime,
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


def _fake_torchserve_bin(tmp_path: Path) -> Path:
    bin_path = tmp_path / "torchserve"
    bin_path.write_text("#!/bin/sh\n")
    bin_path.chmod(0o755)
    return bin_path


def _ok_response():
    resp = MagicMock()
    resp.read.return_value = b'{"status": "Healthy"}'
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *a: None
    return resp


def test_runtime_start_spawns_subprocess_and_polls_health(tmp_path: Path) -> None:
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    config_path = tmp_path / "ts_config.properties"
    config_path.write_text("default_workers_per_model=1\n")
    bin_path = _fake_torchserve_bin(tmp_path)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running

    responses = [ConnectionError(), ConnectionError(), _ok_response()]

    with patch("animate.torchserve_runtime.subprocess.Popen", return_value=mock_proc) as popen, \
         patch("animate.torchserve_runtime.urlopen", side_effect=responses) as urlopen, \
         patch("animate.torchserve_runtime.time.sleep"):
        rt = TorchServeRuntime(
            model_store=model_store,
            config_path=config_path,
            models=["drawn_humanoid_detector.mar"],
            torchserve_bin=bin_path,
            health_url="http://127.0.0.1:8080/ping",
            poll_interval_sec=0.1,
            health_timeout_sec=5.0,
        )
        rt.start()

    assert popen.called
    args = popen.call_args[0][0]
    assert args[0] == str(bin_path)
    assert "--start" in args
    assert "--model-store" in args
    assert str(model_store) in args
    assert urlopen.call_count == 3


def test_runtime_start_raises_if_health_never_ok(tmp_path: Path) -> None:
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    config_path = tmp_path / "ts_config.properties"
    config_path.touch()
    bin_path = _fake_torchserve_bin(tmp_path)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    with patch("animate.torchserve_runtime.subprocess.Popen", return_value=mock_proc), \
         patch("animate.torchserve_runtime.urlopen", side_effect=ConnectionError()), \
         patch("animate.torchserve_runtime.time.sleep"):
        rt = TorchServeRuntime(
            model_store=model_store,
            config_path=config_path,
            models=["m.mar"],
            torchserve_bin=bin_path,
            health_url="http://127.0.0.1:8080/ping",
            poll_interval_sec=0.01,
            health_timeout_sec=0.1,
        )
        with pytest.raises(TorchServeNotReady):
            rt.start()


def test_runtime_start_raises_when_bin_missing(tmp_path: Path) -> None:
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    config_path = tmp_path / "ts_config.properties"
    config_path.touch()

    rt = TorchServeRuntime(
        model_store=model_store,
        config_path=config_path,
        models=["m.mar"],
        torchserve_bin=tmp_path / "does_not_exist",
    )
    with pytest.raises(TorchServeNotReady):
        rt.start()


def test_runtime_stop_invokes_torchserve_stop(tmp_path: Path) -> None:
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    config_path = tmp_path / "ts_config.properties"
    config_path.touch()
    bin_path = _fake_torchserve_bin(tmp_path)

    mock_proc = MagicMock()

    with patch("animate.torchserve_runtime.subprocess.Popen", return_value=mock_proc), \
         patch("animate.torchserve_runtime.subprocess.run") as run_mock, \
         patch("animate.torchserve_runtime.urlopen", return_value=_ok_response()), \
         patch("animate.torchserve_runtime.time.sleep"):
        rt = TorchServeRuntime(
            model_store=model_store,
            config_path=config_path,
            models=["m.mar"],
            torchserve_bin=bin_path,
            health_url="http://127.0.0.1:8080/ping",
            poll_interval_sec=0.01,
            health_timeout_sec=0.5,
        )
        rt.start()
        rt.stop()

    assert any(
        "--stop" in call.args[0] and call.args[0][0] == str(bin_path)
        for call in run_mock.call_args_list
    )
