from pathlib import Path

import config


def test_config_exposes_torchserve_bin_path_pointing_to_animated_drawings_env() -> None:
    assert isinstance(config.TORCHSERVE_BIN, Path)
    assert "animated_drawings" in str(config.TORCHSERVE_BIN)
    assert str(config.TORCHSERVE_BIN).endswith("/torchserve")


def test_config_exposes_ad_python_path_pointing_to_animated_drawings_env() -> None:
    assert isinstance(config.AD_PYTHON, Path)
    assert "animated_drawings" in str(config.AD_PYTHON)
    assert str(config.AD_PYTHON).endswith("/python")


def test_config_paths_overridable_via_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("STICKERBOOK_TORCHSERVE_BIN", "/custom/ts")
    monkeypatch.setenv("STICKERBOOK_AD_PYTHON", "/custom/py")
    import importlib
    import config as config_module
    importlib.reload(config_module)
    try:
        assert str(config_module.TORCHSERVE_BIN) == "/custom/ts"
        assert str(config_module.AD_PYTHON) == "/custom/py"
    finally:
        importlib.reload(config_module)  # restore defaults
