from app import App, AppAction


def test_handle_key_quit_on_lowercase_q() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("q")) == AppAction.QUIT


def test_handle_key_quit_on_uppercase_q() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("Q")) == AppAction.QUIT


def test_handle_key_quit_on_escape() -> None:
    app = App(camera_index=0)
    assert app._handle_key(27) == AppAction.QUIT


def test_handle_key_returns_none_when_no_key_pressed() -> None:
    app = App(camera_index=0)
    assert app._handle_key(-1) is None


def test_handle_key_reset_on_lowercase_r() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("r")) == AppAction.RESET


def test_handle_key_reset_on_uppercase_r() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("R")) == AppAction.RESET


def test_handle_key_save_on_lowercase_s() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("s")) == AppAction.SAVE


def test_handle_key_save_on_uppercase_s() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("S")) == AppAction.SAVE


def test_handle_key_returns_none_for_other_keys() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("a")) is None
    assert app._handle_key(ord("x")) is None
