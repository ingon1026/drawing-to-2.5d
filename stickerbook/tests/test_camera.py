from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capture.camera import Camera, CameraError


def _fake_cap(*, opened: bool = True, read_ok: bool = True, frame: np.ndarray | None = None) -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = opened
    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cap.read.return_value = (read_ok, frame if read_ok else None)
    return cap


@patch("capture.camera.cv2.VideoCapture")
def test_camera_opens_device_at_given_index(mock_video_capture: MagicMock) -> None:
    mock_video_capture.return_value = _fake_cap()
    Camera(index=2)
    mock_video_capture.assert_called_once_with(2)


@patch("capture.camera.cv2.VideoCapture")
def test_camera_read_returns_frame(mock_video_capture: MagicMock) -> None:
    expected = np.ones((480, 640, 3), dtype=np.uint8) * 128
    mock_video_capture.return_value = _fake_cap(frame=expected)

    cam = Camera(index=0)
    frame = cam.read()

    assert np.array_equal(frame, expected)


@patch("capture.camera.cv2.VideoCapture")
def test_camera_read_raises_on_failure(mock_video_capture: MagicMock) -> None:
    mock_video_capture.return_value = _fake_cap(read_ok=False)

    cam = Camera(index=0)
    with pytest.raises(CameraError):
        cam.read()


@patch("capture.camera.cv2.VideoCapture")
def test_camera_release_closes_device(mock_video_capture: MagicMock) -> None:
    cap = _fake_cap()
    mock_video_capture.return_value = cap

    cam = Camera(index=0)
    cam.release()

    cap.release.assert_called_once()


@patch("capture.camera.cv2.VideoCapture")
def test_camera_raises_if_open_fails(mock_video_capture: MagicMock) -> None:
    mock_video_capture.return_value = _fake_cap(opened=False)

    with pytest.raises(CameraError):
        Camera(index=999)
