from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from extract.segmenter import (
    Segmenter,
    SegmentationError,
    StickerAsset,
    extract_sticker_from_mask,
)


def _bgr_frame(h: int = 100, w: int = 100, fill: int = 128) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_extract_sticker_crops_to_mask_bbox() -> None:
    frame = _bgr_frame(100, 100)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:80] = True  # 40 tall, 50 wide

    sticker = extract_sticker_from_mask(frame, mask)

    assert sticker.source_region == (30, 20, 50, 40)
    assert sticker.texture_bgra.shape == (40, 50, 4)
    assert sticker.mask_u8.shape == (40, 50)


def test_extract_sticker_alpha_channel_follows_mask() -> None:
    frame = _bgr_frame(50, 50)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True
    # Add a hole in the mask
    mask[15:18, 15:18] = False

    sticker = extract_sticker_from_mask(frame, mask)

    alpha = sticker.texture_bgra[..., 3]
    # Corner inside mask should be opaque
    assert alpha[0, 0] == 255
    # Hole position: local (5, 5) in 20x20 crop
    assert alpha[5, 5] == 0


def test_extract_sticker_raises_on_empty_mask() -> None:
    frame = _bgr_frame(50, 50)
    mask = np.zeros((50, 50), dtype=bool)

    with pytest.raises(SegmentationError):
        extract_sticker_from_mask(frame, mask)


@patch("extract.segmenter.SAM")
def test_segmenter_predicts_with_foreground_point(MockSAM: MagicMock) -> None:
    mock_model = MagicMock()
    MockSAM.return_value = mock_model

    fake_mask = torch.zeros((1, 100, 100), dtype=torch.uint8)
    fake_mask[0, 20:60, 30:80] = 1
    fake_result = MagicMock()
    fake_result.masks.data = fake_mask
    mock_model.predict.return_value = [fake_result]

    seg = Segmenter(weights_path="fake.pt")
    sticker = seg.segment(_bgr_frame(100, 100), (55, 40))

    mock_model.predict.assert_called_once()
    kwargs = mock_model.predict.call_args.kwargs
    assert kwargs["points"] == [[55, 40]]
    assert kwargs["labels"] == [1]

    assert isinstance(sticker, StickerAsset)
    assert sticker.source_region == (30, 20, 50, 40)


@patch("extract.segmenter.SAM")
def test_segmenter_raises_when_model_returns_no_mask(MockSAM: MagicMock) -> None:
    mock_model = MagicMock()
    MockSAM.return_value = mock_model

    fake_result = MagicMock()
    fake_result.masks = None
    mock_model.predict.return_value = [fake_result]

    seg = Segmenter(weights_path="fake.pt")
    with pytest.raises(SegmentationError):
        seg.segment(_bgr_frame(), (50, 50))
