from pathlib import Path

import numpy as np

from app import AnchoredSticker, AnimationState
from extract.segmenter import StickerAsset
from track.homography_anchor import HomographyAnchor


def _dummy_asset() -> StickerAsset:
    tex = np.zeros((10, 10, 4), dtype=np.uint8)
    tex[..., 3] = 255
    mask = np.full((10, 10), 255, dtype=np.uint8)
    return StickerAsset(texture_bgra=tex, mask_u8=mask, source_region=(0, 0, 10, 10))


def test_anchored_sticker_defaults_to_static_animation_state() -> None:
    item = AnchoredSticker(sticker=_dummy_asset(), anchor=HomographyAnchor())
    assert item.animation_state is AnimationState.STATIC
    assert item.animation_video_path is None
    assert item.animation_started_at is None


def test_animation_state_enum_has_expected_members() -> None:
    names = {s.name for s in AnimationState}
    assert names == {"STATIC", "PREPARING", "ANIMATED", "FAILED"}
