from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from extract.segmenter import StickerAsset
from export.animated_drawings import save_sticker


def _fake_sticker(h: int = 40, w: int = 30) -> StickerAsset:
    texture = np.zeros((h, w, 4), dtype=np.uint8)
    texture[..., 0] = 10   # B
    texture[..., 1] = 128  # G
    texture[..., 2] = 200  # R
    texture[..., 3] = 255  # alpha
    mask = np.full((h, w), 255, dtype=np.uint8)
    return StickerAsset(
        texture_bgra=texture, mask_u8=mask, source_region=(5, 7, w, h)
    )


def test_save_sticker_writes_texture_png_with_expected_size(tmp_path: Path) -> None:
    sticker = _fake_sticker(40, 30)

    save_sticker(sticker, tmp_path)

    texture_png = tmp_path / "texture.png"
    assert texture_png.exists()
    img = cv2.imread(str(texture_png), cv2.IMREAD_UNCHANGED)
    assert img.shape == (40, 30, 4)


def test_save_sticker_writes_mask_png_as_single_channel(tmp_path: Path) -> None:
    sticker = _fake_sticker(20, 25)

    save_sticker(sticker, tmp_path)

    mask_png = tmp_path / "mask.png"
    assert mask_png.exists()
    img = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
    assert img.shape == (20, 25)
    assert img.dtype == np.uint8


def test_save_sticker_writes_char_cfg_yaml_with_dimensions(tmp_path: Path) -> None:
    sticker = _fake_sticker(40, 30)

    save_sticker(sticker, tmp_path)

    yaml_path = tmp_path / "char_cfg.yaml"
    assert yaml_path.exists()
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["height"] == 40
    assert cfg["width"] == 30
    # Skeleton is a Stage 2 TODO — present but empty
    assert "skeleton" in cfg


def test_save_sticker_creates_destination_directory(tmp_path: Path) -> None:
    sticker = _fake_sticker()
    dest = tmp_path / "session_abc" / "sticker_01"
    assert not dest.exists()

    save_sticker(sticker, dest)

    assert dest.is_dir()
    assert (dest / "texture.png").exists()
