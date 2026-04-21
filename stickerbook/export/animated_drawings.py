from __future__ import annotations

from pathlib import Path

import cv2
import yaml

from extract.segmenter import StickerAsset


def save_sticker(sticker: StickerAsset, dest_dir: Path) -> None:
    """Write AnimatedDrawings-compatible assets (texture.png, mask.png, char_cfg.yaml).

    cv2.imwrite reorders BGR(A) → RGB(A) for PNG, so the stored texture is
    standard RGBA suitable for AnimatedDrawings' pipeline.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dest_dir / "texture.png"), sticker.texture_bgra)
    cv2.imwrite(str(dest_dir / "mask.png"), sticker.mask_u8)

    h, w = sticker.texture_bgra.shape[:2]
    cfg = {
        "height": int(h),
        "width": int(w),
        "source_region": list(sticker.source_region),
        "skeleton": [],  # Stage 2 (AnimatedDrawings pose/rig) fills this in.
    }
    with (dest_dir / "char_cfg.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
