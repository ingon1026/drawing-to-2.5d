from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from ultralytics import SAM


class SegmentationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StickerAsset:
    """Raw segmentation output. Channels are BGR+A (cv2-native); convert to RGB on PNG export."""

    texture_bgra: np.ndarray
    mask_u8: np.ndarray
    source_region: Tuple[int, int, int, int]  # (x, y, w, h)


def extract_sticker_from_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> StickerAsset:
    mask_bool = mask.astype(bool)
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        raise SegmentationError("empty mask")

    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

    cropped_bgr = frame_bgr[y0:y1, x0:x1]
    cropped_mask = mask_bool[y0:y1, x0:x1].astype(np.uint8) * 255
    texture_bgra = np.dstack([cropped_bgr, cropped_mask])

    return StickerAsset(
        texture_bgra=texture_bgra,
        mask_u8=cropped_mask,
        source_region=(x0, y0, x1 - x0, y1 - y0),
    )


class Segmenter:
    def __init__(self, weights_path: str) -> None:
        self._model = SAM(weights_path)

    def segment(self, frame_bgr: np.ndarray, point: Tuple[int, int]) -> StickerAsset:
        results = self._model.predict(
            frame_bgr,
            points=[[int(point[0]), int(point[1])]],
            labels=[1],
            verbose=False,
        )
        if not results:
            raise SegmentationError("model returned no results")
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            raise SegmentationError("model produced no mask")
        mask = masks.data[0]
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        return extract_sticker_from_mask(frame_bgr, mask)
