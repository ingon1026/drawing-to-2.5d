from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from extract.segmenter import StickerAsset


def render_sticker_at(
    frame: np.ndarray,
    sticker: StickerAsset,
    position: Tuple[int, int],
    *,
    enable_shadow: bool = True,
    shadow_offset: Tuple[int, int] = (5, 8),
    shadow_alpha: float = 0.4,
    shadow_blur: int = 4,
) -> None:
    """Composites sticker texture onto frame at (x, y) with optional drop shadow.

    Modifies frame in place. Clips to frame bounds; partially-visible stickers
    render only the visible portion.
    """
    if enable_shadow:
        _apply_shadow(
            frame,
            sticker.mask_u8,
            position=(position[0] + shadow_offset[0], position[1] + shadow_offset[1]),
            alpha=shadow_alpha,
            blur=shadow_blur,
        )
    _apply_texture(frame, sticker.texture_bgra, position)


def _clip_region(
    frame_shape: Tuple[int, int], tex_shape: Tuple[int, int], position: Tuple[int, int]
) -> Tuple[slice, slice, slice, slice] | None:
    """Return (frame_y_slice, frame_x_slice, tex_y_slice, tex_x_slice) or None if off-frame."""
    fh, fw = frame_shape
    th, tw = tex_shape
    x, y = position

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + tw), min(fh, y + th)
    if x2 <= x1 or y2 <= y1:
        return None

    tx1, ty1 = x1 - x, y1 - y
    tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)
    return (
        slice(y1, y2),
        slice(x1, x2),
        slice(ty1, ty2),
        slice(tx1, tx2),
    )


def _apply_texture(frame: np.ndarray, texture_bgra: np.ndarray, position: Tuple[int, int]) -> None:
    region = _clip_region(frame.shape[:2], texture_bgra.shape[:2], position)
    if region is None:
        return
    fy, fx, ty, tx = region

    tex = texture_bgra[ty, tx]
    tex_rgb = tex[..., :3].astype(np.float32)
    tex_alpha = (tex[..., 3].astype(np.float32) / 255.0)[..., None]

    frame_region = frame[fy, fx].astype(np.float32)
    blended = frame_region * (1.0 - tex_alpha) + tex_rgb * tex_alpha
    frame[fy, fx] = blended.astype(np.uint8)


def billboard_corners_2d(
    homography: np.ndarray,
    source_region: Tuple[int, int, int, int],
    lift_image_pixels: Optional[float] = None,
    popup_lift_ratio: float = 1.0,
) -> np.ndarray:
    """4 projected 2D corners of the standing billboard quad, in image space.

    Base edge = bottom of `source_region` mapped through `homography`, then lifted
    up in screen by `sh * popup_lift_ratio` pixels so the billboard floats above
    the drawing regardless of how the paper was oriented at capture time.

    Top edge = base further shifted up by `lift_image_pixels` (defaults to `sh`).

    `popup_lift_ratio`:
      0.0 → base lands exactly on the drawing's bottom (only visible pop-up when
            paper has moved since capture)
      1.0 → base lands at the drawing's top (billboard always floats above)
      >1.0 → billboard detaches higher above the drawing

    Order: [top-left, top-right, bottom-right, bottom-left].
    """
    sx, sy, sw, sh = source_region
    lift = float(lift_image_pixels) if lift_image_pixels is not None else float(sh)
    popup_offset = float(sh) * float(popup_lift_ratio)

    base = np.array([[sx, sy + sh], [sx + sw, sy + sh]], dtype=np.float64)
    ones = np.ones((2, 1))
    base_h = np.hstack([base, ones])
    proj = (homography @ base_h.T).T
    proj = proj[:, :2] / proj[:, 2:]

    proj = proj - np.array([0.0, popup_offset])
    top = proj + np.array([0.0, -lift])

    return np.array(
        [top[0], top[1], proj[1], proj[0]],
        dtype=np.float32,
    )


def render_sticker_as_billboard(
    frame: np.ndarray,
    sticker: StickerAsset,
    homography: np.ndarray,
    *,
    enable_shadow: bool = True,
    lift_image_pixels: Optional[float] = None,
    popup_lift_ratio: float = 1.0,
    shadow_alpha: float = 0.4,
    shadow_blur: int = 4,
) -> None:
    """Render sticker as a screen-upright billboard whose base follows the paper.

    The billboard always floats above the drawing by `sh * popup_lift_ratio`
    pixels so the pop-up effect is visible regardless of how the paper was
    oriented when the sticker was captured.
    """
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        return

    projected = billboard_corners_2d(
        homography,
        sticker.source_region,
        lift_image_pixels=lift_image_pixels,
        popup_lift_ratio=popup_lift_ratio,
    )
    if not np.isfinite(projected).all():
        return

    th, tw = sticker.texture_bgra.shape[:2]
    tex_corners = np.array(
        [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
    )

    try:
        M = cv2.getPerspectiveTransform(tex_corners, projected)
    except cv2.error:
        return

    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(
        sticker.texture_bgra,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    if enable_shadow:
        warped_mask = cv2.warpPerspective(
            sticker.mask_u8, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        shadow = warped_mask.astype(np.float32) / 255.0
        k = shadow_blur * 2 + 1
        shadow = cv2.GaussianBlur(shadow, (k, k), sigmaX=0)
        shadow *= shadow_alpha
        frame_f = frame.astype(np.float32)
        frame[:] = (frame_f * (1.0 - shadow[..., None])).astype(np.uint8)

    alpha = (warped[..., 3].astype(np.float32) / 255.0)[..., None]
    rgb = warped[..., :3].astype(np.float32)
    frame_f = frame.astype(np.float32)
    frame[:] = (frame_f * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


def render_sticker_with_homography(
    frame: np.ndarray,
    sticker: StickerAsset,
    homography: np.ndarray,
    *,
    enable_shadow: bool = True,
    shadow_offset: Tuple[int, int] = (5, 8),
    shadow_alpha: float = 0.4,
    shadow_blur: int = 4,
) -> None:
    """Warp sticker texture into frame via homography (sticker-on-paper tracking).

    homography maps reference-frame coords → current-frame coords. The sticker
    texture lives in reference-frame space offset by sticker.source_region.
    """
    sx, sy, _, _ = sticker.source_region
    translate = np.array([[1.0, 0.0, sx], [0.0, 1.0, sy], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = homography @ translate

    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(
        sticker.texture_bgra,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    if enable_shadow:
        warped_mask = cv2.warpPerspective(
            sticker.mask_u8,
            transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        shadow = warped_mask.astype(np.float32) / 255.0
        shift = np.float32(
            [[1, 0, shadow_offset[0]], [0, 1, shadow_offset[1]]]
        )
        shadow = cv2.warpAffine(shadow, shift, (w, h))
        k = shadow_blur * 2 + 1
        shadow = cv2.GaussianBlur(shadow, (k, k), sigmaX=0)
        shadow *= shadow_alpha
        frame_f = frame.astype(np.float32)
        frame[:] = (frame_f * (1.0 - shadow[..., None])).astype(np.uint8)

    alpha = (warped[..., 3].astype(np.float32) / 255.0)[..., None]
    rgb = warped[..., :3].astype(np.float32)
    frame_f = frame.astype(np.float32)
    frame[:] = (frame_f * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


def _apply_shadow(
    frame: np.ndarray,
    mask_u8: np.ndarray,
    position: Tuple[int, int],
    alpha: float,
    blur: int,
) -> None:
    region = _clip_region(frame.shape[:2], mask_u8.shape, position)
    if region is None:
        return
    fy, fx, my, mx = region

    shadow = mask_u8[my, mx].astype(np.float32) / 255.0
    k = blur * 2 + 1
    shadow = cv2.GaussianBlur(shadow, (k, k), sigmaX=0)
    shadow *= alpha

    frame_region = frame[fy, fx].astype(np.float32)
    frame[fy, fx] = (frame_region * (1.0 - shadow[..., None])).astype(np.uint8)
