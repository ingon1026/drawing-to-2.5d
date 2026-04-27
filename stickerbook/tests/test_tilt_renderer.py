import numpy as np

from extract.segmenter import StickerAsset
from render.tilt_renderer import (
    billboard_corners_2d,
    render_sticker_as_billboard,
    render_sticker_at,
    render_sticker_with_homography,
)


def _solid_sticker(bgr: tuple, alpha: int, size: int = 10) -> StickerAsset:
    tex = np.zeros((size, size, 4), dtype=np.uint8)
    tex[..., :3] = bgr
    tex[..., 3] = alpha
    mask = np.full((size, size), 255 if alpha > 0 else 0, dtype=np.uint8)
    return StickerAsset(texture_bgra=tex, mask_u8=mask, source_region=(0, 0, size, size))


def test_render_places_solid_red_texture_at_given_position() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sticker = _solid_sticker(bgr=(0, 0, 255), alpha=255, size=10)

    render_sticker_at(frame, sticker, position=(20, 30), enable_shadow=False)

    # Pixel inside placed region should be red (BGR)
    assert tuple(frame[35, 25]) == (0, 0, 255)
    # Pixel outside placed region should remain zero
    assert tuple(frame[0, 0]) == (0, 0, 0)


def test_render_with_zero_alpha_leaves_frame_unchanged() -> None:
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    original = frame.copy()
    sticker = _solid_sticker(bgr=(255, 255, 255), alpha=0, size=10)

    render_sticker_at(frame, sticker, position=(40, 40), enable_shadow=False)

    assert np.array_equal(frame, original)


def test_render_blends_texture_with_frame_by_alpha() -> None:
    # Frame is pure green, texture is pure red at 50% alpha
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[..., 1] = 255  # green channel
    sticker = _solid_sticker(bgr=(0, 0, 255), alpha=128, size=10)

    render_sticker_at(frame, sticker, position=(10, 10), enable_shadow=False)

    # Inside sticker region, expect ~half-red + ~half-green
    px = frame[15, 15]
    assert px[0] == 0  # blue channel stays 0
    assert 120 <= px[1] <= 135  # green roughly halved
    assert 120 <= px[2] <= 135  # red roughly half of 255


def test_render_clips_when_sticker_extends_past_frame_bounds() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    sticker = _solid_sticker(bgr=(0, 0, 255), alpha=255, size=20)

    render_sticker_at(frame, sticker, position=(40, 40), enable_shadow=False)

    # Only 10x10 region (40-49, 40-49) should be red; rest stays zero
    assert tuple(frame[45, 45]) == (0, 0, 255)
    assert tuple(frame[49, 49]) == (0, 0, 255)


def _sticker_at(bgr: tuple, alpha: int, source_xy: tuple, size: int = 10) -> StickerAsset:
    tex = np.zeros((size, size, 4), dtype=np.uint8)
    tex[..., :3] = bgr
    tex[..., 3] = alpha
    mask = np.full((size, size), 255, dtype=np.uint8)
    return StickerAsset(
        texture_bgra=tex,
        mask_u8=mask,
        source_region=(source_xy[0], source_xy[1], size, size),
    )


def test_render_with_identity_homography_places_at_source_region() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sticker = _sticker_at(bgr=(0, 0, 255), alpha=255, source_xy=(30, 40), size=10)

    render_sticker_with_homography(frame, sticker, np.eye(3), enable_shadow=False)

    # Center of warped region at source_region
    assert tuple(frame[44, 34]) == (0, 0, 255)
    assert tuple(frame[0, 0]) == (0, 0, 0)


def test_render_with_translation_homography_shifts_sticker() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sticker = _sticker_at(bgr=(0, 0, 255), alpha=255, source_xy=(30, 40), size=10)

    # Homography: translate by +15 x, +10 y
    H = np.array([[1, 0, 15], [0, 1, 10], [0, 0, 1]], dtype=np.float64)
    render_sticker_with_homography(frame, sticker, H, enable_shadow=False)

    # Original source center (at 34, 44) should be empty now
    assert tuple(frame[44, 34]) == (0, 0, 0)
    # New center: (34+15, 44+10) = (49, 54)
    assert tuple(frame[54, 49]) == (0, 0, 255)


def test_billboard_corners_2d_zero_popup_lift_base_on_source_bottom() -> None:
    # popup_lift_ratio=0 → billboard base lands exactly on source_region bottom.
    corners = billboard_corners_2d(
        np.eye(3), source_region=(100, 50, 200, 150), popup_lift_ratio=0.0
    )
    assert corners.shape == (4, 2)
    assert tuple(corners[3]) == (100.0, 200.0)  # BL
    assert tuple(corners[2]) == (300.0, 200.0)  # BR
    # Top at source_region top (lifted by sh=150)
    assert tuple(corners[0]) == (100.0, 50.0)
    assert tuple(corners[1]) == (300.0, 50.0)


def test_billboard_corners_2d_default_popup_lift_floats_above_source() -> None:
    # Default popup_lift_ratio=1.0 → billboard base at source_region top,
    # top at sh above source_region top.
    corners = billboard_corners_2d(np.eye(3), source_region=(100, 50, 200, 150))
    # Base at source_region top (y=50)
    assert tuple(corners[3]) == (100.0, 50.0)
    assert tuple(corners[2]) == (300.0, 50.0)
    # Top at y = 50 - 150 = -100
    assert tuple(corners[0]) == (100.0, -100.0)
    assert tuple(corners[1]) == (300.0, -100.0)


def test_billboard_corners_2d_base_translates_with_homography() -> None:
    # Translation by +10 x, +5 y — base edge should move by the same (ratio=0 for clarity)
    H = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float64)
    corners = billboard_corners_2d(
        H, source_region=(100, 50, 200, 150), popup_lift_ratio=0.0
    )
    assert tuple(corners[3]) == (110.0, 205.0)  # BL shifted
    assert tuple(corners[2]) == (310.0, 205.0)  # BR shifted


def test_billboard_render_modifies_frame_with_identity_homography() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    before = frame.copy()
    sticker = _sticker_at(bgr=(0, 0, 255), alpha=255, source_xy=(250, 200), size=80)

    render_sticker_as_billboard(frame, sticker, np.eye(3), enable_shadow=False)

    assert not np.array_equal(frame, before)


def test_billboard_render_leaves_frame_unchanged_on_degenerate_homography() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    before = frame.copy()
    sticker = _sticker_at(bgr=(0, 0, 255), alpha=255, source_xy=(250, 200), size=80)

    render_sticker_as_billboard(frame, sticker, np.zeros((3, 3)), enable_shadow=False)

    assert np.array_equal(frame, before)


from render.tilt_renderer import render_bgra_as_billboard


def test_render_bgra_as_billboard_modifies_frame_with_identity_h() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    before = frame.copy()
    tex_bgra = np.zeros((80, 80, 4), dtype=np.uint8)
    tex_bgra[..., 2] = 255  # red
    tex_bgra[..., 3] = 255  # opaque

    render_bgra_as_billboard(
        frame=frame,
        texture_bgra=tex_bgra,
        source_region=(250, 200, 80, 80),
        homography=np.eye(3),
        enable_shadow=False,
    )
    assert not np.array_equal(frame, before)


def test_render_bgra_as_billboard_degenerate_homography_no_op() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    before = frame.copy()
    tex_bgra = np.zeros((80, 80, 4), dtype=np.uint8)
    tex_bgra[..., 3] = 255

    render_bgra_as_billboard(
        frame=frame, texture_bgra=tex_bgra,
        source_region=(250, 200, 80, 80),
        homography=np.zeros((3, 3)), enable_shadow=False,
    )
    assert np.array_equal(frame, before)
