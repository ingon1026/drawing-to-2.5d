import numpy as np

from animate.animated_drawings_runner import composite_on_white_bg


def test_composite_places_opaque_pixels_over_white() -> None:
    tex = np.zeros((10, 10, 4), dtype=np.uint8)
    tex[2:5, 2:5, :3] = (0, 0, 255)  # red BGR
    tex[2:5, 2:5, 3] = 255            # opaque

    out = composite_on_white_bg(tex)

    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8
    # transparent area is white
    assert tuple(out[0, 0]) == (255, 255, 255)
    # opaque area keeps red
    assert tuple(out[3, 3]) == (0, 0, 255)


def test_composite_blends_semitransparent_pixels_with_white() -> None:
    tex = np.zeros((4, 4, 4), dtype=np.uint8)
    tex[1, 1, :3] = (0, 0, 255)
    tex[1, 1, 3] = 128  # ~50%

    out = composite_on_white_bg(tex)

    # Semi-transparent red over white: B/G blend from 0 toward 255; R stays 255.
    px = out[1, 1]
    # Pure red foreground: R channel stays 255 (both fg and bg are 255)
    assert int(px[2]) == 255
    # B and G blend from 0 → ~127 against white
    assert 120 <= int(px[1]) <= 135
    assert 120 <= int(px[0]) <= 135


def test_composite_accepts_bgra_float_and_returns_uint8() -> None:
    tex = np.zeros((4, 4, 4), dtype=np.float32)
    tex[..., 3] = 255.0
    out = composite_on_white_bg(tex)
    assert out.dtype == np.uint8
