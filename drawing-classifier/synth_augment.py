"""
AR-BOOK Drawing Classifier — Synthetic Domain Randomization

Transforms clean Quick Draw images (black strokes on white bg) into
realistic "camera-captured drawing on paper" images.

Each call produces a different random variation: paper texture, shadows,
lighting, stroke variation, perspective warp, margins, blur, JPEG artifacts.
"""

import cv2
import numpy as np


def random_stroke_variation(img: np.ndarray) -> np.ndarray:
    """Randomly dilate or erode strokes to vary thickness."""
    # img: grayscale, black strokes on white bg
    # Invert for morphology (white strokes on black)
    inv = 255 - img
    k = np.random.randint(1, 4)  # kernel 1-3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if np.random.random() > 0.5:
        inv = cv2.dilate(inv, kernel, iterations=1)
    else:
        inv = cv2.erode(inv, kernel, iterations=1)
    return 255 - inv


def random_stroke_color(img: np.ndarray) -> np.ndarray:
    """Replace pure black strokes with random dark color (pen/pencil variation)."""
    # img: grayscale, 0=stroke, 255=bg
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)

    # Random dark color for strokes
    colors = [
        (30, 30, 30),     # dark gray (pencil)
        (10, 10, 40),     # dark blue (pen)
        (40, 20, 10),     # dark brown (marker)
        (0, 0, 0),        # black
        (50, 50, 50),     # light pencil
        (20, 10, 40),     # purple pen
    ]
    color = np.array(colors[np.random.randint(len(colors))], dtype=np.float32)

    # Mask: where strokes are (dark pixels)
    mask = (img < 128).astype(np.float32)
    mask3 = np.stack([mask] * 3, axis=-1)

    # Blend: stroke areas get the color, bg stays white
    rgb = rgb * (1 - mask3) + color * mask3
    return np.clip(rgb, 0, 255).astype(np.uint8)


def random_paper_background(rgb: np.ndarray) -> np.ndarray:
    """Replace white background with paper-like texture."""
    h, w = rgb.shape[:2]

    # Random paper base color (off-white)
    base = np.random.randint(200, 245)
    paper = np.full((h, w, 3), base, dtype=np.float32)

    # Add grain noise
    noise = np.random.normal(0, np.random.uniform(3, 12), (h, w, 3))
    paper = np.clip(paper + noise, 0, 255)

    # Detect background (bright pixels in all channels)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if len(rgb.shape) == 3 else rgb
    bg_mask = (gray > 200).astype(np.float32)
    bg_mask = cv2.GaussianBlur(bg_mask, (3, 3), 0)  # soft edge
    bg_mask3 = np.stack([bg_mask] * 3, axis=-1)

    # Blend
    result = rgb.astype(np.float32) * (1 - bg_mask3) + paper * bg_mask3
    return np.clip(result, 0, 255).astype(np.uint8)


def random_lighting_gradient(rgb: np.ndarray) -> np.ndarray:
    """Apply random directional lighting/shadow gradient."""
    h, w = rgb.shape[:2]

    # Random brightness at 4 corners
    corners = np.random.uniform(0.7, 1.0, size=4)  # TL, TR, BL, BR

    # Bilinear interpolation
    ys = np.linspace(0, 1, h).reshape(-1, 1)
    xs = np.linspace(0, 1, w).reshape(1, -1)
    gradient = (
        corners[0] * (1 - ys) * (1 - xs) +
        corners[1] * (1 - ys) * xs +
        corners[2] * ys * (1 - xs) +
        corners[3] * ys * xs
    )
    gradient = gradient[:, :, np.newaxis]

    result = rgb.astype(np.float32) * gradient
    return np.clip(result, 0, 255).astype(np.uint8)


def random_perspective_warp(rgb: np.ndarray) -> np.ndarray:
    """Apply slight random perspective transform."""
    h, w = rgb.shape[:2]

    # Max offset: 0-8% of image size
    max_off = int(max(h, w) * 0.08)
    if max_off < 2:
        return rgb

    def jitter(pt):
        return [pt[0] + np.random.randint(-max_off, max_off + 1),
                pt[1] + np.random.randint(-max_off, max_off + 1)]

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([jitter([0, 0]), jitter([w, 0]),
                       jitter([w, h]), jitter([0, h])])

    M = cv2.getPerspectiveTransform(src, dst)
    # Fill border with paper-like color
    bg_color = int(np.mean(rgb[0:5, 0:5]))
    result = cv2.warpPerspective(rgb, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(bg_color, bg_color, bg_color))
    return result


def random_margin_placement(rgb: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Place drawing at random scale/position with paper-texture margins."""
    h, w = rgb.shape[:2]

    # Random scale: drawing fills 55-95% of target
    scale = np.random.uniform(0.55, 0.95)
    new_size = int(target_size * scale)
    drawing = cv2.resize(rgb, (new_size, new_size))

    # Create paper-color canvas
    base = np.random.randint(200, 240)
    noise = np.random.normal(0, 5, (target_size, target_size, 3))
    canvas = np.clip(np.full((target_size, target_size, 3), base, dtype=np.float32) + noise,
                     0, 255).astype(np.uint8)

    # Random position
    max_offset = target_size - new_size
    dy = np.random.randint(0, max(1, max_offset))
    dx = np.random.randint(0, max(1, max_offset))
    canvas[dy:dy + new_size, dx:dx + new_size] = drawing

    return canvas


def random_blur(rgb: np.ndarray) -> np.ndarray:
    """Random slight Gaussian blur (camera focus softness)."""
    if np.random.random() > 0.6:
        k = np.random.choice([3, 5])
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)
    return rgb


def random_jpeg_compress(rgb: np.ndarray) -> np.ndarray:
    """Simulate JPEG compression artifacts."""
    if np.random.random() > 0.5:
        quality = np.random.randint(70, 96)
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
        rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return rgb


def synthesize_camera_image(gray_img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Full pipeline: clean Quick Draw grayscale → realistic camera capture.

    Args:
        gray_img: grayscale image, black strokes on white bg, any size
        target_size: output size (square)

    Returns:
        RGB image (target_size x target_size x 3), uint8 [0, 255]
    """
    # Resize to working resolution
    work_size = target_size + 40  # extra for perspective crop
    img = cv2.resize(gray_img, (work_size, work_size))

    # 1. Stroke thickness variation
    img = random_stroke_variation(img)

    # 2. Stroke color (grayscale → RGB with random pen color)
    rgb = random_stroke_color(img)

    # 3. Paper texture background
    rgb = random_paper_background(rgb)

    # 4. Lighting gradient / shadows
    rgb = random_lighting_gradient(rgb)

    # 5. Perspective warp
    rgb = random_perspective_warp(rgb)

    # 6. Margin + random placement
    rgb = random_margin_placement(rgb, target_size)

    # 7. Slight blur
    rgb = random_blur(rgb)

    # 8. JPEG compression artifacts
    rgb = random_jpeg_compress(rgb)

    # Ensure correct size
    if rgb.shape[:2] != (target_size, target_size):
        rgb = cv2.resize(rgb, (target_size, target_size))

    return rgb
