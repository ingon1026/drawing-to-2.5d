"""
drawing-2p5d — 2.5D Card-Tilt Demo Viewer

Displays the extracted object.png with perspective tilt driven by mouse position,
plus dynamic lighting (highlight/shadow shift) for a convincing 2.5D card effect.

Usage:
    python3 viewer.py --image output/object.png
    python3 viewer.py --image output/object.png --shadow

Keys:
    SPACE  — drop & bounce reset
    1      — tilt only (default)
    2      — tilt + bounce
    3      — tilt + bounce + squash
    ESC/Q  — quit

Mouse:
    Move mouse to tilt the card in 3D perspective.
"""

import argparse
import math
import sys

import numpy as np
import pygame


# === Window ===
WINDOW_W = 800
WINDOW_H = 600
FPS = 60

# === Tilt ===
MAX_TILT_DEG = 25.0       # max perspective tilt angle in degrees
TILT_SMOOTHING = 8.0      # lerp speed (higher = more responsive)

# === Lighting ===
HIGHLIGHT_STRENGTH = 80   # max highlight overlay alpha
SHADOW_STRENGTH = 60      # max shadow overlay alpha

# === Bounce physics ===
GRAVITY = 1200.0
BOUNCE_DAMPING = 0.7
SQUASH_AMOUNT = 0.25
SQUASH_DECAY = 8.0

# === Edge thickness (fake 3D side) ===
EDGE_LAYERS = 6           # number of edge layers for depth illusion
EDGE_COLOR = (60, 50, 40)


def perspective_warp(surface, tilt_x, tilt_y):
    """Apply perspective warp to a pygame surface based on tilt angles.

    tilt_x: -1 to 1 (left/right tilt)
    tilt_y: -1 to 1 (up/down tilt)

    Returns a new surface with perspective transform applied.
    """
    w, h = surface.get_size()

    # Convert surface to numpy array (RGBA)
    arr = pygame.surfarray.pixels_alpha(surface)
    # We need the full RGBA array
    arr_rgb = pygame.surfarray.array3d(surface)  # (W, H, 3) in pygame format
    arr_a = pygame.surfarray.array2d(surface) >> 24 & 0xFF  # crude, let's do it properly

    # Actually, let's use a proper approach with pixel array
    # Convert pygame surface to numpy RGBA
    raw = pygame.image.tobytes(surface, "RGBA")
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4)).copy()

    # Source corners: top-left, top-right, bottom-right, bottom-left
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Perspective shift amounts
    dx = tilt_x * w * 0.08  # horizontal perspective
    dy = tilt_y * h * 0.08  # vertical perspective

    # Destination corners with perspective
    dst = np.float32([
        [0 + dx + dy * 0.3,  0 + dy + dx * 0.3],       # top-left
        [w + dx - dy * 0.3,  0 - dy + dx * 0.3],       # top-right
        [w - dx - dy * 0.3,  h - dy - dx * 0.3],       # bottom-right
        [0 - dx + dy * 0.3,  h + dy - dx * 0.3],       # bottom-left
    ])

    # Compute perspective transform matrix
    import cv2
    M = cv2.getPerspectiveTransform(src, dst)

    # Calculate output bounds
    all_x = dst[:, 0]
    all_y = dst[:, 1]
    min_x, max_x = int(min(all_x)) - 2, int(max(all_x)) + 2
    min_y, max_y = int(min(all_y)) - 2, int(max(all_y)) + 2
    out_w = max_x - min_x
    out_h = max_y - min_y

    # Offset transform to fit in positive coords
    T = np.float32([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    M_shifted = T @ M

    # Warp
    warped = cv2.warpPerspective(img, M_shifted, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))

    # Convert back to pygame surface
    result = pygame.image.frombytes(warped.tobytes(), (out_w, out_h), "RGBA")
    return result, min_x, min_y


def make_lighting_overlay(w, h, tilt_x, tilt_y):
    """Create a lighting overlay: highlight on the tilted-toward side, shadow on the away side."""
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)

    # Highlight: semi-transparent white on the side facing the "light"
    hl_alpha = int(abs(tilt_x) * HIGHLIGHT_STRENGTH + abs(tilt_y) * HIGHLIGHT_STRENGTH * 0.5)
    hl_alpha = min(hl_alpha, HIGHLIGHT_STRENGTH)

    if hl_alpha > 5:
        # Gradient direction based on tilt
        for i in range(h):
            row_factor = i / h
            # Vertical gradient based on tilt_y
            v_factor = (1.0 - row_factor) if tilt_y < 0 else row_factor
            for j_step in range(0, w, 4):  # step by 4 for performance
                col_factor = j_step / w
                h_factor = (1.0 - col_factor) if tilt_x > 0 else col_factor
                alpha = int(hl_alpha * max(0, h_factor * 0.7 + v_factor * 0.3 - 0.3))
                alpha = min(alpha, 255)
                if alpha > 0:
                    pygame.draw.rect(overlay, (255, 255, 255, alpha), (j_step, i, 4, 1))

    return overlay


def make_lighting_overlay_fast(w, h, tilt_x, tilt_y):
    """Fast lighting overlay using numpy."""
    # Create gradient arrays
    col_grad = np.linspace(0, 1, w, dtype=np.float32)
    row_grad = np.linspace(0, 1, h, dtype=np.float32)

    # Horizontal: highlight on the side we tilt toward
    if tilt_x > 0:
        h_factor = 1.0 - col_grad  # highlight on left when tilting right (light from left)
    else:
        h_factor = col_grad

    # Vertical
    if tilt_y < 0:
        v_factor = 1.0 - row_grad  # highlight on top when tilting up
    else:
        v_factor = row_grad

    # Combine: 2D gradient
    light = np.outer(v_factor * 0.3 + 0.7, np.ones(w)) * 0.3 + \
            np.outer(np.ones(h), h_factor) * 0.7

    intensity = (abs(tilt_x) + abs(tilt_y) * 0.5) * 0.8
    intensity = min(intensity, 1.0)

    # Highlight (white) on bright side, shadow (black) on dark side
    highlight = (light * intensity * HIGHLIGHT_STRENGTH).clip(0, 255).astype(np.uint8)
    shadow = ((1.0 - light) * intensity * SHADOW_STRENGTH).clip(0, 255).astype(np.uint8)

    # Build RGBA overlay
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
    # Where highlight > shadow: white with highlight alpha
    mask_hl = highlight > shadow
    overlay_arr[mask_hl, 0] = 255
    overlay_arr[mask_hl, 1] = 255
    overlay_arr[mask_hl, 2] = 255
    overlay_arr[mask_hl, 3] = highlight[mask_hl]
    # Where shadow >= highlight: black with shadow alpha
    mask_sh = ~mask_hl
    overlay_arr[mask_sh, 3] = shadow[mask_sh]

    surf = pygame.image.frombytes(overlay_arr.tobytes(), (w, h), "RGBA")
    return surf


def draw_shadow(screen, cx, ground_y, base_w, tilt_x, height_above, alpha=50):
    """Draw perspective ground shadow."""
    # Shadow shifts opposite to tilt
    shadow_shift = -tilt_x * 40
    # Shadow size varies with height
    spread = max(0.4, 1.0 - height_above / 500.0)
    sw = int(base_w * spread * 1.2)
    sh = max(6, int(16 * spread))
    shadow_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow_surf, (0, 0, 0, int(alpha * spread)),
                        (0, 0, sw, sh))
    screen.blit(shadow_surf, (int(cx + shadow_shift - sw / 2), int(ground_y - sh / 2)))


def draw_edge_layers(screen, surface, x, y, tilt_x, tilt_y):
    """Draw fake 3D edge layers behind the main surface."""
    for i in range(EDGE_LAYERS, 0, -1):
        offset_x = int(tilt_x * i * 1.5)
        offset_y = int(tilt_y * i * 1.5)
        # Darken progressively
        darkness = max(0, 255 - i * 30)
        edge = surface.copy()
        edge.fill((darkness, darkness, darkness), special_flags=pygame.BLEND_RGB_MULT)
        edge.set_alpha(180)
        screen.blit(edge, (x + offset_x, y + offset_y))


def main():
    parser = argparse.ArgumentParser(description="2.5D Card-Tilt Demo Viewer")
    parser.add_argument("--image", "-i", default="output/object.png", help="Transparent PNG path")
    parser.add_argument("--bg-color", nargs=3, type=int, default=[240, 245, 250],
                        help="Background RGB (default: 240 245 250)")
    parser.add_argument("--shadow", action="store_true", default=True, help="Show ground shadow")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scale factor")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("2.5D Card-Tilt Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Load image
    try:
        img = pygame.image.load(args.image).convert_alpha()
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Scale
    if args.scale != 1.0:
        new_w = max(1, int(img.get_width() * args.scale))
        new_h = max(1, int(img.get_height() * args.scale))
        img = pygame.transform.smoothscale(img, (new_w, new_h))

    # Fit to window (max 50% of window height)
    max_h = int(WINDOW_H * 0.5)
    if img.get_height() > max_h:
        ratio = max_h / img.get_height()
        img = pygame.transform.smoothscale(img, (int(img.get_width() * ratio), max_h))

    bg_color = tuple(args.bg_color)
    ground_y = WINDOW_H - 60

    # State
    cx, cy = WINDOW_W / 2.0, WINDOW_H / 2.0 - 30  # center position
    tilt_x_smooth = 0.0  # smoothed tilt
    tilt_y_smooth = 0.0
    mode = 1  # 1=tilt, 2=tilt+bounce, 3=tilt+bounce+squash

    # Bounce state
    bounce_y = cy
    bounce_vy = 0.0
    squash = 0.0
    bouncing = False

    mode_labels = {
        1: "Card Tilt",
        2: "Tilt + Bounce",
        3: "Tilt + Bounce + Squash",
    }

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    bounce_y = 80.0
                    bounce_vy = 0.0
                    squash = 0.0
                    bouncing = True
                elif event.key == pygame.K_1:
                    mode = 1
                    bouncing = False
                    bounce_y = cy
                elif event.key == pygame.K_2:
                    mode = 2
                    bounce_y = 80.0
                    bounce_vy = 0.0
                    bouncing = True
                elif event.key == pygame.K_3:
                    mode = 3
                    bounce_y = 80.0
                    bounce_vy = 0.0
                    squash = 0.0
                    bouncing = True

        # --- Mouse tilt ---
        mx, my = pygame.mouse.get_pos()
        # Normalize to [-1, 1]
        target_tilt_x = (mx - WINDOW_W / 2) / (WINDOW_W / 2)
        target_tilt_y = (my - WINDOW_H / 2) / (WINDOW_H / 2)
        # Clamp
        target_tilt_x = max(-1, min(1, target_tilt_x))
        target_tilt_y = max(-1, min(1, target_tilt_y))
        # Smooth
        tilt_x_smooth += (target_tilt_x - tilt_x_smooth) * TILT_SMOOTHING * dt
        tilt_y_smooth += (target_tilt_y - tilt_y_smooth) * TILT_SMOOTHING * dt

        # --- Bounce physics ---
        if mode >= 2 and bouncing:
            bounce_vy += GRAVITY * dt
            bounce_y += bounce_vy * dt
            obj_bottom = bounce_y + img.get_height() / 2
            if obj_bottom >= ground_y:
                bounce_y = ground_y - img.get_height() / 2
                bounce_vy = -abs(bounce_vy) * BOUNCE_DAMPING
                if mode >= 3:
                    squash = SQUASH_AMOUNT
                if abs(bounce_vy) < 30:
                    bounce_vy = 0
                    bouncing = False
            current_y = bounce_y
        else:
            current_y = cy

        # Squash decay
        if squash > 0.001:
            squash *= math.exp(-SQUASH_DECAY * dt)
        else:
            squash = 0.0

        # --- Render ---
        screen.fill(bg_color)

        # Ground line
        pygame.draw.line(screen, (200, 200, 200), (0, ground_y), (WINDOW_W, ground_y), 1)

        # Ground shadow
        if args.shadow:
            height_above = ground_y - (current_y + img.get_height() / 2)
            draw_shadow(screen, cx, ground_y, img.get_width(), tilt_x_smooth, height_above)

        # Apply squash to base image
        if squash > 0.01:
            sx = 1.0 + squash
            sy = 1.0 - squash
            sq_w = max(1, int(img.get_width() * sx))
            sq_h = max(1, int(img.get_height() * sy))
            base_img = pygame.transform.smoothscale(img, (sq_w, sq_h))
        else:
            base_img = img

        # Perspective warp
        warped, off_x, off_y = perspective_warp(base_img, tilt_x_smooth, tilt_y_smooth)

        # Draw edge layers (fake 3D depth)
        edge_x = int(cx - warped.get_width() / 2 - off_x)
        edge_y = int(current_y - warped.get_height() / 2 - off_y)

        # Bottom-align if squashing
        if squash > 0.01:
            edge_y = int(ground_y - warped.get_height() - off_y)

        draw_edge_layers(screen, warped, edge_x, edge_y, tilt_x_smooth * 3, tilt_y_smooth * 3)

        # Draw main warped image
        screen.blit(warped, (edge_x, edge_y))

        # Lighting overlay
        light_overlay = make_lighting_overlay_fast(
            warped.get_width(), warped.get_height(),
            tilt_x_smooth, tilt_y_smooth
        )
        screen.blit(light_overlay, (edge_x, edge_y))

        # HUD
        label = font.render(
            f"Mode {mode}: {mode_labels[mode]}  |  SPACE=drop  1/2/3=mode  Q=quit  |  "
            f"Tilt: ({tilt_x_smooth:+.2f}, {tilt_y_smooth:+.2f})",
            True, (140, 140, 140))
        screen.blit(label, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
