"""
drawing-2p5d — Live Demo: Camera → Realtime Preview → Click → 2.5D Viewer

Flow:
    1. Camera feed with realtime contour-based segment overlay (30fps)
    2. Hover over segments to highlight them
    3. Click a segment to select it
    4. Pipeline runs: segment → depth → normal → export
    5. 2.5D viewer launches with the extracted object bouncing

Keys (Camera):
    CLICK      — select highlighted segment
    Q/ESC      — quit

Keys (Viewer):
    SPACE      — re-drop (bounce again)
    R          — back to camera
    Q/ESC      — quit
"""

import os
import sys
import math
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pyrealsense2 as rs
import warnings
warnings.filterwarnings("ignore")

import config
import normalize
import segment
import postprocess
import depth
import export
import auto_segment

# === Window ===
WINDOW_W = 800
WINDOW_H = 600
FPS = 60

# === Bounce ===
GRAVITY = 1200.0
BOUNCE_DAMPING = 0.7
SQUASH_AMOUNT = 0.3
SQUASH_DECAY = 8.0
SWAY_AMPLITUDE = 20.0
SWAY_SPEED = 1.5


# ──────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────

click_point = None
mouse_pos = (0, 0)


def on_mouse(event, x, y, flags, param):
    global click_point, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


def open_camera():
    """Open RealSense D455 RGB stream via pyrealsense2.

    Returns (pipeline, align) tuple, or None if no device found.
    Use read_frame(pipeline, align) to get BGR frames.
    """
    try:
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(rs_config)
        print("RealSense D455 RGB stream opened")
        return pipeline
    except Exception as e:
        print(f"RealSense failed: {e}")
        # Fallback to OpenCV
        for idx in [2, 4, 0, 1, 3, 5]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Fallback camera: /dev/video{idx}")
                    return cap
                cap.release()
        return None


def read_frame(cam):
    """Read a BGR frame from either RealSense pipeline or OpenCV VideoCapture."""
    if isinstance(cam, rs.pipeline):
        frames = cam.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())
    else:
        return cam.read()


def close_camera(cam):
    """Close camera resource."""
    if isinstance(cam, rs.pipeline):
        cam.stop()
    else:
        cam.release()


# ──────────────────────────────────────
# Phase 1: Camera + Realtime Contour Preview
# ──────────────────────────────────────

def camera_phase():
    """Camera feed with realtime contour overlay.

    Returns (image_bgr, norm_x, norm_y) on click, or None to quit.
    """
    global click_point, mouse_pos
    click_point = None

    cam = open_camera()
    if cam is None:
        print("No camera found!")
        return None

    win_name = "2.5D Live - Click on drawing"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    print("Point camera at drawing. Segments shown in realtime.")
    print("Hover to highlight, click to select.")

    while True:
        ret, frame = read_frame(cam)
        if not ret:
            continue

        # --- Realtime contour detection ---
        masks = auto_segment.generate_masks_contour(frame)

        # --- Find hover segment ---
        mx, my = mouse_pos
        hover_idx = auto_segment.find_segment_at(masks, mx, my)

        # --- Build display ---
        if masks:
            display = auto_segment.masks_to_overlay(
                frame, masks, alpha=0.35, highlight_idx=hover_idx)
            cv2.putText(display, f"{len(masks)} objects",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            if hover_idx >= 0:
                cv2.putText(display, f"Click to select #{hover_idx + 1}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
        else:
            display = frame.copy()
            cv2.putText(display, "No drawings detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

        cv2.putText(display, "Q=quit",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # --- Handle click ---
        if click_point is not None and masks:
            cx, cy = click_point
            seg_idx = auto_segment.find_segment_at(masks, cx, cy)

            if seg_idx >= 0:
                scx, scy = auto_segment.get_segment_center(masks[seg_idx])
                h, w = frame.shape[:2]

                cv2.drawMarker(display, (scx, scy), (0, 0, 255),
                               cv2.MARKER_CROSS, 30, 2)
                cv2.putText(display, "Processing...",
                            (10, display.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(win_name, display)
                cv2.waitKey(1)

                close_camera(cam)
                cv2.destroyAllWindows()
                return frame.copy(), scx / w, scy / h
            else:
                click_point = None

        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            close_camera(cam)
            cv2.destroyAllWindows()
            return None

    close_camera(cam)
    cv2.destroyAllWindows()
    return None


# ──────────────────────────────────────
# Phase 2: Pipeline processing
# ──────────────────────────────────────

def run_pipeline(image_bgr, norm_x, norm_y):
    """Run segmentation + depth + normal + export. Returns output paths dict or None."""
    t0 = time.time()

    print(f"\n[1/5] Normalizing ...")
    image = normalize.white_balance(image_bgr)

    print(f"[2/5] Segmenting at ({norm_x:.2f}, {norm_y:.2f}) ...")
    segment.download_model_if_needed()
    segmenter = segment.load_segmenter()
    raw_mask = segment.segment_at_point(segmenter, image, norm_x, norm_y)
    fg = raw_mask.sum() / (raw_mask.size * 255) * 100
    print(f"  Foreground: {fg:.1f}%")

    if fg < 0.5:
        print("  WARNING: Almost no foreground detected.")
        return None

    print(f"[3/5] Postprocessing mask ...")
    mask = postprocess.clean_mask(raw_mask)

    print(f"[4/5] Estimating depth & normal ...")
    depth_map = depth.estimate_depth(image, mask)
    normal_map = depth.depth_to_normal(depth_map, strength=config.NORMAL_STRENGTH)
    normal_map[mask == 0] = (128, 128, 255)

    print(f"[5/5] Exporting ...")
    out = config.OUTPUT_DIR
    paths = {
        "mask": export.export_mask(mask, out),
        "object": export.export_object(image, mask, out),
        "depth": export.export_depth(depth_map, out),
        "normal": export.export_normal(normal_map, out),
    }

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s\n")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    return paths


# ──────────────────────────────────────
# Phase 3: 2.5D Pygame Viewer
# ──────────────────────────────────────

def viewer_phase(object_path, depth_path, normal_path):
    """Show the extracted object bouncing with 2.5D effect. Returns 'retake' or 'quit'."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("2.5D Live Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    obj_surf = pygame.image.load(object_path).convert_alpha()
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    max_h = int(WINDOW_H * 0.45)
    ow, oh = obj_surf.get_size()
    if oh > max_h:
        ratio = max_h / oh
        obj_surf = pygame.transform.smoothscale(obj_surf, (int(ow * ratio), max_h))

    sprite_w, sprite_h = obj_surf.get_size()
    ground_y = WINDOW_H - 60
    avg_depth = depth_img[depth_img > 0].mean() / 255.0 if (depth_img > 0).any() else 0.5

    cx = WINDOW_W / 2.0
    cy = 60.0
    vy = 0.0
    squash = 0.0
    sway_phase = 0.0
    bg_color = (240, 245, 250)

    while True:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    return "quit"
                elif event.key == pygame.K_r:
                    pygame.quit()
                    return "retake"
                elif event.key == pygame.K_SPACE:
                    cy = 60.0
                    vy = 0.0
                    squash = 0.0

        vy += GRAVITY * dt
        cy += vy * dt
        sway_phase += SWAY_SPEED * dt * 2 * math.pi
        sway_offset = math.sin(sway_phase) * SWAY_AMPLITUDE

        bottom = cy + sprite_h / 2
        if bottom >= ground_y:
            cy = ground_y - sprite_h / 2
            vy = -abs(vy) * BOUNCE_DAMPING
            squash = SQUASH_AMOUNT
            if abs(vy) < 30:
                vy = 0

        if squash > 0.001:
            squash *= math.exp(-SQUASH_DECAY * dt)
        else:
            squash = 0.0

        screen.fill(bg_color)
        pygame.draw.line(screen, (200, 200, 200), (0, ground_y), (WINDOW_W, ground_y), 1)

        # Shadow
        height_above = max(0, ground_y - (cy + sprite_h / 2))
        spread = max(0.3, 1.0 - height_above / 400.0)
        shadow_w = int(sprite_w * spread * (0.6 + avg_depth * 0.4))
        shadow_h = max(4, int(14 * spread))
        shadow_surf = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, int(60 * spread)),
                            (0, 0, shadow_w, shadow_h))
        screen.blit(shadow_surf,
                    (int(cx + sway_offset - shadow_w / 2), int(ground_y - shadow_h / 2)))

        # Sprite
        sx_f = 1.0 + squash
        sy_f = 1.0 - squash
        draw_w = max(1, int(sprite_w * sx_f))
        draw_h = max(1, int(sprite_h * sy_f))
        scaled = pygame.transform.smoothscale(obj_surf, (draw_w, draw_h))
        rx = int(cx + sway_offset - draw_w / 2)
        ry = int(cy + sprite_h / 2 - draw_h)
        screen.blit(scaled, (rx, ry))

        hud = font.render("SPACE=drop  R=retake camera  Q=quit", True, (150, 150, 150))
        screen.blit(hud, (10, 10))
        pygame.display.flip()

    pygame.quit()
    return "quit"


# ──────────────────────────────────────
# Main loop
# ──────────────────────────────────────

def main():
    print("=" * 50)
    print("  2.5D Live Demo")
    print("  Camera → Realtime Preview → Click → 2.5D Viewer")
    print("=" * 50)

    # Pre-load pipeline models (not SAM2 — contour needs no model)
    print("\nPre-loading models ...")
    segment.download_model_if_needed()
    segment.load_segmenter()
    print("  Segmenter ready")
    depth.load_depth_model()
    print("  Depth model ready")
    print()

    while True:
        result = camera_phase()
        if result is None:
            break
        image_bgr, norm_x, norm_y = result

        paths = run_pipeline(image_bgr, norm_x, norm_y)
        if paths is None:
            print("Segmentation failed. Try again.")
            continue

        action = viewer_phase(paths["object"], paths["depth"], paths["normal"])
        if action == "quit":
            break
        elif action == "retake":
            continue

    print("Bye!")


if __name__ == "__main__":
    main()
