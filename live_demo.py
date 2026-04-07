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
BOUNCE_DAMPING = 0.65
SQUASH_AMOUNT = 0.15
SQUASH_DECAY = 10.0
SWAY_AMPLITUDE = 10.0
SWAY_SPEED = 1.0


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
                contour_mask = auto_segment.contour_mask_to_uint8(masks[seg_idx])
                return frame.copy(), scx / w, scy / h, contour_mask
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

def run_pipeline(image_bgr, contour_mask):
    """Run contour-based segmentation + depth + normal + export."""
    t0 = time.time()

    print(f"\n[1/4] Normalizing ...")
    image = normalize.white_balance(image_bgr)

    print(f"[2/4] Applying contour mask ...")
    h, w = image.shape[:2]
    raw_mask = cv2.resize(contour_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    fg = raw_mask.sum() / (raw_mask.size * 255) * 100
    print(f"  Foreground: {fg:.1f}%")

    if fg < 0.1:
        print("  WARNING: Empty mask.")
        return None

    print(f"[3/4] Postprocessing mask ...")
    mask = postprocess.clean_mask(raw_mask)

    print(f"[4/4] Estimating depth & normal + exporting ...")
    depth_map = depth.estimate_depth(image, mask)
    normal_map = depth.depth_to_normal(depth_map, strength=config.NORMAL_STRENGTH)
    normal_map[mask == 0] = (128, 128, 255)
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

def _build_side_strip(obj_surf, thickness=10):
    """Build a side face strip from the object's silhouette.

    Creates a darkened, slightly shifted version of the outline
    that looks like the 'edge' of a thick sticker/token.
    """
    import pygame

    w, h = obj_surf.get_size()
    raw = pygame.image.tobytes(obj_surf, "RGBA")
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4)).copy()

    # Darken to ~25% brightness for side color
    side_px = pixels.copy()
    side_px[:, :, 0] = (pixels[:, :, 0] * 0.25).astype(np.uint8)
    side_px[:, :, 1] = (pixels[:, :, 1] * 0.2).astype(np.uint8)
    side_px[:, :, 2] = (pixels[:, :, 2] * 0.15).astype(np.uint8)
    # Alpha unchanged — transparent stays transparent

    side_surf = pygame.image.frombytes(side_px.tobytes(), (w, h), "RGBA")
    return side_surf


def viewer_phase(object_path, depth_path, normal_path):
    """Layered pseudo-3D viewer: top face + side face + shadow + auto tilt."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("2.5D Live Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Load assets
    obj_surf = pygame.image.load(object_path).convert_alpha()
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    # Fit to view
    max_h = int(WINDOW_H * 0.4)
    ow, oh = obj_surf.get_size()
    if oh > max_h:
        ratio = max_h / oh
        obj_surf = pygame.transform.smoothscale(obj_surf, (int(ow * ratio), max_h))

    sprite_w, sprite_h = obj_surf.get_size()
    ground_y = WINDOW_H - 70

    # Pre-build side face
    side_surf = _build_side_strip(obj_surf)
    THICKNESS = 5  # side face pixel thickness (subtle)

    # Physics
    cx = WINDOW_W / 2.0
    cy = 60.0
    vy = 0.0
    squash = 0.0
    sway_phase = 0.0
    tilt = 0.0       # current tilt angle in degrees
    tilt_vel = 0.0    # tilt angular velocity
    bg_color = (240, 245, 250)

    # Tilt config — subtle, natural
    TILT_BOUNCE = 3.0     # tilt kick on bounce (degrees)
    TILT_DAMPING = 0.95
    TILT_SPRING = -5.0    # return-to-zero spring

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
                    tilt = 0.0
                    tilt_vel = 0.0

        # --- Physics ---
        vy += GRAVITY * dt
        cy += vy * dt

        sway_phase += SWAY_SPEED * dt * 2 * math.pi
        sway_offset = math.sin(sway_phase) * SWAY_AMPLITUDE

        # Bounce
        bottom = cy + sprite_h / 2
        if bottom >= ground_y:
            cy = ground_y - sprite_h / 2
            vy = -abs(vy) * BOUNCE_DAMPING
            squash = SQUASH_AMOUNT
            # Tilt kick on bounce (alternating direction)
            tilt_vel += TILT_BOUNCE * (1.0 if math.sin(sway_phase) > 0 else -1.0)
            if abs(vy) < 30:
                vy = 0

        # Squash decay
        if squash > 0.001:
            squash *= math.exp(-SQUASH_DECAY * dt)
        else:
            squash = 0.0

        # Tilt spring + damping (auto-oscillate, return to 0)
        tilt_vel += TILT_SPRING * tilt * dt
        tilt_vel *= TILT_DAMPING
        tilt += tilt_vel * dt * 60
        tilt = max(-8, min(8, tilt))  # clamp — subtle range

        # --- Squash-stretch ---
        sx_f = 1.0 + squash
        sy_f = 1.0 - squash
        draw_w = max(1, int(sprite_w * sx_f))
        draw_h = max(1, int(sprite_h * sy_f))

        # Side thickness varies with tilt
        side_visible = abs(tilt) / 20.0  # 0~1
        side_pixels = int(THICKNESS * side_visible + 2)

        # --- Draw ---
        screen.fill(bg_color)
        pygame.draw.line(screen, (210, 210, 210), (0, ground_y), (WINDOW_W, ground_y), 1)

        # Position (bottom-aligned)
        rx = int(cx + sway_offset - draw_w / 2)
        ry = int(cy + sprite_h / 2 - draw_h)

        # 1. Shadow
        height_above = max(0, ground_y - (cy + sprite_h / 2))
        spread = max(0.3, 1.0 - height_above / 400.0)
        shadow_w = int(sprite_w * spread * 0.9)
        shadow_h = max(6, int(16 * spread))
        shadow_surf = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
        shadow_alpha = int(50 * spread)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, shadow_alpha),
                            (0, 0, shadow_w, shadow_h))
        screen.blit(shadow_surf,
                    (int(cx + sway_offset - shadow_w / 2), int(ground_y - shadow_h / 2)))

        # 2. Side face (multiple offset layers for thickness)
        scaled_side = pygame.transform.smoothscale(side_surf, (draw_w, draw_h))
        for i in range(side_pixels, 0, -1):
            # Offset direction based on tilt
            offset_x = int(tilt * 0.15 * i)
            offset_y = i  # always slightly below
            screen.blit(scaled_side, (rx + offset_x, ry + offset_y))

        # 3. Top face (main sprite)
        scaled_top = pygame.transform.smoothscale(obj_surf, (draw_w, draw_h))
        screen.blit(scaled_top, (rx, ry))

        # HUD
        hud = font.render("SPACE=drop  R=retake  Q=quit", True, (150, 150, 150))
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

    # Pre-load depth model (contour needs no model)
    print("\nPre-loading models ...")
    depth.load_depth_model()
    print("  Depth model ready")
    print()

    while True:
        result = camera_phase()
        if result is None:
            break
        image_bgr, _, _, contour_mask = result

        paths = run_pipeline(image_bgr, contour_mask)
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
