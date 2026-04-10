"""
AR-BOOK Drawing Classifier — Preprocessing Comparison Tool

6 methods side-by-side. Press 1-6 to toggle fullscreen on each method.
Press 'q' to quit.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf

import config

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _shadow_remove(frame):
    """Common: shadow removal via divide-by-blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    bg[bg < 1] = 1
    normalized = cv2.divide(gray, bg, scale=255)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _crop_to_drawing(binary, target_size=224):
    """Common: crop to largest contour, pad to square."""
    coords = cv2.findNonZero(binary)
    if coords is not None and len(coords) > 50:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(w, h) // 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)
        cropped = binary[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        size = max(ch, cw)
        if len(cropped.shape) == 2:
            square = np.zeros((size, size), dtype=np.uint8)
            square[(size - ch) // 2:(size - ch) // 2 + ch,
                   (size - cw) // 2:(size - cw) // 2 + cw] = cropped
        else:
            bg_val = int(np.median(cropped[:5, :5]))
            square = np.full((size, size, 3), bg_val, dtype=np.uint8)
            square[(size - ch) // 2:(size - ch) // 2 + ch,
                   (size - cw) // 2:(size - cw) // 2 + cw] = cropped
        return cv2.resize(square, (target_size, target_size))
    return cv2.resize(binary, (target_size, target_size))


# === 6 Methods ===

def method_a_canny(frame):
    """A: Canny edge (white lines on black)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    resized = cv2.resize(edges, (config.IMG_SIZE, config.IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32), cv2.resize(edges, (200, 150))


def method_b_shadow(frame):
    """B: Shadow removal + grayscale."""
    normalized = _shadow_remove(frame)
    resized = cv2.resize(normalized, (config.IMG_SIZE, config.IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32), cv2.resize(normalized, (200, 150))


def method_c_raw(frame):
    """C: Raw color."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (config.IMG_SIZE, config.IMG_SIZE))
    preview = cv2.resize(frame, (200, 150))
    return resized.astype(np.float32), preview


def method_d_shadow_binary_crop(frame):
    """D: Shadow removal → Otsu binary → crop to drawing."""
    normalized = _shadow_remove(frame)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cropped = _crop_to_drawing(binary)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32), cv2.resize(cropped, (200, 150))


def method_e_canny_thick(frame):
    """E: Canny with thick lines + crop."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 20, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cropped = _crop_to_drawing(edges)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32), cv2.resize(cropped, (200, 150))


def method_f_shadow_canny_crop(frame):
    """F: Shadow removal → Canny → crop."""
    normalized = _shadow_remove(frame)
    edges = cv2.Canny(normalized, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    cropped = _crop_to_drawing(edges)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32), cv2.resize(cropped, (200, 150))


def classify(interpreter, inp, out, img):
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(inp[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(out[0]["index"])[0]
    exp = np.exp(output - np.max(output))
    probs = exp / exp.sum()
    idx = np.argmax(probs)
    return config.CLASSES[idx], probs[idx]


def main():
    tflite_path = os.path.join(config.TFLITE_DIR, "drawing_classifier.tflite")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    cap = None
    for idx in [4, 0, 2, 1, 3, 5]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Camera: /dev/video{idx}")
                break
            cap.release()
    else:
        print("No camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    methods = [
        ("A:Canny", method_a_canny),
        ("B:Shadow", method_b_shadow),
        ("C:Raw", method_c_raw),
        ("D:Shd+Bin+Crop", method_d_shadow_binary_crop),
        ("E:Canny Thick", method_e_canny_thick),
        ("F:Shd+Canny+Crop", method_f_shadow_canny_crop),
    ]

    print("6 methods comparison. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create canvas: 3 columns x 2 rows of previews below camera
        pw, ph = 200, 150
        canvas_h = 480 + ph * 2 + 80
        canvas_w = max(640, pw * 3)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:480, :640] = frame

        best_conf = 0
        best_name = ""
        best_cls = ""

        for i, (name, method_fn) in enumerate(methods):
            img, preview = method_fn(frame)
            cls_name, conf = classify(interpreter, inp, out, img)

            if conf > best_conf:
                best_conf = conf
                best_name = name
                best_cls = cls_name

            # Position: 3 columns x 2 rows
            col = i % 3
            row = i // 3
            x = col * pw
            y = 480 + row * (ph + 40)

            # Preview
            if len(preview.shape) == 2:
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            canvas[y:y + ph, x:x + pw] = preview

            # Label
            color = (0, 255, 0) if conf > 0.6 else (100, 100, 255)
            cv2.putText(canvas, f"{name}", (x + 2, y + ph + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(canvas, f"{cls_name} {conf:.0%}", (x + 2, y + ph + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Best result on camera frame
        color = (0, 255, 0) if best_conf > 0.6 else (0, 0, 255)
        cv2.putText(canvas, f"BEST: {best_name} -> {best_cls} {best_conf:.0%}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Preprocessing Comparison (6 methods)", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
