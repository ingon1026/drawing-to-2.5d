"""
AR-BOOK Drawing Classifier — YOLO26 Camera Demo

Pipeline:
  1. Shadow removal (divide-by-blur)
  2. YOLO26 inference
  3. Majority vote

Press 'q' to quit.
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

MODEL_PATH = "runs/classify/outputs/yolo26_cls/weights/best.pt"


def remove_shadow(frame):
    """조명 정규화: 원본을 blur로 나눠서 그림자/조명 편차 제거."""
    img = frame.astype(np.float32)
    bg = cv2.GaussianBlur(img, (51, 51), 0)
    bg[bg < 1] = 1
    result = cv2.divide(img, bg, scale=255)
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    model = YOLO(MODEL_PATH)
    class_names = model.names

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
        print("No camera found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vote_buffer = deque(maxlen=7)

    print(f"Classes: {class_names}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Shadow removal
        processed = remove_shadow(frame)

        # 2. YOLO inference
        results = model(processed, verbose=False)
        probs = results[0].probs

        if probs is None:
            continue

        p = probs.data.cpu().numpy()
        class_idx = np.argmax(p)
        confidence = p[class_idx]

        # 3. Majority vote (최근 7프레임)
        vote_buffer.append(class_idx)
        counts = np.bincount(list(vote_buffer), minlength=len(class_names))
        voted_idx = np.argmax(counts)
        voted_name = class_names[voted_idx]

        # Display
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
        cv2.putText(frame, f"{voted_name} {confidence:.0%}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        for i, cls in class_names.items():
            bar_w = int(p[i] * 200)
            y = 90 + i * 30
            c = (0, 255, 0) if i == voted_idx else (180, 180, 180)
            cv2.putText(frame, f"{cls}: {p[i]:.1%}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
            cv2.rectangle(frame, (200, y - 13), (200 + bar_w, y), c, -1)

        cv2.imshow("YOLO26 Drawing Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
