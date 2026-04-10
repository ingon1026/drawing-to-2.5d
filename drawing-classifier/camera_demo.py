"""
AR-BOOK Drawing Classifier — Real-time Camera Demo

Model is trained with synthetic domain randomization,
so minimal camera preprocessing needed.
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

CONFIDENCE_THRESHOLD = 0.6


def main():
    # Load TFLite model
    tflite_path = os.path.join(config.TFLITE_DIR, "drawing_classifier.tflite")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Camera
    # Try multiple camera indices
    cap = None
    for idx in [4, 0, 2, 1, 3, 5]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Camera opened: /dev/video{idx}")
                break
            cap.release()
    else:
        print("No camera found. Reconnect and try again.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Classes: {config.CLASSES}")
    print("Press 'q' to quit")

    smooth_probs = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES
    alpha = 0.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, w = frame.shape[:2]

        # Minimal preprocessing: just resize and color convert
        img = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_data = np.expand_dims(img_rgb, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        # Softmax + smoothing
        exp = np.exp(output - np.max(output))
        probs = exp / exp.sum()
        smooth_probs = alpha * probs + (1 - alpha) * smooth_probs

        class_idx = np.argmax(smooth_probs)
        class_name = config.CLASSES[class_idx]
        confidence = smooth_probs[class_idx]

        # Confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            label = "no drawing detected"
            label_color = (0, 0, 255)  # red
        else:
            label = f"{class_name} {confidence:.1%}"
            label_color = (0, 255, 0)  # green

        # Display
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

        # Class bars
        for i, cls in enumerate(config.CLASSES):
            bar_width = int(smooth_probs[i] * 200)
            y = 75 + i * 28
            color = (0, 255, 0) if i == class_idx and confidence >= CONFIDENCE_THRESHOLD else (180, 180, 180)
            cv2.putText(frame, f"{cls}: {smooth_probs[i]:.1%}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(frame, (200, y - 13), (200 + bar_width, y), color, -1)

        # Preview: what model sees (top-right)
        preview = cv2.resize(img, (140, 140))
        frame[10:150, w - 150:w - 10] = preview

        cv2.imshow("Drawing Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
