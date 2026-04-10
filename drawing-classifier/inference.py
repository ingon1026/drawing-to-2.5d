"""
AR-BOOK Drawing Classifier — TFLite Inference

Single-image inference using the exported TFLite model.
"""

import sys
import numpy as np
from PIL import Image

os_env_set = True
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import config


def load_tflite_model(tflite_path: str):
    """Load TFLite model and allocate tensors."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image to match training pipeline."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.array(img, dtype=np.float32)  # [0, 255] as-is, matches training

    # NHWC format (TFLite expects this)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(interpreter, input_data: np.ndarray) -> tuple[str, float]:
    """Run inference and return predicted class + confidence."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Softmax
    exp = np.exp(output - np.max(output))
    probs = exp / exp.sum()

    class_idx = np.argmax(probs)
    confidence = probs[class_idx]

    return config.CLASSES[class_idx], float(confidence)


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [tflite_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    tflite_path = sys.argv[2] if len(sys.argv) > 2 else f"{config.TFLITE_DIR}/drawing_classifier.tflite"

    interpreter = load_tflite_model(tflite_path)
    input_data = preprocess_image(image_path)
    class_name, confidence = predict(interpreter, input_data)

    print(f"Prediction: {class_name} ({confidence:.2%})")


if __name__ == "__main__":
    main()
