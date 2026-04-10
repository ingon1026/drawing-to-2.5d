"""
AR-BOOK Drawing Classifier — TFLite Export

Keras model → TFLite. No conversion headaches.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import config


def main():
    os.makedirs(config.TFLITE_DIR, exist_ok=True)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.keras")
    print(f"Loading model: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)

    # Convert to TFLite with dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(config.TFLITE_DIR, "drawing_classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nExport complete.")
    print(f"  TFLite: {tflite_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
