"""
AR-BOOK Drawing Classifier — YOLO26 Classification Training + Export

Usage:
    python3 train_yolo.py          # train
    python3 train_yolo.py export   # export to TFLite
"""

import sys
from ultralytics import YOLO


def train():
    model = YOLO("yolo26s-cls.pt")

    model.train(
        data="data",
        epochs=100,
        imgsz=224,
        batch=64,
        project="outputs",
        name="yolo26_cls",
        exist_ok=True,
        # --- domain gap augmentation ---
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.5,
        degrees=15,
        translate=0.15,
        scale=0.4,
        perspective=0.001,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.5,
        erasing=0.3,
    )

    print("\nTraining complete.")
    print("Best model: outputs/yolo26_cls/weights/best.pt")


def export():
    model = YOLO("models/v2_quickdraw_imagenet_aug.pt")

    # TFLite export
    model.export(format="tflite", imgsz=224)
    print("\nExport complete.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export()
    else:
        train()
