"""
Augment underrepresented classes to target count.
Applies random transforms to existing images to fill the gap.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path

import sys

TRAIN_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/train"
TARGET_COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SEED = 42


def augment_image(img):
    """Apply random augmentation to a single image."""
    h, w = img.shape[:2]

    # Random rotation (-15 ~ 15 degrees)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # Random brightness/contrast
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Random horizontal flip (50%)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Random scale (0.8 ~ 1.2)
    scale = random.uniform(0.85, 1.15)
    M_scale = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    img = cv2.warpAffine(img, M_scale, (w, h), borderValue=(255, 255, 255))

    # Random Gaussian noise (30%)
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    for cls_name in sorted(os.listdir(TRAIN_DIR)):
        cls_dir = os.path.join(TRAIN_DIR, cls_name)
        if not os.path.isdir(cls_dir) or cls_name.endswith("_backup"):
            continue

        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current = len(files)

        if current >= TARGET_COUNT:
            print(f"{cls_name}: {current} images (skip)")
            continue

        need = TARGET_COUNT - current
        print(f"{cls_name}: {current} → {TARGET_COUNT} (augmenting {need})")

        for i in range(need):
            src_file = random.choice(files)
            src_path = os.path.join(cls_dir, src_file)
            img = cv2.imread(src_path)
            if img is None:
                continue

            aug_img = augment_image(img)
            aug_name = f"aug_{i:05d}.png"
            cv2.imwrite(os.path.join(cls_dir, aug_name), aug_img)

        final = len(os.listdir(cls_dir))
        print(f"  done: {final} images")


if __name__ == "__main__":
    main()
