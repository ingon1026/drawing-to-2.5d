"""
Transform Quick Draw images to camera-style using synth_augment.
- Converts qd_* files in-place (28x28 sketches → realistic camera captures)
- Skips train/animal (manually curated)
- Removes aug_* files
- Keeps is_* files as-is
"""

import os
import cv2
import numpy as np
from synth_augment import synthesize_camera_image

DATA_DIR = "data"
SKIP = [("train", "animal")]


def process_split(split):
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        return

    for cls_name in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        if (split, cls_name) in SKIP:
            print(f"[skip] {split}/{cls_name} (curated)")
            continue

        files = sorted(os.listdir(cls_dir))

        # 1. Remove aug_* files
        aug_files = [f for f in files if f.startswith("aug_")]
        for f in aug_files:
            os.remove(os.path.join(cls_dir, f))

        # 2. Transform qd_* files
        qd_files = [f for f in files if f.startswith("qd_") and f.endswith(".png")]
        converted = 0
        for f in qd_files:
            path = os.path.join(cls_dir, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Synthesize camera-style image
            camera_img = synthesize_camera_image(img, target_size=224)
            cv2.imwrite(path, camera_img)
            converted += 1

        remaining = len(os.listdir(cls_dir))
        print(f"{split}/{cls_name}: removed {len(aug_files)} aug, converted {converted} qd → camera style, total {remaining}")


def main():
    np.random.seed(42)
    for split in ["train", "val"]:
        process_split(split)
    print("\nDone.")


if __name__ == "__main__":
    main()
