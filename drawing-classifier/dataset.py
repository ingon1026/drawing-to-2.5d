"""
AR-BOOK Drawing Classifier — Dataset (TF/Keras)

Loads merged data (Quick Draw + ImageNet-Sketch).
Training: synthetic domain randomization for Quick Draw images,
          light augmentation for ImageNet-Sketch images.
Validation: clean images as-is.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

import config
from synth_augment import synthesize_camera_image


def _synth_augment_fn(image):
    """Apply synthetic augmentation to a single image."""
    img_np = image.numpy().astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Check if image is mostly black/white (Quick Draw) or complex (ImageNet-Sketch)
    unique_vals = len(np.unique(gray[::4, ::4]))  # sample for speed

    if unique_vals < 30:
        # Quick Draw: clean B/W sketch → full synthetic pipeline
        augmented = synthesize_camera_image(gray, target_size=config.IMG_SIZE)
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
    else:
        # ImageNet-Sketch: already realistic → light augmentation only
        h, w = img_np.shape[:2]

        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)  # contrast
        beta = np.random.uniform(-20, 20)     # brightness
        augmented = np.clip(img_np.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Random slight rotation
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=(230, 230, 230))

    return augmented.astype(np.float32)


def _apply_synth_augment(images, labels):
    def per_image(img):
        result = tf.py_function(_synth_augment_fn, [img], tf.float32)
        result.set_shape([config.IMG_SIZE, config.IMG_SIZE, 3])
        return result

    images = tf.cast(images, tf.float32)
    images = tf.map_fn(per_image, images, fn_output_signature=tf.float32)
    return images, labels


def get_datasets():
    """Return train and val tf.data.Datasets."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{config.DATA_DIR}/train",
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{config.DATA_DIR}/val",
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")

    # Training: synthetic augmentation
    train_ds = train_ds.map(_apply_synth_augment,
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # Validation: clean
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
