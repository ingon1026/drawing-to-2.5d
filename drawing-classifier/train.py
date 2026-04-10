"""
AR-BOOK Drawing Classifier — Training Script (TF/Keras)

2-phase training:
  Phase 1: Freeze backbone, train classifier head (fast convergence)
  Phase 2: Unfreeze backbone, fine-tune entire model (domain adaptation)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import config
from dataset import get_datasets


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # start frozen

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Dense(config.NUM_CLASSES),
    ])
    return model


def main():
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    train_ds, val_ds, class_names = get_datasets()
    print(f"Class order: {class_names}")

    model = build_model()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    # === Phase 1: Frozen backbone ===
    print("\n" + "=" * 60)
    print("Phase 1: Frozen backbone — training classifier head")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.PHASE1_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE1_EPOCHS,
        callbacks=callbacks,
    )

    # === Phase 2: Unfreeze backbone ===
    print("\n" + "=" * 60)
    print("Phase 2: Unfrozen backbone — fine-tuning entire model")
    print("=" * 60)

    model.layers[0].trainable = True  # unfreeze MobileNetV3 backbone

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.PHASE2_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    total_epochs = config.PHASE1_EPOCHS + config.PHASE2_EPOCHS
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=config.PHASE1_EPOCHS,
        epochs=total_epochs,
        callbacks=callbacks,
    )

    # Load best and report
    best_model = tf.keras.models.load_model(checkpoint_path)
    _, val_acc = best_model.evaluate(val_ds, verbose=0)
    print(f"\nBest val accuracy: {val_acc:.4f}")
    print(f"Model saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
