"""Model Training Pipeline for Crop Disease Detection CNN.
Aligns with PRD Section 13 (Training & Evaluation) and FR-10.
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    INPUT_SHAPE,
    BATCH_SIZE,
    EPOCHS,
    INITIAL_LEARNING_RATE,
    MODEL_PATH,
    BEST_MODEL_PATH,
    EVALUATION_DIR,
    CLASS_INDICES_PATH
)
from src.dataset import load_hf_plantvillage, hf_dataset_to_tf_dataset, create_synthetic_demo_data
from src.model import build_baseline_cnn, build_mobilenet_transfer_model


def plot_training_history(history, save_path=EVALUATION_DIR / "training_curves.png"):
    """Generates and saves Training vs Validation Accuracy & Loss curves (PRD FR-10)."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2E7D32", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#1565C0", linewidth=2)
    plt.title("Model Accuracy across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#C62828", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#EF6C00", linewidth=2)
    plt.title("Model Loss across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")


def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=INITIAL_LEARNING_RATE, use_transfer_learning=False, quick_demo=False):
    """Executes model training with callbacks, checkpointing, and evaluation plots."""
    print("=" * 60)
    print("Starting Crop Disease Detection CNN Training Pipeline")
    print("=" * 60)

    # 1. Load dataset
    if quick_demo:
        print("[Quick Demo Mode] Using fast in-memory synthetic dataset for instant pipeline verification...")
        train_ds, val_ds, test_ds, labels = create_synthetic_demo_data(num_samples=250)
        num_classes = len(labels)
        epochs = min(epochs, 2)
    else:
        train_hf, val_hf, test_hf, labels = load_hf_plantvillage()
        num_classes = len(labels)
        print(f"Total crop disease classes: {num_classes}")

        # 2. Convert to tf.data.Dataset
        train_ds = hf_dataset_to_tf_dataset(train_hf, batch_size=batch_size, is_training=True, num_classes=num_classes)
        val_ds = hf_dataset_to_tf_dataset(val_hf, batch_size=batch_size, is_training=False, num_classes=num_classes)

    # 3. Build Model
    if use_transfer_learning:
        print("Building MobileNetV2 Transfer Learning Model...")
        model = build_mobilenet_transfer_model(input_shape=INPUT_SHAPE, num_classes=num_classes)
    else:
        print("Building Baseline CNN Architecture...")
        model = build_baseline_cnn(input_shape=INPUT_SHAPE, num_classes=num_classes)

    model.summary()

    # 4. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(EVALUATION_DIR / "training_log.csv")
        )
    ]

    # 6. Fit Model
    print(f"\nBeginning training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 7. Save final model
    model.save(str(MODEL_PATH))
    print(f"Final model saved to {MODEL_PATH}")

    # 8. Generate and save training plots
    plot_training_history(history)

    print("\nTraining completed successfully.")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crop Disease Detection CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=INITIAL_LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--transfer", action="store_true", help="Use MobileNetV2 transfer learning")
    parser.add_argument("--quick-demo", action="store_true", help="Run fast verification training on small subset")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_transfer_learning=args.transfer,
        quick_demo=args.quick_demo
    )
