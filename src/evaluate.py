"""Evaluation Pipeline for Crop Disease Detection CNN.
Aligns with PRD Section 13 and FR-09 (Accuracy, Precision, Recall, F1-score, Confusion Matrix).
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from src.config import (
    MODEL_PATH,
    BEST_MODEL_PATH,
    EVALUATION_DIR,
    CLASS_INDICES_PATH,
    BATCH_SIZE
)
from src.dataset import load_hf_plantvillage, hf_dataset_to_tf_dataset


def evaluate_model(model_path=None, max_test_samples=None):
    """Evaluates the trained CNN model on unseen test split and outputs standard metrics."""
    print("=" * 60)
    print("Starting Model Evaluation on Unseen Test Dataset")
    print("=" * 60)

    # 1. Determine model path
    if model_path is None:
        if BEST_MODEL_PATH.exists():
            model_path = BEST_MODEL_PATH
        elif MODEL_PATH.exists():
            model_path = MODEL_PATH
        else:
            raise FileNotFoundError("No trained model found. Please run src/train.py first.")

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 2. Load dataset
    _, _, test_hf, labels = load_hf_plantvillage()
    if max_test_samples:
        test_hf = test_hf.select(range(min(max_test_samples, len(test_hf))))

    print(f"Evaluating on {len(test_hf)} test images...")
    test_ds = hf_dataset_to_tf_dataset(test_hf, batch_size=BATCH_SIZE, is_training=False, num_classes=len(labels))

    # 3. Predict on test set
    y_true = []
    y_pred_probs = []

    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(batch_y.numpy())

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # 4. Calculate Metrics (FR-09)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print("\n--- Summary Performance Metrics ---")
    print(f"Test Accuracy:         {acc * 100:.2f}%")
    print(f"Weighted Precision:    {precision * 100:.2f}%")
    print(f"Weighted Recall:       {recall * 100:.2f}%")
    print(f"Weighted F1-Score:     {f1 * 100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1 * 100:.2f}%")

    # 5. Detailed Classification Report
    present_indices = sorted(list(set(y_true) | set(y_pred)))
    target_names = [labels[i] for i in present_indices]
    report_dict = classification_report(y_true, y_pred, labels=present_indices, target_names=target_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=present_indices, target_names=target_names, zero_division=0)

    # Save reports
    with open(EVALUATION_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)

    with open(EVALUATION_DIR / "evaluation_summary.txt", "w") as f:
        f.write("=== Crop Disease Detection Model Evaluation ===\n")
        f.write(f"Model File: {model_path}\n")
        f.write(f"Test Samples: {len(y_true)}\n\n")
        f.write(f"Accuracy:           {acc * 100:.2f}%\n")
        f.write(f"Weighted Precision: {precision * 100:.2f}%\n")
        f.write(f"Weighted Recall:    {recall * 100:.2f}%\n")
        f.write(f"Weighted F1-Score:  {f1 * 100:.2f}%\n\n")
        f.write("--- Per-Class Breakdown ---\n")
        f.write(report_text)

    # 6. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred, labels=present_indices)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - Crop Disease Classification", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Disease Class", fontsize=11)
    plt.ylabel("Ground Truth Disease Class", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")
    print(f"Evaluation results saved to {EVALUATION_DIR}")


if __name__ == "__main__":
    evaluate_model()
