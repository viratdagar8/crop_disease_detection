"""System and Pipeline Verification Script.
Checks:
 1. Python environment and required deep learning packages.
 2. Baseline CNN model architecture building and summary.
 3. Class indices and agronomic knowledge base integrity.
 4. Mock prediction pipeline execution.
 5. Flask web application route integrity.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_INDICES_PATH, DISEASE_INFO_PATH, INPUT_SHAPE
import json


def run_checks():
    print("=" * 60)
    print("Crop Disease Detection - System Verification")
    print("=" * 60)

    # Check 1: Class Indices
    assert CLASS_INDICES_PATH.exists(), f"Missing {CLASS_INDICES_PATH}"
    with open(CLASS_INDICES_PATH) as f:
        classes = json.load(f)
    print(f"[OK] Class Indices verified: {len(classes)} classes registered.")
    assert len(classes) == 38, f"Expected 38 classes, got {len(classes)}"

    # Check 2: Disease Advisory Info
    assert DISEASE_INFO_PATH.exists(), f"Missing {DISEASE_INFO_PATH}"
    with open(DISEASE_INFO_PATH) as f:
        disease_info = json.load(f)
    print(f"[OK] Agronomic Knowledge Base verified: {len(disease_info)} conditions documented.")

    # Check 3: CNN Model Build
    from src.model import build_baseline_cnn
    model = build_baseline_cnn(input_shape=INPUT_SHAPE, num_classes=len(classes))
    print(f"[OK] Baseline CNN Architecture compiled successfully.")
    print(f"    Total parameters: {model.count_params():,}")

    # Check 4: Preprocessing and Prediction Pipeline
    from PIL import Image
    import numpy as np
    from src.predict import predict_crop_disease

    # Create dummy leaf image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(128, 128, 3) * 255))
    res = predict_crop_disease(dummy_img)
    assert res["success"] is True, "Prediction pipeline failed."
    print(f"[OK] Prediction Pipeline executed cleanly.")
    print(f"    Sample inference: {res['condition']} on {res['crop']} (Confidence: {res['confidence']}%)")

    # Check 5: Flask App Client Test
    from app import app
    client = app.test_client()
    res_home = client.get("/")
    assert res_home.status_code == 200, "Flask home route failed."
    res_health = client.get("/health")
    assert res_health.status_code == 200, "Flask health route failed."
    res_classes = client.get("/classes")
    assert res_classes.status_code == 200, "Flask classes route failed."
    print(f"[OK] Flask Web Application routes verified (GET /, /health, /classes).")

    print("\n" + "=" * 60)
    print("All system checks PASSED! Project is ready for demonstration.")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
