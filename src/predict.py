"""Inference and Prediction Pipeline for Crop Disease Detection.
Aligns with PRD FR-04, FR-05, FR-06, FR-08.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from src.config import (
    MODEL_PATH,
    BEST_MODEL_PATH,
    CLASS_INDICES_PATH,
    DISEASE_INFO_PATH,
    INPUT_SHAPE
)
from src.dataset import preprocess_image

_LOADED_MODEL = None
_CLASS_INDICES = None
_DISEASE_INFO = None


def load_model_and_metadata():
    """Loads the trained CNN model, class indices, and disease advisory information."""
    global _LOADED_MODEL, _CLASS_INDICES, _DISEASE_INFO

    # 1. Load class indices
    if CLASS_INDICES_PATH.exists():
        with open(CLASS_INDICES_PATH, "r") as f:
            raw_indices = json.load(f)
            # Ensure keys are integers
            _CLASS_INDICES = {int(k): v for k, v in raw_indices.items()}
    else:
        _CLASS_INDICES = {}

    # 2. Load disease knowledge base
    if DISEASE_INFO_PATH.exists():
        with open(DISEASE_INFO_PATH, "r") as f:
            _DISEASE_INFO = json.load(f)
    else:
        _DISEASE_INFO = {}

    # 3. Load trained model weights
    target_model_file = None
    if BEST_MODEL_PATH.exists():
        target_model_file = BEST_MODEL_PATH
    elif MODEL_PATH.exists():
        target_model_file = MODEL_PATH

    if target_model_file and _LOADED_MODEL is None:
        try:
            print(f"Loading trained CNN model from {target_model_file}...")
            _LOADED_MODEL = tf.keras.models.load_model(target_model_file)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {target_model_file}: {e}")
            _LOADED_MODEL = None

    return _LOADED_MODEL, _CLASS_INDICES, _DISEASE_INFO


def format_class_name(raw_name):
    """Formats raw class name (e.g., 'Tomato___Early_blight') into clean crop & condition."""
    if "___" in raw_name:
        crop_part, disease_part = raw_name.split("___", 1)
    else:
        crop_part, disease_part = "Unknown", raw_name

    crop_clean = crop_part.replace("_", " ").replace("(", "").replace(")", "").strip()
    disease_clean = disease_part.replace("_", " ").strip()
    is_healthy = "healthy" in disease_clean.lower()

    return crop_clean, disease_clean, is_healthy


def predict_crop_disease(image_path_or_pil, top_k=3):
    """Performs end-to-end prediction on a crop leaf image (PRD FR-05, FR-06).
    
    Args:
        image_path_or_pil: Path to leaf image or PIL Image object.
        top_k: Number of highest-probability classes to return.
        
    Returns:
        dict: Prediction results including predicted class, crop, disease, 
              confidence %, top_k probabilities, and agricultural recommendations.
    """
    model, class_indices, disease_info = load_model_and_metadata()

    # Preprocess image with identical pipeline as training (PRD FR-03)
    preprocessed_img = preprocess_image(image_path_or_pil)

    if model is not None:
        # Model forward pass
        predictions = model.predict(preprocessed_img, verbose=0)[0]
    else:
        # Fallback if model is not yet trained
        num_classes = len(class_indices) if class_indices else 38
        # Create deterministic pseudo-probabilities based on pixel statistics for demonstration
        rng = np.random.RandomState(int(np.sum(preprocessed_img) * 1000) % 10000)
        raw_scores = rng.exponential(scale=1.0, size=num_classes)
        predictions = raw_scores / np.sum(raw_scores)

    # Find highest probability index
    top_index = int(np.argmax(predictions))
    top_confidence = float(predictions[top_index])

    raw_label = class_indices.get(top_index, f"Class_{top_index}")
    crop_name, disease_name, is_healthy = format_class_name(raw_label)

    # Top-K predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = []
    for idx in top_indices:
        lbl = class_indices.get(int(idx), f"Class_{idx}")
        c, d, _ = format_class_name(lbl)
        top_predictions.append({
            "class_raw": lbl,
            "crop": c,
            "condition": d,
            "probability": round(float(predictions[idx]) * 100, 2)
        })

    # Retrieve advisory information from knowledge base
    info = disease_info.get(raw_label, {
        "crop": crop_name,
        "disease": disease_name,
        "cause": "Fungal/Bacterial pathogen" if not is_healthy else "None (Healthy)",
        "symptoms": "Leaf examination required" if not is_healthy else "Foliage shows uniform pigmentation with no lesions or chlorosis.",
        "management": "Consult local agricultural extension service for specific recommendations." if not is_healthy else "Maintain regular irrigation, soil nutrients, and scout periodically.",
        "disclaimer": "This is a computer vision decision-support tool, not a professional agricultural diagnosis."
    })

    return {
        "success": True,
        "raw_label": raw_label,
        "crop": crop_name,
        "condition": disease_name,
        "is_healthy": is_healthy,
        "confidence": round(top_confidence * 100, 2),
        "top_predictions": top_predictions,
        "advisory": info,
        "is_model_trained": model is not None
    }
