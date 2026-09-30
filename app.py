"""Flask Web Application for Crop Disease Detection Using CNN.
Aligns with PRD Sections 9, 14, 15, and Sprint 8 requirements.
"""

import os
import uuid
import sys
from pathlib import Path

# Silence verbose C++ TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import UPLOAD_FOLDER, CLASS_INDICES_PATH, PROJECT_ROOT
from src.predict import predict_crop_disease, load_model_and_metadata

# Initialize Flask App
app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
app.config["SECRET_KEY"] = "crop-disease-detection-secret-2026"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    """Validates allowed image file extensions (PRD FR-02)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Renders the main crop disease detection web portal."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming backend and model readiness."""
    model, class_indices, _ = load_model_and_metadata()
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(class_indices)
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    """Returns the list of supported crop disease classes."""
    _, class_indices, _ = load_model_and_metadata()
    return jsonify({
        "total_classes": len(class_indices),
        "classes": list(class_indices.values())
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, validation, preprocessing, and CNN inference.
    Aligns with PRD Section 14 (POST /predict flow) and FR-01 through FR-08.
    """
    # 1. Validate file presence
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file. Please select a crop leaf image."}), 400

    # 2. Validate file format (PRD FR-02)
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file format. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        # 3. Secure filename and save
        original_ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename.split('.')[0])}.{original_ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(file_path)

        # 4. Validate image readability (PRD FR-02)
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": False, "error": "Corrupted or unreadable image file."}), 400

        # 5. Run inference pipeline (PRD FR-03, FR-04, FR-05, FR-06)
        result = predict_crop_disease(file_path, top_k=3)
        result["image_url"] = url_for("static", filename=f"uploads/{unique_filename}")

        return jsonify(result)

    except Exception as e:
        # Graceful error handling (PRD FR-08)
        return jsonify({"success": False, "error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Crop Disease Detection Web Application")
    print("ABES Engineering College - HCLTech Industry Aligned Project")
    print("[1/2] Loading neural network model into memory...")
    print("=" * 60)
    load_model_and_metadata()
    print("[2/2] Model ready! Starting web server...")
    print("Server running at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
