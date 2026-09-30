"""Configuration settings and hyperparameters for Crop Disease Detection project.
Aligns with PRD Sections 11, 12, 13, 16.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "PlantVillage"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "crop_disease_cnn.keras"
BEST_MODEL_PATH = MODEL_DIR / "crop_disease_cnn_best.keras"
AADITYA_MODEL_PATH = MODEL_DIR / "plant_disease_model.h5"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"
DISEASE_INFO_PATH = MODEL_DIR / "disease_info.json"
EVALUATION_DIR = PROJECT_ROOT / "model" / "evaluation"
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"

# Hugging Face Dataset Details
HF_DATASET_NAME = "mohanty/PlantVillage"
HF_CONFIG_NAME = "color"

# Image Preprocessing & Model Hyperparameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

BATCH_SIZE = 32
EPOCHS = 20
INITIAL_LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Splits
VAL_SPLIT_RATIO = 0.15

# Ensure output directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
