"""Dataset handling, preprocessing, and data pipeline for PlantVillage.
Aligns with PRD Sections 11, 13, and Sprint 2-3 requirements.
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
from PIL import Image
from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
    VAL_SPLIT_RATIO,
    RANDOM_SEED,
    CLASS_INDICES_PATH,
    RAW_DATA_DIR
)


def get_data_augmentation():
    """Returns Keras data augmentation pipeline (PRD Section 11).
    Applies rotation, flipping, and slight zoom while preserving disease lesion patterns.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED),
        tf.keras.layers.RandomRotation(0.1, seed=RANDOM_SEED),
        tf.keras.layers.RandomZoom(0.1, seed=RANDOM_SEED),
        tf.keras.layers.RandomTranslation(0.05, 0.05, seed=RANDOM_SEED),
    ], name="data_augmentation")


def preprocess_image(image_path_or_pil, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    """Preprocesses a single image for CNN input (PRD FR-03).
    Ensures identical resizing and normalization used during training.
    
    Args:
        image_path_or_pil: File path string, Path object, or PIL Image.
        target_size: Tuple (height, width).
        
    Returns:
        np.ndarray with shape (1, height, width, 3) normalized to [0, 1].
    """
    if isinstance(image_path_or_pil, (str, Path)):
        img = Image.open(image_path_or_pil).convert("RGB")
    elif isinstance(image_path_or_pil, Image.Image):
        img = image_path_or_pil.convert("RGB")
    else:
        raise ValueError("Unsupported image type. Provide a file path or PIL Image.")

    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def create_synthetic_demo_data(num_samples=200):
    """Creates a fast synthetic in-memory dataset across all 38 classes for immediate viva demo."""
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    labels = [class_indices[str(i)] for i in range(len(class_indices))]
    num_classes = len(labels)

    rng = np.random.RandomState(RANDOM_SEED)
    images = rng.uniform(0.1, 0.9, size=(num_samples, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
    sample_labels = (np.arange(num_samples) % num_classes).astype(np.int32)

    split_train = int(num_samples * 0.7)
    split_val = int(num_samples * 0.85)

    train_ds = tf.data.Dataset.from_tensor_slices((images[:split_train], sample_labels[:split_train])).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((images[split_train:split_val], sample_labels[split_train:split_val])).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((images[split_val:], sample_labels[split_val:])).batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds, labels


def load_hf_plantvillage(config_name="color"):
    """Loads the official PlantVillage dataset from Hugging Face (mohanty/PlantVillage).
    Returns train, validation, and test splits along with class labels.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print(f"Loading '{config_name}' configuration of mohanty/PlantVillage from Hugging Face...")
    try:
        # Download plant_village.py script directly from the Hugging Face repository
        script_path = hf_hub_download(repo_id="mohanty/PlantVillage", filename="plant_village.py", repo_type="dataset")
        dataset = load_dataset(script_path, config_name, trust_remote_code=True)
    except Exception as e:
        print(f"Standard loading failed ({e}). Attempting direct repo load...")
        dataset = load_dataset("mohanty/PlantVillage", config_name, trust_remote_code=True)
    
    labels = dataset["train"].features["label"].names
    
    # Save class indices for model prediction & inference
    class_indices = {i: name for i, name in enumerate(labels)}
    CLASS_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(class_indices, f, indent=4)
    print(f"Saved {len(labels)} class indices to {CLASS_INDICES_PATH}")

    # Split train set to create a dedicated validation split (PRD Section 11)
    split = dataset["train"].train_test_split(test_size=VAL_SPLIT_RATIO, seed=RANDOM_SEED)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = dataset["test"]

    print(f"Dataset Splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_ds, val_ds, test_ds, labels


def hf_dataset_to_tf_dataset(hf_split, batch_size=BATCH_SIZE, is_training=False, num_classes=38):
    """Converts a Hugging Face image dataset split into a high-performance tf.data.Dataset."""
    def generator():
        for item in hf_split:
            img = item["image"].convert("RGB").resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            label = int(item["label"])
            yield img_arr, label

    output_signature = (
        tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    tf_ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if is_training:
        aug = get_data_augmentation()
        tf_ds = tf_ds.shuffle(buffer_size=1000, seed=RANDOM_SEED)
        tf_ds = tf_ds.batch(batch_size)
        tf_ds = tf_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        tf_ds = tf_ds.batch(batch_size)

    return tf_ds.prefetch(tf.data.AUTOTUNE)


def export_hf_to_directory(output_dir=RAW_DATA_DIR, max_samples_per_class=None):
    """Optional utility to save PlantVillage locally in standard image folder structure:
    output_dir/
      train/class_name/xxx.jpg
      val/class_name/xxx.jpg
      test/class_name/xxx.jpg
    """
    from datasets import load_dataset
    from tqdm import tqdm

    output_dir = Path(output_dir)
    train_ds, val_ds, test_ds, labels = load_hf_plantvillage()
    splits = {"train": train_ds, "val": val_ds, "test": test_ds}

    for split_name, split_data in splits.items():
        print(f"Exporting {split_name} split to {output_dir / split_name}...")
        class_counts = {lbl: 0 for lbl in labels}

        for idx, item in enumerate(tqdm(split_data)):
            label_name = labels[item["label"]]
            if max_samples_per_class and class_counts[label_name] >= max_samples_per_class:
                continue

            target_dir = output_dir / split_name / label_name
            target_dir.mkdir(parents=True, exist_ok=True)
            item["image"].save(target_dir / f"{idx}.jpg")
            class_counts[label_name] += 1

    print("Export complete.")
