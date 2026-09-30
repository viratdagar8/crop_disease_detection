"""Utility script to download a small set of representative sample images 
from PlantVillage for immediate testing of the web interface and prediction pipeline.
"""

from pathlib import Path
from PIL import Image, ImageDraw
from datasets import load_dataset
from src.config import PROJECT_ROOT

SAMPLES_DIR = PROJECT_ROOT / "static" / "sample_images"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def fetch_samples_from_hf(num_samples=5):
    """Fetches sample images across different disease classes from Hugging Face."""
    print("Downloading sample leaf images from Hugging Face PlantVillage...")
    try:
        ds = load_dataset("mohanty/PlantVillage", "color", split="test")
        labels = ds.features["label"].names

        saved_classes = set()
        count = 0

        for item in ds:
            lbl_name = labels[item["label"]]
            if lbl_name not in saved_classes:
                clean_name = lbl_name.replace("___", "_").replace(" ", "_")
                target_file = SAMPLES_DIR / f"{clean_name}.jpg"
                item["image"].save(target_file)
                saved_classes.add(lbl_name)
                count += 1
                print(f"Saved sample: {target_file.name}")

            if count >= num_samples:
                break

        print(f"Successfully saved {count} sample images to {SAMPLES_DIR}")
    except Exception as e:
        print(f"Error fetching from Hugging Face: {e}")
        create_synthetic_test_leaf()


def create_synthetic_test_leaf():
    """Generates synthetic test images if offline or running in isolated environment."""
    print("Generating synthetic test leaf samples for offline testing...")
    
    # 1. Tomato Early Blight synthetic leaf
    img = Image.new("RGB", (256, 256), color=(46, 125, 50))
    draw = ImageDraw.Draw(img)
    # Draw leaf veins
    draw.line([(128, 20), (128, 236)], fill=(120, 190, 80), width=4)
    draw.line([(128, 80), (60, 120)], fill=(100, 170, 70), width=2)
    draw.line([(128, 80), (196, 120)], fill=(100, 170, 70), width=2)
    draw.line([(128, 150), (40, 190)], fill=(100, 170, 70), width=2)
    draw.line([(128, 150), (216, 190)], fill=(100, 170, 70), width=2)
    # Draw fungal concentric spots
    draw.ellipse([(70, 100), (110, 140)], fill=(80, 50, 20), outline=(130, 90, 40), width=3)
    draw.ellipse([(80, 110), (100, 130)], fill=(40, 25, 10))
    draw.ellipse([(150, 140), (190, 180)], fill=(80, 50, 20), outline=(130, 90, 40), width=3)

    sample_path = SAMPLES_DIR / "sample_tomato_early_blight.jpg"
    img.save(sample_path)
    print(f"Saved synthetic sample to {sample_path}")


if __name__ == "__main__":
    fetch_samples_from_hf()
