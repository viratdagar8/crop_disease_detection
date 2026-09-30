# PlantVillage Dataset Information

This project uses the official **PlantVillage** dataset, accessed via the Hugging Face hub repository:
- **Repository ID**: `mohanty/PlantVillage`
- **Configuration**: `color` (Original RGB leaf photographs)
- **Source Paper**: *Mohanty et al. (2016), "Using Deep Learning for Image-Based Plant Disease Detection", Frontiers in Plant Science.*
- **Dataset Link**: [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)

---

## Dataset Statistics & Coverage
- **Total Images**: 54,306 images
- **Crop Species**: 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- **Disease Categories**: 26 plant diseases + healthy controls
- **Total Classes**: 38 distinct classification categories

---

## Dataset Splits
1. **Predefined Train Split**: 43,444 images (80%)
2. **Predefined Test Split**: 10,862 images (20%)
3. **Internal Validation Split**: 15% carved from the training split (~6,516 images) for hyperparameter tuning and model checkpointing.

The train/test splits preserve leaf grouping logic so images of the same leaf specimen do not appear across both train and test splits, preventing data leakage.

---

## How to Download / Stream Dataset
The project is built to stream or load the dataset on demand using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("mohanty/PlantVillage", "color")
```

To export the raw dataset into local directories (`dataset/train/`, `dataset/val/`, `dataset/test/`), run:

```bash
python -c "from src.dataset import export_hf_to_directory; export_hf_to_directory()"
```
