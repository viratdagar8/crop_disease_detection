# Crop Disease Detection Using Convolutional Neural Networks (CNN)

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![HCLTech Industry Project](https://img.shields.io/badge/HCLTech-Industry%20Aligned%20Project-brightgreen.svg)
![ABES Engineering College](https://img.shields.io/badge/ABES%20EC-CSE%20(Data%20Science)-red.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📌 Project Information
- **Project Title:** Crop Disease Detection Using Convolutional Neural Networks (CNN)
- **Prepared By:** Virat Dagar
- **Roll Number / Student:** Virat Dagar
- **Department:** Computer Science & Engineering (Data Science)
- **Institution:** ABES Engineering College, Ghaziabad
- **Program:** HCLTech Industry Aligned Projects Program
- **Academic Session:** 2026–2027
- **PRD Version:** 1.0 (Status: Active Development)

---

## 📖 Executive Summary & Product Vision
Crop diseases cause severe economic losses and compromise food security globally. Traditional visual inspection by farmers requires scarce agricultural expertise and is prone to delays. 

**AgriScan AI** is an end-to-end Computer Vision decision-support system powered by Deep Learning. It enables users to upload a leaf photograph and receive an instant, accurate disease classification with an empirical confidence score and actionable agronomic guidance.

> **Important Agricultural Disclaimer:**  
> This system is designed as an educational and decision-support tool to assist in early disease identification. It does not replace professional agronomist lab diagnostics or automatic pesticide prescription (aligned with PRD Section 3 & 8).

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    User["User / Demonstrator"] -->|"Uploads leaf image"| WebUI["Web Interface (HTML5/CSS3/JS)"]
    WebUI -->|"POST /predict"| FlaskAPI["Flask Backend API (app.py)"]
    FlaskAPI -->|"Format & Size Check"| Validator["Input Validation Module"]
    Validator -->|"128x128 Resizing & [0,1] Normalization"| Preprocessor["Preprocessing Pipeline"]
    Preprocessor -->|"Tensor Input"| CNN["CNN Classifier (Baseline / MobileNet)"]
    CNN -->|"Softmax Probabilities"| Predictor["Inference Module (predict.py)"]
    Predictor -->|"Top-1 & Top-3 Probabilities + Advisory"| FlaskAPI
    FlaskAPI -->|"JSON Response"| WebUI
    WebUI -->|"Displays Diagnosis, Confidence & Advisory"| User
```

---

## 🌿 Dataset Details
- **Dataset:** Official [PlantVillage Dataset](https://huggingface.co/datasets/mohanty/PlantVillage) hosted on Hugging Face (`mohanty/PlantVillage`).
- **Configuration:** `color` (Original RGB photographs).
- **Scale:** 54,306 high-resolution leaf images.
- **Coverage:** 14 crop species across 26 distinct diseases and healthy leaf controls (**38 total classes**).
- **Splits:**
  - **Train Split:** 43,444 images (80%)
  - **Test Split:** 10,862 images (20%)
  - **Validation Split:** 15% carved from the training split (~6,516 images) for real-time model checkpointing.

### Supported Crops
1. **Apple** (Apple Scab, Black Rot, Cedar Apple Rust, Healthy)
2. **Blueberry** (Healthy)
3. **Cherry** (Powdery Mildew, Healthy)
4. **Corn (Maize)** (Cercospora Leaf Spot/Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy)
5. **Grape** (Black Rot, Esca/Black Measles, Leaf Blight/Isariopsis Spot, Healthy)
6. **Orange** (Huanglongbing / Citrus Greening)
7. **Peach** (Bacterial Spot, Healthy)
8. **Bell Pepper** (Bacterial Spot, Healthy)
9. **Potato** (Early Blight, Late Blight, Healthy)
10. **Raspberry** (Healthy)
11. **Soybean** (Healthy)
12. **Squash** (Powdery Mildew)
13. **Strawberry** (Leaf Scorch, Healthy)
14. **Tomato** (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Mosaic Virus, Healthy)

---

## 🧠 Neural Network Architecture

In accordance with PRD Section 12, a clear, modular **Baseline Convolutional Neural Network (CNN)** is implemented for viva explanation and robust performance:

| Layer (Type) | Filters / Units | Kernel / Pool Size | Activation | Regularization | Output Shape |
|---|---|---|---|---|---|
| **Input** | - | - | - | - | `(None, 128, 128, 3)` |
| **Conv2D Block 1** | 32 | `(3, 3)` | ReLU | BatchNormalization, Dropout(0.2) | `(None, 64, 64, 32)` |
| **Conv2D Block 2** | 64 | `(3, 3)` | ReLU | BatchNormalization, Dropout(0.2) | `(None, 32, 32, 64)` |
| **Conv2D Block 3** | 128 | `(3, 3)` | ReLU | BatchNormalization, Dropout(0.25) | `(None, 16, 16, 128)` |
| **Conv2D Block 4** | 256 | `(3, 3)` | ReLU | BatchNormalization, Dropout(0.25) | `(None, 8, 8, 256)` |
| **Flatten** | - | - | - | - | `(None, 16384)` |
| **Dense** | 256 | - | ReLU | BatchNormalization, Dropout(0.5) | `(None, 256)` |
| **Output (Dense)** | 38 | - | Softmax | - | `(None, 38)` |

*(Optional Transfer Learning with MobileNetV2 is also supported via `--transfer`).*

---

## 📁 Repository Structure
Conforming to PRD Section 17:

```text
crop_disease_detection/
├── dataset/                    # Dataset metadata and acquisition utilities
│   └── README.md
├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│   └── exploration_and_demo.ipynb
├── src/                        # Core Python ML source code
│   ├── __init__.py
│   ├── config.py               # Paths, hyperparameters, image dimensions
│   ├── dataset.py              # HF dataset loading, preprocessing, tf.data pipeline
│   ├── model.py                # Baseline CNN & MobileNetV2 architectures
│   ├── train.py                # Training loop, callbacks, learning curves
│   ├── evaluate.py             # Evaluation metrics, confusion matrix, F1-scores
│   └── predict.py              # Single-image inference & advisory retrieval
├── model/                      # Saved models and class mapping
│   ├── class_indices.json      # Mapping from integer IDs to 38 class labels
│   ├── disease_info.json       # Agronomic knowledge base (symptoms & management)
│   └── evaluation/             # Training curves & confusion matrix plots
├── templates/                  # Frontend HTML templates
│   └── index.html              # Responsive upload & diagnosis interface
├── static/                     # Static web assets
│   ├── css/style.css           # Modern agricultural tech design
│   ├── js/main.js              # Drag-drop, async prediction, progress animation
│   └── uploads/                # Temporary uploaded images
├── app.py                      # Flask web application backend
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── viva_prep.md                # Academic viva & evaluation preparation guide
├── .gitignore
└── LICENSE                     # MIT License
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.12 (recommended) or 3.10 / 3.11
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/viratdagar/crop_disease_detection.git
cd crop_disease_detection

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Running the Web Application
```bash
python app.py
```
Open your browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

### 4. Training the CNN Model
To train the baseline CNN on the PlantVillage dataset:

```bash
# Full training (20 epochs)
python src/train.py --epochs 20 --batch-size 32

# Quick verification / viva demo run (small subset)
python src/train.py --quick-demo

# Optional: Train with MobileNetV2 Transfer Learning
python src/train.py --transfer --epochs 15
```

### 5. Evaluating the Model
To compute multi-class metrics (Accuracy, Precision, Recall, F1-Score) and generate the confusion matrix heatmap:

```bash
python src/evaluate.py
```
Outputs are saved to `model/evaluation/`:
- `confusion_matrix.png`
- `classification_report.json`
- `evaluation_summary.txt`

---

## 📊 Evaluation & Metrics (PRD FR-09)
The model is evaluated using standard industry metrics:
- **Categorical Accuracy:** Overall fraction of correctly identified leaf conditions.
- **Weighted Precision & Recall:** Accounting for class distribution across rare and common diseases.
- **F1-Score (Macro & Weighted):** Harmonic mean of precision and recall.
- **Confusion Matrix:** Identifies inter-class misclassification patterns (e.g., distinguishing Early Blight vs Late Blight).

---

## 🎓 Academic Document Control (PRD Section 25)
- **Institution:** ABES Engineering College
- **Program:** B.Tech CSE (Data Science)
- **Corporate Training:** HCLTech Industry Aligned Projects Program
- **Student Author:** Virat Dagar
- **Session:** 2026–2027
