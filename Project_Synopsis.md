# Academic Project Synopsis
## Crop Disease Detection Using Convolutional Neural Networks (CNN)

- **Project Title:** Crop Disease Detection Using Convolutional Neural Networks (CNN) with Hierarchical Binary Gate Verification
- **Student Name:** Virat Dagar
- **Department:** Computer Science & Engineering (Data Science)
- **Institution:** ABES Engineering College, Ghaziabad
- **Program:** HCLTech Industry Aligned Projects Program
- **Academic Session:** 2026–2027

---

## 1. Executive Summary & Problem Statement
Crop diseases are a major threat to agricultural productivity, causing severe economic damage and global food insecurity. Traditional manual scouting relies on specialized agricultural expertise, which is scarce, slow, and expensive for smallholder farmers. 

This project delivers **AgriScan AI**, an end-to-end computer vision decision-support system powered by Deep Learning. The system allows farmers and agricultural scouts to upload a leaf photograph and receive an instant, accurate disease diagnosis, confidence score, and actionable agronomic treatment guidance.

---

## 2. Methodology & Multi-Stage Architecture

The project features a **Two-Stage Hierarchical Deep Learning Architecture**:

```
[ User Leaf Image ]
        │
        ▼
┌─────────────────────────────────────────┐
│ Stage 1: Approach 1 - Binary Gate Model │ ──► (Probability < 0.5) ──► [ REJECT: Non-Leaf Image ]
└─────────────────────────────────────────┘
        │ (Probability >= 0.5)
        ▼
┌─────────────────────────────────────────┐
│ Stage 2: 38-Class Multiclass CNN        │ ──► [ Softmax Probabilities across 38 Classes ]
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Stage 3: Agronomic Advisory Engine      │ ──► [ Symptoms, Cause, Prevention & Management ]
└─────────────────────────────────────────┘
```

1. **Stage 1 (Approach 1: Binary Leaf Classifier Gate Model):**  
   A dedicated binary CNN (Conv2D $\to$ BatchNorm $\to$ MaxPool $\to$ Sigmoid) acts as a primary security gate. It evaluates whether the uploaded image is a valid plant leaf ($P(\text{Leaf}) \ge 0.5$) vs non-leaf / background artifact. Invalid uploads are rejected before disease classification.
2. **Stage 2 (38-Class Multiclass CNN Classifier):**  
   Valid leaves are passed to a 4-block baseline CNN. Inputs are resized to $128 \times 128 \times 3$ and normalized to $[0, 1]$. The network outputs Softmax probabilities across 38 target classes (14 crop species).
3. **Stage 3 (Agronomic Advisory Engine):**  
   The predicted disease maps to `disease_info.json`, returning symptoms, cause, organic control, and chemical treatment guidelines.

---

## 3. Technical Specifications

| Parameter | Value / Technical Detail |
|---|---|
| **Dataset** | PlantVillage Dataset (54,306 RGB photographs across 14 crops & 38 classes) |
| **Language & Version** | Python 3.12 |
| **Deep Learning Library** | TensorFlow 2.18 / Keras |
| **Backend Web Framework** | Flask 3.0 |
| **Stage 1 Model Loss** | Binary Crossentropy (`binary_crossentropy`) |
| **Stage 2 Model Loss** | Sparse Categorical Crossentropy (`sparse_categorical_crossentropy`) |
| **Optimizer** | Adam Optimizer ($\eta = 0.001$) with Learning Rate Reduction on Plateau |
| **Evaluation Metrics** | Categorical Accuracy, Weighted Precision, Recall, F1-Score, Confusion Matrix |

---

## 4. Completed Project Deliverables (Exact Scope Completed)

1. **Modular Python Source Code (`src/`):**
   - [`src/config.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/config.py): Hyperparameters and system paths.
   - [`src/model.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/model.py): Baseline CNN, MobileNetV2, and Approach 1 Binary Gate architectures.
   - [`src/gate_model.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/gate_model.py): Stage 1 Binary Gate classification module.
   - [`src/dataset.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/dataset.py): Hugging Face `mohanty/PlantVillage` dataset pipeline & augmentations.
   - [`src/train.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/train.py) & [`src/train_full_dataset.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/train_full_dataset.py): Model training loops & checkpointing.
   - [`src/evaluate.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/evaluate.py): Multi-class metrics & confusion matrix generator.
   - [`src/predict.py`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/src/predict.py): Single-image inference engine.
2. **Web Portal (`app.py`):** Flask application with HTML5/CSS3/JS UI ([`templates/index.html`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/templates/index.html)).
3. **Academic Defense Guide ([`viva_prep.md`](file:///C:/Users/PC/.gemini/antigravity/scratch/crop_disease_detection/viva_prep.md)):** Oral viva Q&As and mathematical justifications.
4. **PDF Documentation:** `Crop_Disease_Detection_Project_Report.pdf` & `Project_Synopsis.pdf`.
