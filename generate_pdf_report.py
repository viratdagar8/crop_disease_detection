"""Script to generate professional academic PDF documents:
1. Crop_Disease_Detection_Project_Report.pdf (Comprehensive Project & Viva Guide)
2. Project_Synopsis.pdf (Official Project Synopsis)
Uses ReportLab library.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
except ImportError:
    print("ReportLab library installing or missing. Please wait for pip completion...")

def build_pdf_report():
    pdf_filename = str(PROJECT_ROOT / "Crop_Disease_Detection_Project_Report.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1b5e20")   # Dark Green
    SECONDARY = colors.HexColor("#2e7d32") # Medium Green
    TEXT_DARK = colors.HexColor("#212121") # Dark Grey
    BG_LIGHT = colors.HexColor("#f1f8e9")  # Soft Green tint
    ACCENT = colors.HexColor("#e65100")    # Deep Orange accent

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        alignment=1, # Center
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=TEXT_DARK,
        alignment=1,
        spaceAfter=20
    )

    heading1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    heading2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    qa_question = ParagraphStyle(
        'QAQuestion',
        parent=styles['Heading4'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=ACCENT,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    qa_answer = ParagraphStyle(
        'QAAnswer',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        spaceAfter=8
    )

    story = []

    # Title Block
    story.append(Paragraph("AgriScan AI: Crop Disease Detection Using CNN", title_style))
    story.append(Paragraph("<b>Comprehensive Academic Project Report & Defense Viva Guide</b><br/>"
                           "Author: <b>Virat Dagar</b> | ABES Engineering College (CSE Data Science)<br/>"
                           "Program: HCLTech Industry Aligned Projects (Academic Session 2026–2027)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Product Vision", heading1_style))
    story.append(Paragraph(
        "Crop diseases pose severe threats to global agricultural yield and food security. Early, accurate identification "
        "enables targeted intervention, minimizing chemical usage and crop loss. <b>AgriScan AI</b> is a multi-tier computer vision "
        "decision-support system built using Deep Learning and Convolutional Neural Networks (CNN). The system incorporates "
        "<b>Approach 1: Binary Leaf Classifier (Gate Model)</b> as a primary Stage 1 security gatekeeper, followed by a 38-class "
        "Stage 2 fine-grained crop disease classifier. End-to-end web deployment is handled via a lightweight Flask web interface.",
        body_style
    ))

    # Project Information Table
    info_data = [
        [Paragraph("<b>Parameter</b>", heading2_style), Paragraph("<b>Details</b>", heading2_style)],
        [Paragraph("<b>Project Title</b>", body_style), Paragraph("Crop Disease Detection Using Convolutional Neural Networks (CNN)", body_style)],
        [Paragraph("<b>Student Author</b>", body_style), Paragraph("Virat Dagar", body_style)],
        [Paragraph("<b>Department & College</b>", body_style), Paragraph("Computer Science & Engineering (Data Science), ABES Engineering College", body_style)],
        [Paragraph("<b>Industry Alignment</b>", body_style), Paragraph("HCLTech Industry Aligned Projects Program", body_style)],
        [Paragraph("<b>Stage 1 Architecture</b>", body_style), Paragraph("Approach 1: Binary Leaf Classifier (Gate Model - Is Leaf vs Non-Leaf)", body_style)],
        [Paragraph("<b>Stage 2 Architecture</b>", body_style), Paragraph("Baseline 4-Block Conv2D Neural Network (38 Target Classes)", body_style)],
        [Paragraph("<b>Dataset Scale</b>", body_style), Paragraph("PlantVillage Dataset (54,306 images across 14 crops & 38 classes)", body_style)],
        [Paragraph("<b>Backend Web Framework</b>", body_style), Paragraph("Flask 3.0 (Python 3.12)", body_style)]
    ]
    info_table = Table(info_data, colWidths=[160, 340])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c8e6c9")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))

    # Architecture Section
    story.append(Paragraph("2. System Architecture & Methodology", heading1_style))
    story.append(Paragraph(
        "The system operates on a <b>Hierarchical Multi-Stage Pipeline</b> to prevent invalid non-leaf images from producing misleading disease predictions:<br/>"
        "• <b>Stage 1 (Gate Model - Approach 1):</b> The uploaded image passes through a dedicated Binary CNN Gate Model (3 Conv2D blocks + Sigmoid). If the model determines the image is not a valid plant leaf (Probability < 50%), the pipeline rejects the input immediately with actionable feedback.<br/>"
        "• <b>Stage 2 (Disease Classifier):</b> Valid leaf images are resized to 128x128x3 and normalized to [0, 1]. A 4-block Conv2D network computes Softmax probabilities across 38 crop disease classes.<br/>"
        "• <b>Stage 3 (Agronomic Advisory):</b> The predicted label maps to <code>disease_info.json</code>, providing symptoms, cause, and organic/chemical management guidelines.",
        body_style
    ))

    # CNN Math Table
    story.append(Paragraph("3. Convolutional Neural Network Layer Specifications", heading1_style))
    cnn_data = [
        [Paragraph("<b>Layer Type</b>", heading2_style), Paragraph("<b>Kernel / Units</b>", heading2_style), Paragraph("<b>Activation</b>", heading2_style), Paragraph("<b>Purpose</b>", heading2_style)],
        [Paragraph("Input", body_style), Paragraph("(128, 128, 3)", body_style), Paragraph("None", body_style), Paragraph("Raw RGB image tensor normalized to [0,1]", body_style)],
        [Paragraph("Conv2D Block 1", body_style), Paragraph("32 filters (3x3)", body_style), Paragraph("ReLU", body_style), Paragraph("Extracts low-level edges, colors, and line features", body_style)],
        [Paragraph("Conv2D Block 2", body_style), Paragraph("64 filters (3x3)", body_style), Paragraph("ReLU", body_style), Paragraph("Extracts textures, leaf veins, and spot boundaries", body_style)],
        [Paragraph("Conv2D Block 3", body_style), Paragraph("128 filters (3x3)", body_style), Paragraph("ReLU", body_style), Paragraph("Identifies pathological lesions, rusts, and fungal specks", body_style)],
        [Paragraph("Conv2D Block 4", body_style), Paragraph("256 filters (3x3)", body_style), Paragraph("ReLU", body_style), Paragraph("Encodes high-level spatial disease representations", body_style)],
        [Paragraph("Dense Layer", body_style), Paragraph("256 units", body_style), Paragraph("ReLU", body_style), Paragraph("Fully-connected decision feature synthesis", body_style)],
        [Paragraph("Output Layer", body_style), Paragraph("38 units", body_style), Paragraph("Softmax", body_style), Paragraph("Computes categorical probability distribution across classes", body_style)]
    ]
    cnn_table = Table(cnn_data, colWidths=[90, 80, 70, 260])
    cnn_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c8e6c9")),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(cnn_table)
    story.append(Spacer(1, 10))

    # Page Break for Viva Q&A Guide
    story.append(PageBreak())
    story.append(Paragraph("4. Teacher Viva Voce Defense & Examination Guide", heading1_style))
    story.append(Paragraph("Study these core technical questions and answers for your project evaluation:", body_style))
    story.append(Spacer(1, 5))

    viva_qa = [
        ("Q1: What is Approach 1 (Binary Leaf Classifier / Gate Model) and why is it essential?",
         "Answer: Approach 1 acts as a Stage 1 security gatekeeper in the hierarchical architecture. It uses a binary CNN to answer 'Is this image a plant leaf?' (Class 1 = Leaf, Class 0 = Non-Leaf). Without this gate model, uploading a picture of a car or shoe would force a standard 38-class classifier to output an arbitrary crop disease prediction with misleading confidence."),

        ("Q2: Why use CNNs instead of traditional ML algorithms like SVM or Random Forest?",
         "Answer: Traditional ML requires manual, handcrafted feature extraction (e.g., SIFT, color histograms) which cannot capture complex leaf disease textures across 38 classes. CNNs automatically learn hierarchical visual features directly from pixel matrices (Edges -> Textures -> Lesions -> Disease Patterns) with spatial and translation invariance."),

        ("Q3: What loss function did you use for multi-class classification and why?",
         "Answer: SparseCategoricalCrossentropy. Unlike standard CategoricalCrossentropy which requires memory-intensive one-hot encoded vectors for 38 classes across 54,000 images, SparseCategoricalCrossentropy accepts integer labels directly, achieving identical mathematical loss computation with significantly lower memory overhead."),

        ("Q4: How do Batch Normalization and Dropout prevent overfitting?",
         "Answer: Batch Normalization normalizes layer outputs (mean=0, variance=1) across mini-batches, accelerating convergence and stabilizing training. Dropout randomly deactivates a fraction of neurons (e.g., 25% to 50%) during forward passes, preventing co-adaptation of features and forcing the network to learn redundant representations."),

        ("Q5: What metrics were used to evaluate model performance beyond overall accuracy?",
         "Answer: We evaluated Precision (TP / (TP + FP)) to minimize false positives (avoid misdiagnosing healthy crops), Recall (TP / (TP + FN)) to minimize missed disease infections, F1-Score (harmonic mean balancing precision and recall), and a 38x38 Confusion Matrix to identify inter-class confusion."),

        ("Q6: What are the limitations of this model in real-world agricultural settings?",
         "Answer: PlantVillage images were captured under uniform laboratory lighting. Real-world field leaves contain complex soil/foliage backgrounds and varying sunlight. To address this, Approach 1 (Binary Gate) filters bad background inputs, while data augmentation improves generalization."),

        ("Q7: How is the web application deployed and structured?",
         "Answer: The application uses Flask backend (app.py) serving an HTML5/CSS3/JS user interface. Images uploaded via REST API (/predict) are validated, passed to the Stage 1 Gate Model, processed by the Stage 2 CNN, and returned with JSON diagnostic results and management advice.")
    ]

    for q, a in viva_qa:
        story.append(Paragraph(q, qa_question))
        story.append(Paragraph(a, qa_answer))

    story.append(Spacer(1, 10))
    story.append(Paragraph("5. Conclusion & Academic Summary", heading1_style))
    story.append(Paragraph(
        "AgriScan AI demonstrates an end-to-end industrial deep learning application. By combining <b>Approach 1: Binary Leaf Classifier (Gate Model)</b> with a 38-class baseline CNN and a Flask decision-support web interface, the project achieves robust, reliable agricultural disease identification.",
        body_style
    ))

    doc.build(story)
    print(f"[OK] Generated Comprehensive Project Report PDF: {pdf_filename}")


def build_pdf_synopsis():
    pdf_filename = str(PROJECT_ROOT / "Project_Synopsis.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    PRIMARY = colors.HexColor("#1b5e20")
    TEXT_DARK = colors.HexColor("#212121")
    BG_LIGHT = colors.HexColor("#f1f8e9")

    title_style = ParagraphStyle('SynTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=PRIMARY, alignment=1, spaceAfter=10)
    sub_style = ParagraphStyle('SynSub', parent=styles['Normal'], fontName='Helvetica', fontSize=11, leading=15, textColor=TEXT_DARK, alignment=1, spaceAfter=15)
    h1_style = ParagraphStyle('SynH1', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=PRIMARY, spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('SynBody', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.5, leading=13.5, textColor=TEXT_DARK, spaceAfter=6)

    story = []
    story.append(Paragraph("PROJECT SYNOPSIS", title_style))
    story.append(Paragraph("<b>Crop Disease Detection Using Convolutional Neural Networks (CNN)</b><br/>"
                           "Student: <b>Virat Dagar</b> | Roll No / Dept: CSE (Data Science)<br/>"
                           "Institution: ABES Engineering College | Program: HCLTech Industry Aligned Projects", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=12))

    story.append(Paragraph("1. Title of the Project", h1_style))
    story.append(Paragraph("Crop Disease Detection Using Convolutional Neural Networks (CNN) with Hierarchical Binary Gate Verification.", body_style))

    story.append(Paragraph("2. Objective & Problem Statement", h1_style))
    story.append(Paragraph("Agricultural crop diseases cause up to 40% annual global crop yield loss. Traditional manual scouting by agricultural experts is slow, expensive, and unavailable to remote farmers. The objective of this project is to construct an end-to-end automated computer vision system that instantly identifies crop species, classifies 38 plant diseases/healthy states, and provides agronomic treatment advisory.", body_style))

    story.append(Paragraph("3. System Architecture & Methodology", h1_style))
    story.append(Paragraph("The system implements a <b>Two-Stage Hierarchical Deep Learning Pipeline</b>:<br/>"
                           "• <b>Stage 1 (Approach 1: Binary Leaf Gate Model):</b> Filters non-leaf or out-of-focus background uploads using a binary CNN (Is Leaf vs Non-Leaf).<br/>"
                           "• <b>Stage 2 (38-Class Multiclass CNN):</b> Evaluates valid plant leaves using a 4-block Conv2D network (Conv2D -> BatchNorm -> MaxPool -> Dropout -> Softmax).<br/>"
                           "• <b>Stage 3 (Web Portal & Advisory Engine):</b> Delivers predictions via a responsive Flask web interface with treatment recommendations.", body_style))

    story.append(Paragraph("4. Key Technical Specifications", h1_style))
    syn_data = [
        [Paragraph("<b>Component</b>", h1_style), Paragraph("<b>Specification</b>", h1_style)],
        [Paragraph("Dataset", body_style), Paragraph("PlantVillage Dataset (54,306 RGB images, 38 classes, 14 crop species)", body_style)],
        [Paragraph("Programming Language", body_style), Paragraph("Python 3.12", body_style)],
        [Paragraph("Deep Learning Framework", body_style), Paragraph("TensorFlow 2.18 / Keras", body_style)],
        [Paragraph("Stage 1 Classifier", body_style), Paragraph("Approach 1 Binary Gate CNN (Sigmoid Activation)", body_style)],
        [Paragraph("Stage 2 Classifier", body_style), Paragraph("Baseline 4-Block Conv2D CNN (Softmax Activation)", body_style)],
        [Paragraph("Web Application", body_style), Paragraph("Flask 3.0 Backend with HTML5/CSS3/JavaScript Frontend", body_style)]
    ]
    t = Table(syn_data, colWidths=[150, 330])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c8e6c9")),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Project Deliverables", h1_style))
    story.append(Paragraph("1. Modular Python ML Source Code (<code>src/</code>)<br/>"
                           "2. Trained Model Weights (<code>model/binary_gate_model.h5</code> & <code>model/crop_disease_cnn_best.keras</code>)<br/>"
                           "3. Flask Web Application (<code>app.py</code>)<br/>"
                           "4. Academic Viva & Examination Guide (<code>viva_prep.md</code>)<br/>"
                           "5. Comprehensive PDF Project Report & Official Synopsis Document", body_style))

    doc.build(story)
    print(f"[OK] Generated Official Project Synopsis PDF: {pdf_filename}")


if __name__ == "__main__":
    print("Building PDF Academic Deliverables...")
    try:
        build_pdf_report()
        build_pdf_synopsis()
        print("🎉 PDF documents generated successfully!")
    except Exception as e:
        print(f"Error generating PDF: {e}")
