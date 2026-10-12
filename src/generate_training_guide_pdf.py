"""
PDF Generation Engine: Model Training Architecture & Metric Derivation Guide
Generates a comprehensive, professional PDF report:
  docs/Model_Training_Architecture_And_Accuracy_Guide.pdf
Includes:
  - Theoretical & Practical Training Explanations
  - Data Pipeline & Batching Mechanics
  - Loss Functions & Backpropagation Math
  - Step-by-Step Accuracy / Precision / Recall / mAP Calculation
  - Embedded Matplotlib Block Diagrams
  - Production Video Generation Prompt (Runway / Sora / Pika)
  - Project Memory & Architectural State Index
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True, parents=True)
PDF_PATH = DOCS_DIR / "Model_Training_Architecture_And_Accuracy_Guide.pdf"
DIAGRAMS_DIR = DOCS_DIR / "diagrams"
DIAGRAMS_DIR.mkdir(exist_ok=True, parents=True)

# ── 1. GENERATE HIGH-RES MATPLOTLIB BLOCK DIAGRAMS ─────────────────────────

def generate_training_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)

    boxes = [
        {"x": 2, "y": 14, "w": 14, "h": 16, "title": "Raw Images\n& Labels", "sub": "10k+ Images\nYOLO .txt bbox", "color": "#E3F2FD", "edge": "#1976D2"},
        {"x": 19, "y": 14, "w": 16, "h": 16, "title": "Data Pipeline\n& Augment", "sub": "Mosaic 4-mix\nHSV Jitter\n768x768 pad", "color": "#E8F5E9", "edge": "#388E3C"},
        {"x": 38, "y": 14, "w": 16, "h": 16, "title": "YOLO11s\nBackbone", "sub": "CSPDarknet\nSPPF Pool\nP3, P4, P5", "color": "#FFF3E0", "edge": "#F57C00"},
        {"x": 57, "y": 14, "w": 16, "h": 16, "title": "PANet Neck &\nDetect Head", "sub": "Feature Pyramid\nBox Regress [x,y,w,h]\nClass Logits", "color": "#F3E5F5", "edge": "#7B1FA2"},
        {"x": 76, "y": 24, "w": 22, "h": 14, "title": "Loss Calculation", "sub": "CIoU + DFL + Focal Loss\nPred vs Ground-Truth", "color": "#FFEBEE", "edge": "#D32F2F"},
        {"x": 76, "y": 4, "w": 22, "h": 14, "title": "Metric Validation", "sub": "IoU Matching (>=0.50)\nPrecision, Recall, mAP", "color": "#E0F7FA", "edge": "#0097A7"},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"],
                                      boxstyle="round,pad=0.8,rounding_size=1.5",
                                      facecolor=b["color"], edgecolor=b["edge"], linewidth=1.8)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.62, b["title"],
                ha="center", va="center", fontsize=9, fontweight="bold", color="#212121")
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.25, b["sub"],
                ha="center", va="center", fontsize=7.2, color="#555555")

    # Arrows
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=1.8, color="#424242")
    ax.annotate("", xy=(19, 22), xytext=(16, 22), arrowprops=arrow_props)
    ax.annotate("", xy=(38, 22), xytext=(35, 22), arrowprops=arrow_props)
    ax.annotate("", xy=(57, 22), xytext=(54, 22), arrowprops=arrow_props)
    ax.annotate("", xy=(76, 31), xytext=(73, 22), arrowprops=arrow_props)
    ax.annotate("", xy=(76, 11), xytext=(73, 22), arrowprops=arrow_props)

    # Backpropagation Feedback Loop
    ax.annotate("", xy=(46, 14), xytext=(87, 24),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=1.8, color="#D32F2F",
                                connectionstyle="arc3,rad=-0.35", linestyle="--"))
    ax.text(66, 7, "Backward Pass: AdamW Gradient Update", fontsize=7.5, color="#D32F2F", fontweight="bold")

    plt.tight_layout()
    img_path = DIAGRAMS_DIR / "training_pipeline_block_diagram.png"
    plt.savefig(str(img_path), dpi=300, bbox_inches="tight")
    plt.close()
    return img_path


def generate_accuracy_calculation_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)

    steps = [
        {"x": 2, "y": 10, "w": 16, "h": 20, "step": "Step 1: NMS", "title": "Suppress Duplicates", "desc": "Filter overlapping\nboxes with IoU > 0.45\nkeep highest score", "color": "#EDE7F6", "edge": "#512DA8"},
        {"x": 21, "y": 10, "w": 18, "h": 20, "step": "Step 2: IoU Test", "title": "Match Ground Truth", "desc": "IoU = Area(P ∩ G)\n      -----------------\n      Area(P ∪ G)\nIoU >= 0.50 threshold", "color": "#E1F5FE", "edge": "#0288D1"},
        {"x": 42, "y": 10, "w": 18, "h": 20, "step": "Step 3: Confusion", "title": "Classify TP/FP/FN", "desc": "TP: Matched IoU>=0.5\nFP: False alarm (IoU<0.5)\nFN: Missed actual hazard", "color": "#FFF8E1", "edge": "#FFA000"},
        {"x": 63, "y": 10, "w": 17, "h": 20, "step": "Step 4: P-R Curve", "title": "Sweep Confidence", "desc": "Precision = TP/(TP+FP)\nRecall    = TP/(TP+FN)\nPlot P vs R across conf", "color": "#E8F5E9", "edge": "#2E7D32"},
        {"x": 83, "y": 10, "w": 15, "h": 20, "step": "Step 5: Area / mAP", "title": "Compute AP", "desc": "AP = Area under P-R\nmAP@50 = Mean AP\nover all hazard classes", "color": "#FBE9E7", "edge": "#D84315"},
    ]

    for s in steps:
        rect = patches.FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"],
                                      boxstyle="round,pad=0.8,rounding_size=1.5",
                                      facecolor=s["color"], edgecolor=s["edge"], linewidth=1.8)
        ax.add_patch(rect)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.82, s["step"],
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1A237E")
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.62, s["title"],
                ha="center", va="center", fontsize=8.0, fontweight="bold", color="#333333")
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.28, s["desc"],
                ha="center", va="center", fontsize=7.0, color="#424242")

    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=1.8, color="#37474F")
    ax.annotate("", xy=(21, 20), xytext=(18, 20), arrowprops=arrow_props)
    ax.annotate("", xy=(42, 20), xytext=(39, 20), arrowprops=arrow_props)
    ax.annotate("", xy=(63, 20), xytext=(60, 20), arrowprops=arrow_props)
    ax.annotate("", xy=(83, 20), xytext=(80, 20), arrowprops=arrow_props)

    plt.tight_layout()
    img_path = DIAGRAMS_DIR / "accuracy_metric_calculation_flowchart.png"
    plt.savefig(str(img_path), dpi=300, bbox_inches="tight")
    plt.close()
    return img_path


# ── 2. NUMBERED CANVAS FOR PAGE NUMBERS ────────────────────────────────────

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#757575"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Edge-AI Cooperative Road Hazard Perception | Training & Evaluation Guide")
            self.setStrokeColor(colors.HexColor("#E0E0E0"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)

        # Footer
        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Confidential — Edge-AI Enabled Cooperative Road Hazard Perception Project")
        self.setStrokeColor(colors.HexColor("#E0E0E0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()


# ── 3. BUILD THE PDF STORY ─────────────────────────────────────────────────

def build_pdf():
    diag1 = generate_training_pipeline_diagram()
    diag2 = generate_accuracy_calculation_diagram()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#424242"),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1565C0"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#283593"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.2,
        textColor=colors.HexColor("#212121"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#1A237E"),
        backColor=colors.HexColor("#F5F7FA"),
        borderColor=colors.HexColor("#CFD8DC"),
        borderWidth=0.6,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=body_style,
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0D47A1"),
        backColor=colors.HexColor("#E3F2FD"),
        borderColor=colors.HexColor("#90CAF9"),
        borderWidth=0.8,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=8
    )

    story = []

    # Title Banner
    story.append(Paragraph("Edge-AI Road Hazard Detection & V2X Dissemination", title_style))
    story.append(Paragraph("<b>Comprehensive Engineering Guide:</b> Deep Learning Training Mechanics, Image Ingestion, Mathematical Metric Derivations, Video Prompts & System Memory", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1565C0"), spaceAfter=12))

    # SECTION 1
    story.append(Paragraph("1. System Overview & The Three Dedicated Hazard Detectors", h1_style))
    story.append(Paragraph(
        "This project develops an autonomous cooperative edge hazard perception system for connected vehicles and smart Roadside Units (RSUs). "
        "Rather than deploying a single generic detection model, we trained three dedicated <b>Ultralytics YOLO11s</b> models (9.4M parameters each) "
        "trained on domain-specific datasets (10,000+ images per class) and dynamically quantized to INT8 for real-time edge execution on a <b>Raspberry Pi 5</b>:",
        body_style
    ))

    model_summary_data = [
        ["Model Domain", "Target Classes", "Training Images", "Optimal Input", "mAP@50", "Precision", "INT8 Size"],
        ["Pothole Detector", "Pothole, Surface Crack", "10,000+", "768x768", "95.64%", "93.11%", "9.48 MB"],
        ["Construction Detector", "Barricades, Cones, Drums", "10,000+", "768x768", "92.70%", "95.20%", "9.48 MB"],
        ["Accident & Obstacle", "Collision, Overturned, Debris", "10,000+", "768x768", "92.47%", "96.50%", "9.48 MB"]
    ]
    t1 = Table(model_summary_data, colWidths=[100, 95, 65, 55, 50, 50, 55])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFD")]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # SECTION 2: HOW IMAGES ARE INGESTED
    story.append(Paragraph("2. How the Script Trains the Model: Image Ingestion & Pipeline", h1_style))
    story.append(Paragraph(
        "When you execute <code>python train_yolo.py</code>, Ultralytics initiates a multi-stage data ingestion pipeline. "
        "Here is the exact step-by-step path every single road hazard image follows:",
        body_style
    ))

    story.append(Paragraph("<b>Step 2.1 — Disk Reading & Decoding:</b> OpenCV loads the JPEG/PNG image from disk as a raw matrix of BGR pixels. The bounding box labels are read from the corresponding <code>.txt</code> file containing normalized coordinates <code>[class_id, x_center, y_center, width, height]</code>.", bullet_style))
    story.append(Paragraph("<b>Step 2.2 — Mosaic 4-Image Blending:</b> Ultralytics combines 4 distinct road hazard images into a single synthetic canvas. This exposes the network to multiple hazards simultaneously, small-scale objects, and border truncations, drastically improving distant pothole detection.", bullet_style))
    story.append(Paragraph("<b>Step 2.3 — Letterbox Resizing to 768×768:</b> Road cameras capture non-square images (e.g., 1920×1080). To preserve aspect ratio without distortion, the image is scaled so its longest side is 768 pixels, and neutral grey borders (RGB 114, 114, 114) pad the remaining space.", bullet_style))
    story.append(Paragraph("<b>Step 2.4 — Normalization & Tensor Batching:</b> The array is converted to float32, divided by 255.0 to map pixel values to <code>[0.0, 1.0]</code>, transposed from <code>(H, W, C)</code> to PyTorch format <code>(C, H, W)</code>, and stacked into a GPU mini-batch tensor of shape <code>(B, 3, 768, 768)</code>.", bullet_style))

    # EMBED DIAGRAM 1
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Figure 1:</b> End-to-End Deep Learning Training Pipeline & Backward Feedback Loop", h2_style))
    story.append(Image(str(diag1), width=490, height=205))
    story.append(Spacer(1, 10))

    # SECTION 3: FORWARD & BACKWARD PASS
    story.append(Paragraph("3. Forward Pass, Loss Computation & Backpropagation", h1_style))
    story.append(Paragraph(
        "Inside the GPU (Tesla T4 / A100), the tensor passes through the neural network layers:",
        body_style
    ))
    story.append(Paragraph("<b>1. Backbone (CSPDarknet with C3k2 & SPPF):</b> Progressively downsamples the image across 3 pyramidal feature maps: "
                           "<code>P3</code> (stride 8: 96×96 cells for tiny potholes), <code>P4</code> (stride 16: 48×48 cells for road debris), and <code>P5</code> (stride 32: 24×24 cells for massive construction barricades).", bullet_style))
    story.append(Paragraph("<b>2. PANet Neck:</b> Merges high-level semantic features with low-level spatial geometry using bidirectional top-down and bottom-up cross-connections.", bullet_style))
    story.append(Paragraph("<b>3. Anchor-Free Detection Head:</b> At every grid cell, the head outputs prediction vectors containing: "
                           "4 bounding box offsets <code>[cx, cy, w, h]</code> and class probability logits.", bullet_style))

    story.append(Paragraph("Mathematical Loss Functions Combined:", h2_style))
    story.append(Paragraph(
        "Python calculates a compound loss function $\\mathcal{L}_{\\text{total}} = \\lambda_{\\text{box}} \\mathcal{L}_{\\text{CIoU}} + \\lambda_{\\text{cls}} \\mathcal{L}_{\\text{Focal}} + \\lambda_{\\text{dfl}} \\mathcal{L}_{\\text{DFL}}$:",
        body_style
    ))
    story.append(Paragraph("• <b>Complete IoU Loss (CIoU):</b> Penalizes bounding box discrepancy, center distance, and aspect ratio divergence:<br/>"
                           "&nbsp;&nbsp;&nbsp;&nbsp;$\\mathcal{L}_{\\text{CIoU}} = 1 - \\text{IoU} + \\frac{\\rho^2(b, b^{\\text{gt}})}{c^2} + \\alpha v$<br/>"
                           "where $\\rho(b, b^{\\text{gt}})$ is the Euclidean distance between box centroids and $c$ is the diagonal length of the smallest enclosing box.", code_style))
    story.append(Paragraph("• <b>Task-Aligned Focal Loss:</b> Evaluates classification while heavily weighting hard examples and penalizing predictions that do not align with high IoU.", code_style))
    story.append(Paragraph("• <b>Backward Pass (AdamW Optimizer):</b> PyTorch autograd calculates $\\nabla_\\theta \\mathcal{L}$, and the optimizer adjusts model weights via cosine annealing schedule: $\\theta_{t} = \\theta_{t-1} - \\eta_t \\nabla \\mathcal{L}$.", code_style))

    story.append(PageBreak())

    # SECTION 4: HOW PYTHON CALCULATES ACCURACY
    story.append(Paragraph("4. How Python Calculates Accuracy by Itself: The Exact Math", h1_style))
    story.append(Paragraph(
        "You asked: <i>'How does Python tell you the number by itself: the accuracy and everything after model training?'</i> "
        "Here is the exact mathematical and algorithmic sequence Python runs during <code>model.val()</code>:",
        body_style
    ))

    # EMBED DIAGRAM 2
    story.append(Paragraph("<b>Figure 2:</b> Algorithmic Flowchart of Validation & mAP Calculation in Python", h2_style))
    story.append(Image(str(diag2), width=490, height=195))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step 4.1 — Non-Maximum Suppression (NMS):", h2_style))
    story.append(Paragraph(
        "A single pothole can trigger 10 adjacent grid cells. Python removes redundant overlapping boxes: "
        "it sorts boxes by confidence, takes the highest, and deletes any overlapping box with $\\text{IoU} > 0.45$.",
        body_style
    ))

    story.append(Paragraph("Step 4.2 — Intersection over Union (IoU) Matching:", h2_style))
    story.append(Paragraph(
        "For every remaining predicted box ($P$) and ground-truth annotation ($G$), Python calculates the overlap ratio:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>IoU = Area(P ∩ G) / Area(P ∪ G)</b><br/>"
        "If $\\text{IoU} \\ge 0.50$ AND the predicted class matches, Python increments <b>True Positive (TP)</b>.<br/>"
        "If $\\text{IoU} < 0.50$ or there is no matching ground truth, it is a <b>False Positive (FP)</b>.<br/>"
        "If an actual hazard in the image has no predicted box, Python marks it a <b>False Negative (FN)</b>.",
        callout_style
    ))

    story.append(Paragraph("Step 4.3 — Precision & Recall Formulas:", h2_style))
    story.append(Paragraph(
        "Python then computes the two fundamental detection metrics:<br/>"
        "• <b>Precision = TP / (TP + FP)</b> &nbsp;&nbsp;<i>(Of all alerts the car sounded, how many were actual hazards?)</i><br/>"
        "• <b>Recall = TP / (TP + FN)</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>(Of all actual road hazards out there, how many did the car detect?)</i>",
        code_style
    ))

    story.append(Paragraph("Step 4.4 — Average Precision (AP) & mAP@50:", h2_style))
    story.append(Paragraph(
        "A model's Precision changes as you lower the confidence threshold. Python sweeps confidence from 0.0 to 1.0, "
        "plots the <b>Precision-Recall Curve</b>, and computes the Area Under the Curve (AUC) using 101-point numerical integration:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$\\text{AP} = \\frac{1}{101} \\sum_{r \\in \\{0.0, 0.01, ..., 1.0\\}} p_{\\text{interp}}(r)$<br/>"
        "<b>mAP@50</b> is the mean AP across all classes at IoU threshold 0.50. "
        "For potholes, our Python validation script returned <b>95.64%</b>; for construction, <b>92.70%</b>; and for accidents, <b>92.47%</b>.",
        body_style
    ))

    story.append(Spacer(1, 8))

    # SECTION 5: VIDEO GENERATION PROMPT
    story.append(Paragraph("5. AI Video Generation Prompt for Project Showcase", h1_style))
    story.append(Paragraph(
        "To create a video explaining this project (for presentation, YouTube, or funding pitch using tools like Sora, Runway Gen-3, or Pika), "
        "use the following prompt:",
        body_style
    ))

    video_prompt_text = (
        "<b>Production Video Prompt (Runway Gen-3 / Sora / Pika):</b><br/><br/>"
        "<i>\"Cinematic 4K drone shot tracking a modern connected electric vehicle cruising along a 1.0-kilometer arterial road corridor at 45 km/h on an overcast afternoon. "
        "Split-screen visualization showing a sleek in-cabin HUD display powered by an onboard Raspberry Pi 5. "
        "As the car approaches 650 meters, a high-resolution forward dashcam activates with animated glowing amber bounding boxes detecting a severe road surface pothole [Model: YOLO11s INT8, Latency: 34.8ms, Precision: 95.6%]. "
        "A holographic circular telemetry pulse radiates from the car roof upward to a roadside unit (RSU) tower on the sidewalk. "
        "The RSU immediately flashes a cyan beacon radiating wireless ETSI DENM V2X hazard packets to oncoming traffic. "
        "A second approaching white sedan 50 meters behind receives the early warning alert on its dashboard screen, smoothly merges into Lane 1 without hard braking, and safely clears the hazard. "
        "Hyper-realistic lighting, depth of field, futuristic engineering HUD overlays, photorealistic 8k render, Unreal Engine 5 aesthetic.\"</i>"
    )
    story.append(Paragraph(video_prompt_text, callout_style))

    # SECTION 6: PERSISTENT MEMORY INDEX
    story.append(Paragraph("6. Project Memory & Architectural State Index", h1_style))
    story.append(Paragraph(
        "All critical parameters from this project have been recorded into the persistent AI Memory graph and this document "
        "so any future session has instant recall:",
        body_style
    ))

    memory_table_data = [
        ["Subsystem / Asset", "Specification / Path", "Key Metric / Operational Value"],
        ["Pothole Model", "quantization/pothole_yolo11s_int8.onnx", "9.48 MB | mAP: 95.64% | Precision: 93.11% | Latency: 34.8ms"],
        ["Construction Model", "quantization/construction_yolo11s_int8.onnx", "9.48 MB | mAP: 92.70% | Precision: 95.20% | Latency: 36.1ms"],
        ["Accident Model", "quantization/accident_obstacle_yolo11s_int8.onnx", "9.48 MB | mAP: 92.47% | Precision: 96.50% | Latency: 38.2ms"],
        ["Edge Target", "Raspberry Pi 5 (Quad Cortex-A76 @ 2.4GHz)", "RAM: 107.88 MB total for all 3 models (<1.4% of 8GB RAM)"],
        ["SUMO 1.0 km Corridor", "simulation/corridor_1km/corridor.sumocfg", "2 junctions (TL_01, TL_02), 3 hazard zones, RSU at x=500m"],
        ["SUMO 2.5 km Blueprint", "sumo_advanced_craftable_plan.md", "4 junctions, dynamic incident spawning, 3 RSU GeoNetworking"],
        ["GitHub Repository", "origin/main", "Commits 1679fad & 191bf40 pushed with all simulations & benchmarks"]
    ]
    t2 = Table(memory_table_data, colWidths=[105, 185, 180])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
    ]))
    story.append(t2)

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Status Note:</b> Physical testing on the Raspberry Pi 5 is currently paused to allow thermal dissipation. "
        "All weight binaries, evaluation scripts, and simulation environments are ready to resume whenever hardware is cooled.",
        ParagraphStyle("Notice", parent=body_style, fontName="Helvetica-Oblique", textColor=colors.HexColor("#C62828"))
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✓ PDF successfully generated at: {PDF_PATH}")
    return PDF_PATH

if __name__ == "__main__":
    build_pdf()
