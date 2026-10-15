import os
import sys
import json
import time
import shutil
import random
import subprocess
from pathlib import Path

print("=== [Google Colab T4 GPU] Starting Unified Multi-Hazard YOLO11s Training Pipeline ===")
start_time = time.time()

# 1. Mount Kaggle credentials inside VM
os.makedirs("/root/.kaggle", exist_ok=True)
if os.path.exists("/content/kaggle.json"):
    shutil.copy2("/content/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("✓ Kaggle credentials successfully initialized.")
else:
    print("! Note: /content/kaggle.json not found, proceeding with local caches if available.")

# 2. Ensure dependencies are installed
print("--- Installing Ultralytics, ONNX, and Quantization Libraries ---")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime", "kaggle"], check=True)

import torch
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
print(f"✓ Active Compute Hardware: {gpu_name} (PyTorch {torch.__version__})")

# 3. Create Clean Working Workspace
work_dir = Path("/content/road_hazards_unified")
if work_dir.exists():
    shutil.rmtree(work_dir)

for split in ["train", "val", "test"]:
    (work_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (work_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

raw_downloads = Path("/content/raw_datasets")
raw_downloads.mkdir(parents=True, exist_ok=True)

# 4. Download Datasets using Kaggle CLI
print("\n--- Ingesting Multi-Source Hazard Datasets ---")

# Dataset 1: Pothole Detection (Optimized)
pothole_dest = raw_downloads / "pothole_dataset"
pothole_dest.mkdir(exist_ok=True)
print("Downloading Roboflow Pothole Dataset...")
subprocess.run([
    "kaggle", "datasets", "download", "-d",
    "muskanverma24/pothole-detection-dataset-yolov11-optimized",
    "-p", str(pothole_dest), "--unzip"
], check=False)

# Dataset 2: Dashcam Barricades & Road Hazards
dashcam_dest = raw_downloads / "dashcam_dataset"
dashcam_dest.mkdir(exist_ok=True)
print("Downloading Dashcam Road Hazards Dataset...")
subprocess.run([
    "kaggle", "datasets", "download", "-d",
    "jee1oner/dashcam-pov-potholes-and-barricades-for-yolo26n",
    "-p", str(dashcam_dest), "--unzip"
], check=False)

# 5. Applied Data Science Preprocessing & Multi-Class Harmonization
# Class 0: pothole
# Class 1: construction_hazard
print("\n--- Unifying & Balancing Annotation Schemas ---")

samples_potholes = []      # (img_path, [yolo_label_lines])
samples_construction = []  # (img_path, [yolo_label_lines])

# Process Potholes from Dataset 1
pothole_img_dirs = list(pothole_dest.rglob("images"))
for p_dir in pothole_img_dirs:
    lbl_dir = p_dir.parent / "labels"
    if not lbl_dir.exists():
        continue
    for img_p in p_dir.glob("*.*"):
        if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        with open(lbl_p, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        
        # Remap to class 0 (pothole)
        remapped = []
        for l in lines:
            parts = l.split()
            parts[0] = "0"
            remapped.append(" ".join(parts))
        samples_potholes.append((img_p, remapped))

print(f"✓ Extracted {len(samples_potholes)} Pothole Hazard Samples")

# Process Construction Hazards / Barricades from Dataset 2
dashcam_img_dirs = list(dashcam_dest.rglob("images"))
for d_dir in dashcam_img_dirs:
    lbl_dir = d_dir.parent / "labels"
    if not lbl_dir.exists():
        continue
    for img_p in d_dir.glob("*.*"):
        if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        with open(lbl_p, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        
        remapped = []
        has_construction = False
        for l in lines:
            parts = l.split()
            # In dashcam-pov, class 0 represents barricade/construction hazard
            cls_id = int(parts[0])
            if cls_id == 0:
                parts[0] = "1"  # Remap to class 1: construction_hazard
                remapped.append(" ".join(parts))
                has_construction = True
            elif cls_id == 1:
                parts[0] = "0"  # Also contains potholes! Remap to class 0
                remapped.append(" ".join(parts))
        
        if remapped:
            if has_construction:
                samples_construction.append((img_p, remapped))
            else:
                samples_potholes.append((img_p, remapped))

print(f"✓ Extracted {len(samples_construction)} Construction Hazard Samples")

# Stratified Split (80% Train, 10% Val, 10% Test)
random.seed(42)
random.shuffle(samples_potholes)
random.shuffle(samples_construction)

def split_items(items):
    n = len(items)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:]
    }

split_potholes = split_items(samples_potholes)
split_const = split_items(samples_construction)

for sp in ["train", "val", "test"]:
    combined_split = split_potholes[sp] + split_const[sp]
    random.shuffle(combined_split)
    print(f"Populating {sp} split with {len(combined_split)} images...")
    for idx, (img_p, labels) in enumerate(combined_split):
        ext = img_p.suffix
        new_name = f"haz_{sp}_{idx:05d}{ext}"
        shutil.copy2(img_p, work_dir / "images" / sp / new_name)
        with open(work_dir / "labels" / sp / f"haz_{sp}_{idx:05d}.txt", "w") as f:
            f.write("\n".join(labels) + "\n")

# 6. Generate Dataset YAML
yaml_path = work_dir / "dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {work_dir}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
  1: construction_hazard
""")

print("✓ Dataset YAML generated successfully.")

# 7. Model Training with Applied Data Science Hyperparameters
print("\n=== Initializing YOLO11s Multi-Hazard Model ===")
model = YOLO("yolo11s.pt")
project_dir = "/content/runs"
exp_name = "multihazard_yolo11s"

train_args = {
    "data": str(yaml_path),
    "epochs": 50,
    "imgsz": 640,
    "batch": 32,
    "optimizer": "AdamW",
    "lr0": 0.0025,
    "lrf": 0.001,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 8.0,          # Enhanced bbox regression
    "cls": 1.5,          # High class loss to balance multi-hazard recognition
    "dfl": 1.5,          # Distribution focal loss for edge precision
    "close_mosaic": 10,  # Turn off mosaic 10 epochs before end
    "mixup": 0.15,
    "copy_paste": 0.25,
    "project": project_dir,
    "name": exp_name,
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "plots": True,
    "verbose": True
}

print("Starting training loop...")
results = model.train(**train_args)
best_pt = Path(project_dir) / exp_name / "weights" / "best.pt"
print(f"\n✓ Training completed! Best PyTorch weights: {best_pt}")

# 8. Test Split Evaluation
print("\n=== Evaluating on Held-Out Test Set ===")
eval_model = YOLO(str(best_pt))
test_metrics = eval_model.val(data=str(yaml_path), split="test", project=project_dir, name="test_eval", exist_ok=True)

precision = float(test_metrics.box.mp)
recall = float(test_metrics.box.mr)
map50 = float(test_metrics.box.map50)
map50_95 = float(test_metrics.box.map)

print(f"\n--- Unified Multi-Hazard Test Metrics ---")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"mAP@50:    {map50*100:.2f}%")
print(f"mAP@50-95: {map50_95*100:.2f}%")

# 9. Edge-AI Optimization: ONNX & Dynamic INT8 Quantization
print("\n=== Exporting ONNX Model for Raspberry Pi 5 ===")
onnx_path = eval_model.export(format="onnx", opset=12, dynamic=False)
onnx_file = Path(onnx_path)

int8_file = Path("/content/multihazard_yolo11s_int8.onnx")
print("\n=== Dynamic INT8 Quantization ===")
quantize_dynamic(
    model_input=str(onnx_file),
    model_output=str(int8_file),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False
)

int8_size = int8_file.stat().st_size / (1024**2)
comp_ratio = onnx_file.stat().st_size / int8_file.stat().st_size
print(f"✓ INT8 Quantized Model: {int8_file} ({int8_size:.2f} MB, {comp_ratio:.2f}x compression)")

# 10. Save Deliverables & Benchmark Summary
shutil.copy2(best_pt, "/content/multihazard_yolo11s_best.pt")
shutil.copy2(onnx_file, "/content/multihazard_yolo11s_best.onnx")

summary = {
    "model": "YOLO11s_MultiHazard_Unified",
    "target_hardware": "Raspberry Pi 5 (8GB/16GB) & Edge RSUs",
    "classes": ["pothole", "construction_hazard"],
    "total_images": len(samples_potholes) + len(samples_construction),
    "epochs": 50,
    "precision": precision,
    "recall": recall,
    "map50": map50,
    "map50_95": map50_95,
    "pt_size_mb": best_pt.stat().st_size / (1024**2),
    "onnx_size_mb": onnx_file.stat().st_size / (1024**2),
    "int8_size_mb": int8_size,
    "compression_ratio": comp_ratio,
    "total_time_minutes": (time.time() - start_time) / 60
}

with open("/content/multihazard_evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Google Colab Multi-Hazard Pipeline Run Completed Successfully! ===")
