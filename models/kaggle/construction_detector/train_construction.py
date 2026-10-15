import os
import sys
import json
import time
import shutil
import random
from pathlib import Path
from collections import Counter
import subprocess

print("=== Starting 10,000+ Image Road Construction Hazard Detector Training on Kaggle GPU ===")
start_time = time.time()

# 1. Install ultralytics, onnx, onnxruntime
try:
    import ultralytics
    import onnx
    import onnxruntime
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"], check=True)

input_base = Path("/kaggle/input")
working_dir = Path("/kaggle/working/construction_superboost_dataset")
if working_dir.exists():
    shutil.rmtree(working_dir)

for split in ["train", "val", "test"]:
    (working_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (working_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

all_samples = []  # List of tuples: (image_path, label_content_lines, prefix)

# ==========================================
# Source 1: Dashcam POV Road Barricades (jee1oner)
# ==========================================
dashcam_dirs = list(input_base.rglob("*dashcam*"))
dashcam_base = None
for d in dashcam_dirs:
    for sub in d.rglob("label_studio_dataset"):
        dashcam_base = sub
        break

dc_barricade_count = 0
dc_neg_count = 0
if dashcam_base:
    dc_img_dir = dashcam_base / "images"
    dc_lbl_dir = dashcam_base / "labels"
    for img_p in sorted(dc_img_dir.glob("*.*")):
        lbl_p = dc_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        
        boxes = []
        is_clean_neg = True
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 0:  # 0 is Barricade / Construction warning barrier
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    boxes.append(f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    is_clean_neg = False
                elif cls_id == 1:
                    is_clean_neg = False
        
        if boxes:
            all_samples.append((img_p, boxes, f"dc_barr_{img_p.stem}"))
            dc_barricade_count += 1
        elif is_clean_neg:
            all_samples.append((img_p, [], f"dc_neg_{img_p.stem}"))
            dc_neg_count += 1

print(f"[Source 1] Loaded {dc_barricade_count} dashcam barricades + {dc_neg_count} clean negative road images.")

# ==========================================
# Source 2: ROD Road Obstacle & Construction Cones/Barrels (abtinzandi)
# ==========================================
rod_dirs = list(input_base.rglob("*obstacle-detection-dataset*")) or list(input_base.rglob("*ROD-Dataset*"))
rod_count = 0
for rd in rod_dirs:
    for img_p in rd.rglob("*.jpg"):
        lbl_p = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            lbl_p = img_p.with_suffix(".txt")
        if not lbl_p.exists():
            continue
        
        boxes = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    # Map obstacle/cone/barrel instances to class 0
                    boxes.append(f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
        
        if boxes:
            all_samples.append((img_p, boxes, f"rod_{img_p.stem}"))
            rod_count += 1

print(f"[Source 2] Loaded {rod_count} ROD construction/barrier images.")

# ==========================================
# Source 3: Construction Site Safety & Work Zones (snehilsanyal)
# ==========================================
css_dirs = list(input_base.rglob("*construction-site-safety*"))
css_count = 0
for cd in css_dirs:
    for img_p in cd.rglob("*.jpg"):
        lbl_p = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            lbl_p = img_p.with_suffix(".txt")
        if not lbl_p.exists():
            continue
        
        boxes = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    boxes.append(f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
        
        if boxes:
            all_samples.append((img_p, boxes, f"css_{img_p.stem}"))
            css_count += 1

print(f"[Source 3] Loaded {css_count} Construction Site Safety images.")
print(f"\n=======================================================")
print(f"TOTAL AGGREGATED CONSTRUCTION HAZARD IMAGES: {len(all_samples)}")
print(f"=======================================================")

# ==========================================
# 4. Stratified Split & Dataset Construction
# ==========================================
random.seed(42)
random.shuffle(all_samples)

n_total = len(all_samples)
n_train = int(n_total * 0.80)
n_val = int(n_total * 0.10)

splits = {
    "train": all_samples[:n_train],
    "val": all_samples[n_train:n_train + n_val],
    "test": all_samples[n_train + n_val:]
}

print(f"Splits -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

for s_name, items in splits.items():
    d_img = working_dir / "images" / s_name
    d_lbl = working_dir / "labels" / s_name
    for img_p, boxes, stem in items:
        shutil.copy2(img_p, d_img / f"{stem}{img_p.suffix}")
        with open(d_lbl / f"{stem}.txt", "w") as f:
            f.writelines(boxes)

yaml_path = working_dir / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {working_dir}
train: images/train
val: images/val
test: images/test

names:
  0: construction_hazard
""")

# ==========================================
# 5. Train YOLO11s on Kaggle GPU
# ==========================================
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model = YOLO("yolo11s.pt")
project_dir = "/kaggle/working/runs"
exp_name = "construction_yolo11s_10k"

train_args = {
    "data": str(yaml_path),
    "epochs": 40,
    "imgsz": 768,             # Sub-pixel boundary resolution for distant cones/barricades
    "batch": 20,              # Tuned for 16GB GPU VRAM at 768px
    "optimizer": "AdamW",
    "lr0": 0.0015,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 8.5,
    "cls": 1.0,
    "dfl": 1.8,
    "close_mosaic": 10,
    "project": project_dir,
    "name": exp_name,
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "plots": True,
    "verbose": True
}

print("\n--- Starting YOLO11s 40-Epoch GPU Training ---")
results = model.train(**train_args)
best_pt = Path(project_dir) / exp_name / "weights" / "best.pt"
print(f"\nTraining completed! Best PyTorch weights: {best_pt}")

# ==========================================
# 6. Evaluation & INT8 Quantization
# ==========================================
print("\n=== Evaluating on Held-Out Test Split ===")
eval_model = YOLO(str(best_pt))
test_metrics = eval_model.val(data=str(yaml_path), split="test", imgsz=768, project=project_dir, name="test_eval", exist_ok=True)

precision = float(test_metrics.box.mp)
recall = float(test_metrics.box.mr)
map50 = float(test_metrics.box.map50)
map50_95 = float(test_metrics.box.map)

print(f"\n--- Final Test Metrics ---")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"mAP@50:    {map50*100:.2f}%")
print(f"mAP@50-95: {map50_95*100:.2f}%")

print("\n=== Exporting ONNX Model ===")
onnx_path = eval_model.export(format="onnx", opset=12, dynamic=False)
onnx_file = Path(onnx_path)

int8_file = Path("/kaggle/working/best_int8.onnx")
quantize_dynamic(
    model_input=str(onnx_file),
    model_output=str(int8_file),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False
)

int8_size = int8_file.stat().st_size / (1024**2)
comp_ratio = onnx_file.stat().st_size / int8_file.stat().st_size
print(f"Quantized INT8 Model: {int8_file} ({int8_size:.2f} MB, {comp_ratio:.2f}x compression)")

shutil.copy2(best_pt, "/kaggle/working/best.pt")
shutil.copy2(onnx_file, "/kaggle/working/best.onnx")

summary = {
    "model": "YOLO11s_Construction_10k_Plus",
    "hazard": "construction_hazard",
    "total_images": len(all_samples),
    "train_images": len(splits["train"]),
    "val_images": len(splits["val"]),
    "test_images": len(splits["test"]),
    "epochs": 40,
    "precision": precision,
    "recall": recall,
    "map50": map50,
    "map50_95": map50_95,
    "onnx_size_mb": onnx_file.stat().st_size / (1024**2),
    "int8_size_mb": int8_size,
    "total_time_minutes": (time.time() - start_time) / 60
}

with open("/kaggle/working/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Construction Hazard 10k+ Model Training Successfully Completed ===")
