import os
import sys
import json
import time
import shutil
import random
from pathlib import Path
import subprocess

print("=== Starting 10,000+ Image Road Accident & Obstacle Hazard Detector on Kaggle GPU ===")
start_time = time.time()

# 1. Install ultralytics, onnx, onnxruntime
try:
    import ultralytics
    import onnx
    import onnxruntime
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"], check=True)

input_base = Path("/kaggle/input")
working_dir = Path("/kaggle/working/accident_obstacle_superboost_dataset")
if working_dir.exists():
    shutil.rmtree(working_dir)

for split in ["train", "val", "test"]:
    (working_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (working_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

all_samples = []

# Source 1: CubeAI Traffic Accidents
cube_dirs = list(input_base.rglob("*traffic-accident-detection*"))
cube_count = 0
for cd in cube_dirs:
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
                    boxes.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        if boxes:
            all_samples.append((img_p, boxes, f"cube_{img_p.stem}"))
            cube_count += 1
print(f"[Source 1] Loaded {cube_count} CubeAI traffic accident images.")

# Source 2: Car Accidents and Deformation
mars_dirs = list(input_base.rglob("*car-accidents-and-deformation*"))
mars_count = 0
for md in mars_dirs:
    for img_p in md.rglob("*.jpg"):
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
                    boxes.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        if boxes:
            all_samples.append((img_p, boxes, f"mars_{img_p.stem}"))
            mars_count += 1
print(f"[Source 2] Loaded {mars_count} car accident & deformation images.")

# Source 3: Road Obstacle & Debris Detection
sak_dirs = list(input_base.rglob("*road-obstacledebris-detection*"))
sak_count = 0
for sd in sak_dirs:
    for img_p in sd.rglob("*.jpg"):
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
                    boxes.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        if boxes:
            all_samples.append((img_p, boxes, f"sak_{img_p.stem}"))
            sak_count += 1
print(f"[Source 3] Loaded {sak_count} road debris/obstacle images.")

# Source 4: ROD Road Obstacles
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
                    boxes.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        if boxes:
            all_samples.append((img_p, boxes, f"rod_{img_p.stem}"))
            rod_count += 1
            if rod_count >= 5000:
                break
print(f"[Source 4] Loaded {rod_count} ROD hazard/obstacle images.")

print(f"\n=======================================================")
print(f"TOTAL AGGREGATED ACCIDENT & OBSTACLE IMAGES: {len(all_samples)}")
print(f"=======================================================")

# Stratified Split
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
  0: road_accident_obstacle
""")

# Train YOLO11s on Kaggle GPU
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model = YOLO("yolo11s.pt")
project_dir = "/kaggle/working/runs"
exp_name = "accident_obstacle_yolo11s_10k"

train_args = {
    "data": str(yaml_path),
    "epochs": 40,
    "imgsz": 768,             # Sub-pixel boundary resolution for deformation and small debris
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

print("\n--- Starting YOLO11s 40-Epoch GPU Training on Kaggle ---")
results = model.train(**train_args)
best_pt = Path(project_dir) / exp_name / "weights" / "best.pt"
print(f"\nTraining completed! Best PyTorch weights: {best_pt}")

# Evaluation & INT8 Quantization
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
    "model": "YOLO11s_Accident_Obstacle_10k_Plus",
    "hazard": "road_accident_obstacle",
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

print("\n=== Road Accident & Obstacle 10k+ Model Training Successfully Completed ===")
