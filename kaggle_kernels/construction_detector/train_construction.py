import os
import sys
import json
import time
import shutil
import random
from pathlib import Path
from collections import Counter
import subprocess

print("=== Starting Dedicated Construction Hazard Detector Training on Kaggle GPU ===")
start_time = time.time()

# Install ultralytics & onnx packages if needed
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"], check=True)

# Locate dataset in /kaggle/input
input_base = Path("/kaggle/input")
label_studio_dirs = list(input_base.rglob("label_studio_dataset"))
if label_studio_dirs:
    dataset_source = label_studio_dirs[0]
else:
    # Fallback search for labels
    label_dirs = list(input_base.rglob("labels"))
    dataset_source = label_dirs[0].parent

print(f"Located source dataset at: {dataset_source}")
images_dir = dataset_source / "images"
labels_dir = dataset_source / "labels"

working_dir = Path("/kaggle/working/construction_dataset")
if working_dir.exists():
    shutil.rmtree(working_dir)

for split in ["train", "val", "test"]:
    (working_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (working_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Filter for barricade images (class 0 in label_studio) and clean negative images
barricade_pairs = []
empty_pairs = []

for img_p in images_dir.glob("*.*"):
    lbl_p = labels_dir / f"{img_p.stem}.txt"
    if not lbl_p.exists():
        continue
    
    has_barricade = False
    is_empty = True
    with open(lbl_p, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                is_empty = False
                if int(parts[0]) == 0:  # 0 is barricade
                    has_barricade = True
                    break
    
    if has_barricade:
        barricade_pairs.append((img_p, lbl_p))
    elif is_empty:
        empty_pairs.append((img_p, lbl_p))

print(f"Found {len(barricade_pairs)} barricade images and {len(empty_pairs)} clean negative images.")

random.seed(42)
random.shuffle(barricade_pairs)
random.shuffle(empty_pairs)

# Combine with 80/10/10 split
splits = {"train": [], "val": [], "test": []}

for group in [barricade_pairs, empty_pairs]:
    n = len(group)
    n_tr = int(n * 0.80)
    n_v = int(n * 0.10)
    splits["train"].extend(group[:n_tr])
    splits["val"].extend(group[n_tr:n_tr + n_v])
    splits["test"].extend(group[n_tr + n_v:])

print(f"Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

# Process and write files
total_boxes = Counter()

for split_name, pairs in splits.items():
    d_img = working_dir / "images" / split_name
    d_lbl = working_dir / "labels" / split_name
    
    for img_p, lbl_p in pairs:
        shutil.copy2(img_p, d_img / img_p.name)
        new_lines = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 0:  # barricade
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    new_lines.append(f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    total_boxes[split_name] += 1
        
        with open(d_lbl / f"{img_p.stem}.txt", "w") as f:
            f.writelines(new_lines)

print("Bounding boxes written per split:", dict(total_boxes))

yaml_path = working_dir / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {working_dir}
train: images/train
val: images/val
test: images/test

names:
  0: construction_hazard
""")

# Train YOLO11n
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model = YOLO("yolo11n.pt")
project_dir = "/kaggle/working/runs"
exp_name = "construction_yolo11n"

train_args = {
    "data": str(yaml_path),
    "epochs": 40,
    "imgsz": 640,
    "batch": 32,
    "optimizer": "AdamW",
    "lr0": 0.003,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 2,
    "close_mosaic": 5,
    "project": project_dir,
    "name": exp_name,
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "plots": True,
    "verbose": True
}

print("\n--- Training Parameters ---")
for k, v in train_args.items():
    print(f"  {k}: {v}")

results = model.train(**train_args)
best_pt = Path(project_dir) / exp_name / "weights" / "best.pt"
print(f"\nTraining completed! Best PyTorch weights: {best_pt}")

# Test Set Evaluation
print("\n=== Evaluating on Held-Out Test Set ===")
eval_model = YOLO(str(best_pt))
test_metrics = eval_model.val(data=str(yaml_path), split="test", project=project_dir, name="test_eval", exist_ok=True)

precision = float(test_metrics.box.mp)
recall = float(test_metrics.box.mr)
map50 = float(test_metrics.box.map50)
map50_95 = float(test_metrics.box.map)

print(f"\n--- Construction Hazard Test Metrics ---")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"mAP@50:    {map50*100:.2f}%")
print(f"mAP@50-95: {map50_95*100:.2f}%")

# ONNX Export
print("\n=== Exporting Full Precision ONNX Model ===")
onnx_path = eval_model.export(format="onnx", opset=12, dynamic=False)
onnx_file = Path(onnx_path)

# Dynamic INT8 Quantization
print("\n=== Dynamic INT8 Quantization ===")
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
print(f"INT8 Model generated: {int8_file} ({int8_size:.2f} MB, {comp_ratio:.2f}x compression)")

# Copy main artifacts to /kaggle/working root for immediate retrieval
shutil.copy2(best_pt, "/kaggle/working/best.pt")
shutil.copy2(onnx_file, "/kaggle/working/best.onnx")

exp_dir = Path(project_dir) / exp_name
for fname in ["results.csv", "results.png", "confusion_matrix.png"]:
    fpath = exp_dir / fname
    if fpath.exists():
        shutil.copy2(fpath, f"/kaggle/working/{fname}")

# Save summary
summary = {
    "model": "YOLO11n",
    "hazard": "construction_hazard",
    "train_images": len(splits["train"]),
    "val_images": len(splits["val"]),
    "test_images": len(splits["test"]),
    "epochs": 40,
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

with open("/kaggle/working/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== All Construction Hazard Detector Deliverables Ready in /kaggle/working ===")
