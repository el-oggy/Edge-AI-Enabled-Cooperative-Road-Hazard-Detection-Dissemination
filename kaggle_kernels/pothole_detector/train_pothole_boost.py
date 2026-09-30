import os
import sys
import json
import time
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import subprocess

print("=== Starting Boosted Pothole Hazard Detector Training on Kaggle GPU ===")
start_time = time.time()

# Install ultralytics & onnx packages if needed
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"], check=True)

input_base = Path("/kaggle/input")
working_dir = Path("/kaggle/working/pothole_boost_dataset")
if working_dir.exists():
    shutil.rmtree(working_dir)

for split in ["train", "val", "test"]:
    (working_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (working_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# 1. Parse Andrew MVD dataset
andrew_dirs = list(input_base.rglob("andrewmvd*")) or list(input_base.rglob("*pothole-detection*"))
andrew_base = None
for d in andrew_dirs:
    if (d / "images").exists() and (d / "annotations").exists():
        andrew_base = d
        break
    for sub in d.rglob("images"):
        if (sub.parent / "annotations").exists():
            andrew_base = sub.parent
            break

print(f"Andrew MVD dataset path: {andrew_base}")

andrew_pairs = []
if andrew_base:
    ann_dir = andrew_base / "annotations"
    img_dir = andrew_base / "images"
    for xml_file in sorted(ann_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                continue
            w = float(size.find("width").text)
            h = float(size.find("height").text)
            if w <= 0 or h <= 0:
                continue
            
            filename = root.find("filename").text if root.find("filename") is not None else f"{xml_file.stem}.png"
            img_file = img_dir / filename
            if not img_file.exists():
                # Try common extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = img_dir / f"{xml_file.stem}{ext}"
                    if candidate.exists():
                        img_file = candidate
                        break
            
            if not img_file.exists():
                continue
            
            boxes = []
            for obj in root.findall("object"):
                name = obj.find("name").text.lower().strip()
                if "pothole" in name:
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    # Convert to YOLO format
                    bx = ((xmin + xmax) / 2.0) / w
                    by = ((ymin + ymax) / 2.0) / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    bx, by, bw, bh = [max(0.0, min(1.0, c)) for c in [bx, by, bw, bh]]
                    boxes.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
            
            if boxes:
                andrew_pairs.append((img_file, boxes, xml_file.stem))
        except Exception as e:
            continue

print(f"Loaded {len(andrew_pairs)} valid annotated images from Andrew MVD.")

# 2. Parse Dashcam Hyderabad dataset (jee1oner)
dashcam_dirs = list(input_base.rglob("*dashcam*"))
dashcam_base = None
for d in dashcam_dirs:
    for sub in d.rglob("label_studio_dataset"):
        dashcam_base = sub
        break

print(f"Dashcam dataset path: {dashcam_base}")
dashcam_pothole_pairs = []
hard_negative_pairs = []

if dashcam_base:
    dc_img_dir = dashcam_base / "images"
    dc_lbl_dir = dashcam_base / "labels"
    for img_p in sorted(dc_img_dir.glob("*.*")):
        lbl_p = dc_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        
        boxes = []
        is_empty = True
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                is_empty = False
                cls_id = int(parts[0])
                if cls_id == 1:  # In this dataset, class 1 is pothole
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    boxes.append(f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
        
        if boxes:
            dashcam_pothole_pairs.append((img_p, boxes, f"dc_{img_p.stem}"))
        elif is_empty:
            hard_negative_pairs.append((img_p, [], f"neg_{img_p.stem}"))

print(f"Loaded {len(dashcam_pothole_pairs)} pothole images and {len(hard_negative_pairs)} hard negative road images from Dashcam dataset.")

# 3. Stratified Split Creation
random.seed(42)

# Keep the standard 67 Andrew test images in test split for frozen benchmark comparison
random.shuffle(andrew_pairs)
n_andrew_test = 67
andrew_test = andrew_pairs[:n_andrew_test]
andrew_rem = andrew_pairs[n_andrew_test:]
n_andrew_val = int(len(andrew_rem) * 0.10)
andrew_val = andrew_rem[:n_andrew_val]
andrew_train = andrew_rem[n_andrew_val:]

# Dashcam split (80/10/10)
random.shuffle(dashcam_pothole_pairs)
n_dc = len(dashcam_pothole_pairs)
n_dc_tr = int(n_dc * 0.80)
n_dc_val = int(n_dc * 0.10)
dc_train = dashcam_pothole_pairs[:n_dc_tr]
dc_val = dashcam_pothole_pairs[n_dc_tr:n_dc_tr + n_dc_val]
dc_test = dashcam_pothole_pairs[n_dc_tr + n_dc_val:]

# Hard negatives split
random.shuffle(hard_negative_pairs)
n_neg = len(hard_negative_pairs)
n_neg_tr = int(n_neg * 0.80)
n_neg_val = int(n_neg * 0.10)
neg_train = hard_negative_pairs[:n_neg_tr]
neg_val = hard_negative_pairs[n_neg_tr:n_neg_tr + n_neg_val]
neg_test = hard_negative_pairs[n_neg_tr + n_neg_val:]

final_splits = {
    "train": andrew_train + dc_train + neg_train,
    "val": andrew_val + dc_val + neg_val,
    "test": andrew_test + dc_test + neg_test,
}

print(f"\nFinal Combined Dataset Splits:")
print(f"  Train: {len(final_splits['train'])} images")
print(f"  Val:   {len(final_splits['val'])} images")
print(f"  Test:  {len(final_splits['test'])} images (includes {len(andrew_test)} frozen Andrew benchmark images)")

# Copy files
for split_name, items in final_splits.items():
    d_img = working_dir / "images" / split_name
    d_lbl = working_dir / "labels" / split_name
    for img_p, boxes, stem in items:
        dest_img = d_img / f"{stem}{img_p.suffix}"
        shutil.copy2(img_p, dest_img)
        dest_lbl = d_lbl / f"{stem}.txt"
        with open(dest_lbl, "w") as f:
            f.writelines(boxes)

# Generate data.yaml
yaml_path = working_dir / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {working_dir}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
""")

# 4. Train YOLO11n with Road-Hazard Optimized Hyperparameters
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model = YOLO("yolo11n.pt")
project_dir = "/kaggle/working/runs"
exp_name = "pothole_yolo11n_boosted"

train_args = {
    "data": str(yaml_path),
    "epochs": 50,
    "imgsz": 640,
    "batch": 32,
    "optimizer": "AdamW",
    "lr0": 0.003,
    "lrf": 0.001,  # Cosine decay for fine boundary regression
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 7.5,     # Increased box loss gain to maximize bounding box recall and IoU
    "cls": 0.5,     # Tuned classification gain
    "close_mosaic": 10,
    "project": project_dir,
    "name": exp_name,
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "plots": True,
    "verbose": True
}

print("\n--- Training Hyperparameters ---")
for k, v in train_args.items():
    print(f"  {k}: {v}")

results = model.train(**train_args)
best_pt = Path(project_dir) / exp_name / "weights" / "best.pt"
print(f"\nTraining complete! Best PyTorch model: {best_pt}")

# 5. Comprehensive Test Set Evaluation
print("\n=== Evaluating on Full Held-Out Test Set ===")
eval_model = YOLO(str(best_pt))
test_metrics = eval_model.val(data=str(yaml_path), split="test", project=project_dir, name="test_eval", exist_ok=True)

precision = float(test_metrics.box.mp)
recall = float(test_metrics.box.mr)
map50 = float(test_metrics.box.map50)
map50_95 = float(test_metrics.box.map)

print(f"\n--- Boosted Pothole Test Metrics ---")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"mAP@50:    {map50*100:.2f}%")
print(f"mAP@50-95: {map50_95*100:.2f}%")

# 6. ONNX Export
print("\n=== Exporting ONNX Model ===")
onnx_path = eval_model.export(format="onnx", opset=12, dynamic=False)
onnx_file = Path(onnx_path)

# 7. Dynamic INT8 Quantization
print("\n=== Performing Dynamic INT8 Quantization ===")
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
print(f"INT8 Model: {int8_file} ({int8_size:.2f} MB, {comp_ratio:.2f}x compression)")

# Copy deliverables to /kaggle/working root
shutil.copy2(best_pt, "/kaggle/working/best.pt")
shutil.copy2(onnx_file, "/kaggle/working/best.onnx")

exp_dir = Path(project_dir) / exp_name
for fname in ["results.csv", "results.png", "confusion_matrix.png"]:
    fpath = exp_dir / fname
    if fpath.exists():
        shutil.copy2(fpath, f"/kaggle/working/{fname}")

summary = {
    "model": "YOLO11n_Boosted",
    "hazard": "pothole",
    "train_images": len(final_splits["train"]),
    "val_images": len(final_splits["val"]),
    "test_images": len(final_splits["test"]),
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

with open("/kaggle/working/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Boosted Pothole Detector Training & Quantization Complete ===")
