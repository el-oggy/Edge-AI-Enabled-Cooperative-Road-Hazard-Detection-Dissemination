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

print("=== Starting 10,000+ Image Pothole & Road Damage Detector Training on Kaggle GPU ===")
start_time = time.time()

# 1. Install ultralytics, onnx, onnxruntime
try:
    import ultralytics
    import onnx
    import onnxruntime
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"], check=True)

input_base = Path("/kaggle/input")
working_dir = Path("/kaggle/working/pothole_superboost_dataset")
if working_dir.exists():
    shutil.rmtree(working_dir)

for split in ["train", "val", "test"]:
    (working_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (working_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

all_samples = []  # List of tuples: (image_path, label_content_lines, prefix)

# ==========================================
# Source 1: Roboflow YOLOv11 Optimized Potholes (muskanverma24)
# ==========================================
rf_dirs = list(input_base.rglob("*pothole-detection-dataset-yolov11-optimized*")) or list(input_base.rglob("*vasanthakumar*"))
rf_base = None
for d in rf_dirs:
    if (d / "train" / "images").exists() or (d / "test" / "images").exists():
        rf_base = d
        break
    for sub in d.rglob("train"):
        if (sub / "images").exists():
            rf_base = sub.parent
            break

rf_count = 0
if rf_base:
    for sub_split in ["train", "valid", "test"]:
        sub_img_dir = rf_base / sub_split / "images"
        sub_lbl_dir = rf_base / sub_split / "labels"
        if not sub_img_dir.exists():
            continue
        for img_p in sub_img_dir.glob("*.*"):
            lbl_p = sub_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                with open(lbl_p, "r") as f:
                    lines = [f"0 {l.strip().split()[1]} {l.strip().split()[2]} {l.strip().split()[3]} {l.strip().split()[4]}\n"
                             for l in f if l.strip() and len(l.strip().split()) >= 5]
                if lines:
                    all_samples.append((img_p, lines, f"rf_{img_p.stem}"))
                    rf_count += 1

print(f"[Source 1] Loaded {rf_count} images from YOLOv11 Optimized dataset.")

# ==========================================
# Source 2: Andrew MVD Potholes (Pascal VOC XML)
# ==========================================
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

andrew_count = 0
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
                    
                    bx = ((xmin + xmax) / 2.0) / w
                    by = ((ymin + ymax) / 2.0) / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    bx, by, bw, bh = [max(0.0, min(1.0, c)) for c in [bx, by, bw, bh]]
                    boxes.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
            
            if boxes:
                all_samples.append((img_file, boxes, f"and_{xml_file.stem}"))
                andrew_count += 1
        except Exception:
            continue

print(f"[Source 2] Loaded {andrew_count} Andrew MVD pothole images.")

# ==========================================
# Source 3: Dashcam POV Potholes & Hard Negatives (jee1oner)
# ==========================================
dashcam_dirs = list(input_base.rglob("*dashcam*"))
dashcam_base = None
for d in dashcam_dirs:
    for sub in d.rglob("label_studio_dataset"):
        dashcam_base = sub
        break

dc_pothole_count = 0
dc_neg_count = 0
if dashcam_base:
    dc_img_dir = dashcam_base / "images"
    dc_lbl_dir = dashcam_base / "labels"
    for img_p in sorted(dc_img_dir.glob("*.*")):
        lbl_p = dc_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        
        has_barricade = False
        boxes = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 0:
                    has_barricade = True
                elif cls_id == 1:
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    if coords[2] * coords[3] >= 0.0002:
                        boxes.append(f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
        
        if not has_barricade:
            if boxes:
                all_samples.append((img_p, boxes, f"dc_{img_p.stem}"))
                dc_pothole_count += 1
            else:
                all_samples.append((img_p, [], f"neg_{img_p.stem}"))
                dc_neg_count += 1

print(f"[Source 3] Loaded {dc_pothole_count} dashcam potholes + {dc_neg_count} hard negatives.")

# ==========================================
# Source 4: Sabid Rahman Potholes, Cracks & Open Manholes
# ==========================================
sabid_dirs = list(input_base.rglob("*sabidrahman*")) or list(input_base.rglob("*pothole-cracks*"))
sabid_count = 0
for sd in sabid_dirs:
    # Look for image files inside classes/potholes, classes/cracks, classes/open_manhole
    for img_p in sd.rglob("*.jpg"):
        lbl_txt = img_p.parent.parent / "labels" / "txt" / f"{img_p.stem}.txt"
        if not lbl_txt.exists():
            lbl_txt = img_p.parent / f"{img_p.stem}.txt"
        if not lbl_txt.exists():
            lbl_txt = img_p.with_suffix(".txt")
        
        boxes = []
        if lbl_txt.exists():
            with open(lbl_txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        if boxes:
            all_samples.append((img_p, boxes, f"sab_{img_p.stem}"))
            sabid_count += 1

print(f"[Source 4] Loaded {sabid_count} Sabid Rahman road damage images.")

# ==========================================
# Source 5: RDD2022 India Pothole D40 Dataset (vidishbijalwan)
# ==========================================
rdd_dirs = list(input_base.rglob("*rdd2022*")) or list(input_base.rglob("*india-pothole*"))
rdd_count = 0
for rd in rdd_dirs:
    for xml_p in rd.rglob("*.xml"):
        try:
            tree = ET.parse(xml_p)
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                continue
            w = float(size.find("width").text)
            h = float(size.find("height").text)
            if w <= 0 or h <= 0:
                continue
            
            # Find matching image
            img_cand = None
            parent_d = xml_p.parent.parent
            for cand in [parent_d / "images" / f"{xml_p.stem}.jpg", xml_p.parent / f"{xml_p.stem}.jpg"]:
                if cand.exists():
                    img_cand = cand
                    break
            if not img_cand:
                matches = list(rd.rglob(f"{xml_p.stem}.jpg"))
                if matches:
                    img_cand = matches[0]
            if not img_cand or not img_cand.exists():
                continue
            
            boxes = []
            for obj in root.findall("object"):
                name = obj.find("name").text.upper().strip()
                # D40 is the international road damage standard class for Potholes
                # D00/D10/D20 are cracks
                if "D40" in name or "POTHOLE" in name or "D20" in name:
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    bx = ((xmin + xmax) / 2.0) / w
                    by = ((ymin + ymax) / 2.0) / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    bx, by, bw, bh = [max(0.0, min(1.0, c)) for c in [bx, by, bw, bh]]
                    boxes.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
            
            if boxes:
                all_samples.append((img_cand, boxes, f"rdd_{xml_p.stem}"))
                rdd_count += 1
        except Exception:
            continue

print(f"[Source 5] Loaded {rdd_count} RDD2022 road hazard images.")
print(f"\n=======================================================")
print(f"TOTAL AGGREGATED IMAGES FOR TRAINING: {len(all_samples)}")
print(f"=======================================================")

# ==========================================
# 6. Stratified Split & Dataset Construction
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
  0: pothole
""")

# ==========================================
# 7. Train YOLO11s on Kaggle GPU
# ==========================================
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model = YOLO("yolo11s.pt")
project_dir = "/kaggle/working/runs"
exp_name = "pothole_yolo11s_10k"

train_args = {
    "data": str(yaml_path),
    "epochs": 50,
    "imgsz": 768,             # 44% more resolution for sub-pixel boundary localization
    "batch": 20,              # Tuned for 16GB GPU VRAM at 768px
    "optimizer": "AdamW",
    "lr0": 0.0015,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 8.5,               # Higher weight for precise bounding box boundaries
    "cls": 1.0,
    "dfl": 1.8,               # Distribution Focal Loss for jagged pothole contours
    "close_mosaic": 10,       # Turn off mosaic for final 10 epochs for realistic edge alignment
    "copy_paste": 0.15,
    "mixup": 0.10,
    "scale": 0.5,
    "project": project_dir,
    "name": exp_name,
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "plots": True,
    "verbose": True
}

print("\n--- Starting YOLO11s 50-Epoch GPU Training ---")
results = model.train(**train_args)
best_pt = Path(project_dir) / exp_name / "weights" / "best.pt"
print(f"\nTraining completed! Best PyTorch weights: {best_pt}")

# ==========================================
# 8. Evaluation & INT8 Quantization
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
    "model": "YOLO11s_Pothole_10k_Plus",
    "hazard": "pothole",
    "total_images": len(all_samples),
    "train_images": len(splits["train"]),
    "val_images": len(splits["val"]),
    "test_images": len(splits["test"]),
    "epochs": 50,
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

print("\n=== Pothole Hazard 10k+ Model Training Successfully Completed ===")
