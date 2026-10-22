"""
PFPD Unified Three-Class Road Hazard Detector: Controlled v3 Ablation B
===================================================================================
Ablation: B (Ablation B: v3 Balanced Dataset + Stem-Only Freeze (Layers 0-2))
Freeze Target: 3 layers

Classes:
  0: pothole                 (Road surface craters, subsidence, asphalt voids)
  1: construction_hazard     (Workzone barricades, cones, concrete Jersey barriers)
  2: road_accident_obstacle  (Vehicle collisions, overturned vehicles, fallen trees, boulders)

Controlled Settings:
  - Base Model : unified_yolo11n_best.pt (50-epoch frozen baseline checkpoint)
  - Epochs     : 10 epochs (patience=4 early stopping)
  - Batch Size : 32
  - Resolution : 512x512
  - Learning Rate: lr0=5.0e-5, lrf=0.02 (cosine decay to 1.0e-6)
  - Mosaic     : 0.15 (mild multi-scale augmentation, close_mosaic=3 for final precision)
  - Scale      : 0.20 (scale jitter to bridge large trees vs small cones)
  - Export     : 384x384 static ONNX opset 17
"""

import os
import sys
import time
import json
import shutil
import random
import hashlib
import subprocess
from pathlib import Path
from collections import Counter

ABLATION = "B"
FREEZE_LAYERS = 3
ABLATION_DESC = "Ablation B: v3 Balanced Dataset + Stem-Only Freeze (Layers 0-2)"
EPOCHS = 10
BATCH_SIZE = 32
IMGSZ = 512
LR0 = 5.0e-5
LRF = 0.02
MOSAIC = 0.15
CLOSE_MOSAIC = 3
SCALE = 0.20

print("=" * 80)
print(f"  PFPD CONTROLLED v3 YOLO11n ROAD HAZARD PIPELINE: ABLATION {ABLATION}")
print(f"  {ABLATION_DESC}")
print("=" * 80)
start_time = time.time()

# -------------------------------------------------------------------------
# 1. Environment & Accelerator Verification
# -------------------------------------------------------------------------
print("\n--- STAGE 1: ACCELERATOR & ENVIRONMENT AUDIT ---")
print(f"Python Version : {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Accelerator: {gpu_name} ({gpu_mem_gb:.2f} GB VRAM)")
    else:
        gpu_name = "CPU"
        gpu_mem_gb = 0.0
        print("WARNING: CUDA is NOT available! Running on CPU.")
except ImportError:
    torch = None
    gpu_name = "Unavailable"
    gpu_mem_gb = 0.0

try:
    import ultralytics
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
    import yaml
except ImportError:
    print("Installing ultralytics, onnx, onnxruntime, pyyaml...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime", "pyyaml"], check=True)
    import ultralytics
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
    import yaml

print(f"Ultralytics Ver: {ultralytics.__version__}")
print(f"ONNX Version   : {onnx.__version__}")
print(f"ONNX Runtime   : {ort.__version__}")

disk_stat = shutil.disk_usage("/kaggle/working")
print(f"Working Storage: {disk_stat.free / (1024**3):.2f} GB free of {disk_stat.total / (1024**3):.2f} GB total")

# -------------------------------------------------------------------------
# 2. Dataset Provisioning & Symlink Setup
# -------------------------------------------------------------------------
print("\n--- STAGE 2: DATASET PROVISIONING & AUDIT ---")

# 2.1 Locate Base Weights
pt_candidates = list(Path("/kaggle/input").rglob("unified_yolo11n_best.pt"))
if not pt_candidates:
    raise FileNotFoundError("Base weights 'unified_yolo11n_best.pt' not found anywhere in /kaggle/input!")
base_weights = pt_candidates[0]
print(f"Base Weights   : {base_weights} ({base_weights.stat().st_size} bytes)")

# 2.2 Locate Dataset Source Directory containing images/train
img_train_candidates = list(Path("/kaggle/input").rglob("images/train"))
if not img_train_candidates:
    img_candidates = [p for p in Path("/kaggle/input").rglob("train") if "images" in str(p)]
    if img_candidates:
        dataset_source_dir = img_candidates[0].parent.parent
    else:
        raise FileNotFoundError("Could not locate 'images/train' directory in /kaggle/input!")
else:
    dataset_source_dir = img_train_candidates[0].parent.parent

print(f"Dataset Source : {dataset_source_dir}")

# 2.3 Setup Writable Working Dataset Directory
working_dataset = Path(f"/kaggle/working/finetune_v3_ablation_{ABLATION.lower()}")
if working_dataset.exists():
    shutil.rmtree(working_dataset)
working_dataset.mkdir(parents=True, exist_ok=True)

for split in ["train", "val"]:
    (working_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
    (working_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

split_img_counts = Counter()
split_inst_counts = {0: Counter(), 1: Counter(), 2: Counter()}

for split in ["train", "val"]:
    src_img_dir = dataset_source_dir / "images" / split
    src_lbl_dir = dataset_source_dir / "labels" / split
    dst_img_dir = working_dataset / "images" / split
    dst_lbl_dir = working_dataset / "labels" / split

    for img_p in src_img_dir.glob("*.*"):
        if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            dst_img = dst_img_dir / img_p.name
            try:
                os.symlink(img_p, dst_img)
            except OSError:
                shutil.copy2(img_p, dst_img)
            split_img_counts[split] += 1

            lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
            dst_lbl = dst_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.copy2(lbl_p, dst_lbl)
                with open(lbl_p, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            split_inst_counts[cls_id][split] += 1

print(f"Images linked  : Train = {split_img_counts['train']}, Val = {split_img_counts['val']}")
for cid, cname in {0: "pothole", 1: "construction_hazard", 2: "road_accident_obstacle"}.items():
    print(f"  Class {cid} ({cname:24s}): Train = {split_inst_counts[cid]['train']}, Val = {split_inst_counts[cid]['val']}")

# 2.4 Write data.yaml pointing to working dataset
data_yaml_path = working_dataset / "data.yaml"
data_yaml_content = {
    "path": str(working_dataset.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": 3,
    "names": ["pothole", "construction_hazard", "road_accident_obstacle"]
}
with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)
print(f"Data YAML generated at: {data_yaml_path}")

artifacts_dir = Path(f"/kaggle/working/pfpd_artifacts_ablation_{ABLATION.lower()}")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 3. Base Model Baseline Evaluation on Validation Split
# -------------------------------------------------------------------------
print("\n--- STAGE 3: BASELINE MODEL PRE-FLIGHT EVALUATION ---")
base_model = YOLO(str(base_weights))
print("Evaluating baseline model (50-epoch checkpoint) on v3 validation split (384x384)...")
val_res_base = base_model.val(
    data=str(data_yaml_path),
    imgsz=384,
    batch=BATCH_SIZE,
    device=0 if torch.cuda.is_available() else "cpu",
    plots=False,
    verbose=False
)

base_metrics = {
    "mAP50": float(val_res_base.box.map50),
    "mAP50_95": float(val_res_base.box.map),
    "per_class": {}
}
for idx, cname in enumerate(val_res_base.names.values()):
    p = float(val_res_base.box.p[idx]) if idx < len(val_res_base.box.p) else 0.0
    r = float(val_res_base.box.r[idx]) if idx < len(val_res_base.box.r) else 0.0
    f1 = float(val_res_base.box.f1[idx]) if idx < len(val_res_base.box.f1) else 0.0
    m50 = float(val_res_base.box.ap50[idx]) if idx < len(val_res_base.box.ap50) else 0.0
    base_metrics["per_class"][cname] = {"precision": p, "recall": r, "f1": f1, "mAP50": m50}
    print(f"  Baseline Class {idx} ({cname:24s}): P={p:.4f}, R={r:.4f}, F1={f1:.4f}, mAP50={m50:.4f}")

# -------------------------------------------------------------------------
# 4. Execute Ablation Training
# -------------------------------------------------------------------------
print(f"\n--- STAGE 4: EXECUTING ABLATION {ABLATION} TRAINING ({EPOCHS} EPOCHS, FREEZE {FREEZE_LAYERS}) ---")
ft_model = YOLO(str(base_weights))

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

t_train_start = time.time()
train_results = ft_model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE,
    freeze=FREEZE_LAYERS,
    lr0=LR0,
    lrf=LRF,
    cos_lr=True,
    warmup_epochs=1,
    mosaic=MOSAIC,
    close_mosaic=CLOSE_MOSAIC,
    scale=SCALE,
    box=7.5,
    cls=1.2,
    dfl=1.5,
    patience=4,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=4,
    save=True,
    plots=True,
    project="/kaggle/working/runs",
    name=f"ablation_{ABLATION.lower()}"
)
train_duration = time.time() - t_train_start
print(f"\nAblation {ABLATION} Training completed in {train_duration:.1f}s ({train_duration / 60:.2f} mins).")

# -------------------------------------------------------------------------
# 5. Candidate Validation & Comparison
# -------------------------------------------------------------------------
print(f"\n--- STAGE 5: POST-TRAINING VALIDATION & GAIN/LOSS AUDIT ---")
best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
if not best_pt.exists():
    best_pt = Path(train_results.save_dir) / "weights" / "last.pt"

cand_model = YOLO(str(best_pt))
val_res_cand = cand_model.val(
    data=str(data_yaml_path),
    imgsz=384,
    batch=BATCH_SIZE,
    device=0 if torch.cuda.is_available() else "cpu",
    plots=True,
    verbose=False
)

cand_metrics = {
    "mAP50": float(val_res_cand.box.map50),
    "mAP50_95": float(val_res_cand.box.map),
    "per_class": {}
}

print("\n" + "=" * 76)
print(f"  PFPD ABLATION {ABLATION} vs BASELINE VALIDATION AUDIT (384x384)")
print("=" * 76)
print(f"{'Hazard Class':<26} | {'Base Rec':<10} | {'Cand Rec':<10} | {'Rec Delta':<10} | {'Base Prec':<10} | {'Cand Prec':<10}")
print("-" * 76)

for idx, cname in enumerate(val_res_cand.names.values()):
    p = float(val_res_cand.box.p[idx]) if idx < len(val_res_cand.box.p) else 0.0
    r = float(val_res_cand.box.r[idx]) if idx < len(val_res_cand.box.r) else 0.0
    f1 = float(val_res_cand.box.f1[idx]) if idx < len(val_res_cand.box.f1) else 0.0
    m50 = float(val_res_cand.box.ap50[idx]) if idx < len(val_res_cand.box.ap50) else 0.0
    cand_metrics["per_class"][cname] = {"precision": p, "recall": r, "f1": f1, "mAP50": m50}

    b_r = base_metrics["per_class"][cname]["recall"]
    b_p = base_metrics["per_class"][cname]["precision"]
    delta_r = r - b_r
    delta_str = f"{delta_r:+.2%}"
    print(f"{cname:<26} | {b_r:<10.2%} | {r:<10.2%} | {delta_str:<10} | {b_p:<10.2%} | {p:<10.2%}")

print("-" * 76)
print(f"Overall mAP50    : Baseline = {base_metrics['mAP50']:.4f} | Candidate = {cand_metrics['mAP50']:.4f} ({cand_metrics['mAP50'] - base_metrics['mAP50']:+.4f})")
print(f"Overall mAP50-95 : Baseline = {base_metrics['mAP50_95']:.4f} | Candidate = {cand_metrics['mAP50_95']:.4f} ({cand_metrics['mAP50_95'] - base_metrics['mAP50_95']:+.4f})")
print("=" * 76)

# -------------------------------------------------------------------------
# 6. Static ONNX Export (384x384 Ops 17)
# -------------------------------------------------------------------------
print("\n--- STAGE 6: STATIC ONNX EXPORT (384x384 OPS 17) ---")
export_onnx_384 = cand_model.export(
    format="onnx",
    imgsz=384,
    opset=17,
    simplify=True,
    dynamic=False
)

target_onnx = artifacts_dir / f"unified_yolo11n_ablation_{ABLATION.lower()}_384.onnx"
target_pt = artifacts_dir / f"unified_yolo11n_ablation_{ABLATION.lower()}_best.pt"
shutil.copy2(export_onnx_384, target_onnx)
shutil.copy2(best_pt, target_pt)

# Also copy to /kaggle/working root for direct download accessibility
shutil.copy2(export_onnx_384, Path(f"/kaggle/working/unified_yolo11n_ablation_{ABLATION.lower()}_384.onnx"))
shutil.copy2(best_pt, Path(f"/kaggle/working/unified_yolo11n_ablation_{ABLATION.lower()}_best.pt"))

# Copy run plots and confusion matrices
runs_weights_dir = Path(train_results.save_dir)
for plot_file in runs_weights_dir.glob("*.png"):
    shutil.copy2(plot_file, artifacts_dir / plot_file.name)
for plot_file in runs_weights_dir.glob("*.jpg"):
    shutil.copy2(plot_file, artifacts_dir / plot_file.name)
if (runs_weights_dir / "results.csv").exists():
    shutil.copy2(runs_weights_dir / "results.csv", artifacts_dir / "results.csv")

# Save complete metrics report
final_report = {
    "ablation": ABLATION,
    "description": ABLATION_DESC,
    "freeze_layers": FREEZE_LAYERS,
    "train_duration_sec": train_duration,
    "baseline_metrics": base_metrics,
    "candidate_metrics": cand_metrics,
    "hyperparameters": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "imgsz": IMGSZ,
        "lr0": LR0,
        "lrf": LRF,
        "mosaic": MOSAIC,
        "close_mosaic": CLOSE_MOSAIC,
        "scale": SCALE
    }
}
with open(artifacts_dir / f"ablation_{ABLATION.lower()}_metrics.json", "w") as f:
    json.dump(final_report, f, indent=2)
with open(Path(f"/kaggle/working/ablation_{ABLATION.lower()}_metrics.json"), "w") as f:
    json.dump(final_report, f, indent=2)

print(f"\nAblation {ABLATION} Artifacts successfully saved to {artifacts_dir} and /kaggle/working/")
print(f"Total Pipeline Runtime: {time.time() - start_time:.1f} seconds.")
