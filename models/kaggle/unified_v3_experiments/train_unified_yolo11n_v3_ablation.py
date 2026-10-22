"""
PFPD Unified Three-Class Road Hazard Detector: Controlled v3 Ablation Experiments
===================================================================================
Ablations:
  A: v3 Balanced Dataset + Original Freezing Configuration (--freeze 8, layers 0-7 frozen)
     --> Isolates the exact effect of dataset re-balancing from architectural changes.
  B: v3 Balanced Dataset + Stem-Only Freezing (--freeze 3, layers 0-2 frozen, C3k2+SPPF trainable)
     --> Measures the exact contribution of backbone multi-scale feature adaptation.

Classes:
  0: pothole                 (Road surface craters, subsidence, asphalt voids)
  1: construction_hazard     (Workzone barricades, cones, concrete Jersey barriers)
  2: road_accident_obstacle  (Vehicle collisions, overturned vehicles, fallen trees, boulders)

Common Controlled Settings:
  - Base Model : unified_yolo11n_best.pt (50-epoch frozen baseline checkpoint)
  - Epochs     : 10 epochs (with patience=4 early stopping)
  - Batch Size : 32 (auto-fallback to 16 on VRAM limits)
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
import argparse
import hashlib
import subprocess
from pathlib import Path
from collections import Counter

print("=" * 80)
print("  PFPD CONTROLLED v3 YOLO11n ROAD HAZARD EXPERIMENTAL PIPELINE")
print("=" * 80)
start_time = time.time()

parser = argparse.ArgumentParser(description="PFPD v3 Ablation Experiment Runner")
parser.add_argument("--ablation", type=str, choices=["A", "B"], default="A",
                    help="A: Freeze 8 (Backbone layers 0-7) | B: Freeze 3 (Stem layers 0-2)")
parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--imgsz", type=int, default=512, help="Training image size (default: 512)")
parser.add_argument("--lr0", type=float, default=5.0e-5, help="Initial learning rate (default: 5e-5)")
parser.add_argument("--mosaic", type=float, default=0.15, help="Mild mosaic probability (default: 0.15)")
args = parser.parse_args()

freeze_layers = 8 if args.ablation == "A" else 3
ablation_desc = "A: Balanced Replay + Original Freeze (Layers 0-7)" if args.ablation == "A" else "B: Balanced Replay + Stem-Only Freeze (Layers 0-2)"

print(f"\nTarget Experiment : Ablation {args.ablation}")
print(f"Description       : {ablation_desc}")
print(f"Freeze Target     : {freeze_layers} layers")
print(f"Initial LR        : {args.lr0}")
print(f"Mild Mosaic       : {args.mosaic} (close_mosaic=3)")
print(f"Epochs            : {args.epochs}")

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
except ImportError:
    print("Installing ultralytics, onnx, onnxruntime...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"], check=True)
    import ultralytics
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort

print(f"Ultralytics Ver: {ultralytics.__version__}")
print(f"ONNX Version   : {onnx.__version__}")
print(f"ONNX Runtime   : {ort.__version__}")

# Check Working Storage
disk_stat = shutil.disk_usage("/kaggle/working")
print(f"Working Storage: {disk_stat.free / (1024**3):.2f} GB free of {disk_stat.total / (1024**3):.2f} GB total")

# -------------------------------------------------------------------------
# 2. Dataset Ingestion & Verification
# -------------------------------------------------------------------------
print("\n--- STAGE 2: DATASET PROVISIONING & AUDIT ---")
manifest_candidates = list(Path("/kaggle/input").rglob("dataset_manifest.json"))
if manifest_candidates:
    dataset_input_dir = manifest_candidates[0].parent
else:
    pt_candidates = list(Path("/kaggle/input").rglob("unified_yolo11n_best.pt"))
    if pt_candidates:
        dataset_input_dir = pt_candidates[0].parent
    else:
        raise FileNotFoundError("Could not locate dataset root under /kaggle/input")

print(f"Dataset Input  : {dataset_input_dir}")
data_yaml_candidates = list(dataset_input_dir.rglob("data.yaml"))
if not data_yaml_candidates:
    raise FileNotFoundError("data.yaml not found in dataset directory!")
data_yaml_path = data_yaml_candidates[0]
print(f"Data YAML Path : {data_yaml_path}")

artifacts_dir = Path(f"/kaggle/working/pfpd_artifacts_ablation_{args.ablation.lower()}")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Locate baseline weights
base_weights_candidates = list(Path("/kaggle/input").rglob("unified_yolo11n_best.pt"))
if not base_weights_candidates:
    raise FileNotFoundError("Base weights 'unified_yolo11n_best.pt' not found in /kaggle/input!")
base_weights = base_weights_candidates[0]
print(f"Base Weights   : {base_weights}")

# -------------------------------------------------------------------------
# 3. Base Model Baseline Evaluation on Validation Split
# -------------------------------------------------------------------------
print("\n--- STAGE 3: BASELINE MODEL PRE-FLIGHT EVALUATION ---")
base_model = YOLO(str(base_weights))
print("Evaluating baseline model (50-epoch checkpoint) on v3 validation split...")
val_res_base = base_model.val(
    data=str(data_yaml_path),
    imgsz=384,
    batch=args.batch,
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
print(f"\n--- STAGE 4: EXECUTING ABLATION {args.ablation} TRAINING ({args.epochs} EPOCHS) ---")
ft_model = YOLO(str(base_weights))

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

t_train_start = time.time()
train_results = ft_model.train(
    data=str(data_yaml_path),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    freeze=freeze_layers,
    lr0=args.lr0,
    lrf=0.02,
    cos_lr=True,
    warmup_epochs=1,
    mosaic=args.mosaic,
    close_mosaic=3,
    scale=0.20,
    box=7.5,
    cls=1.2,
    dfl=1.5,
    patience=4,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=4,
    save=True,
    plots=True,
    project="/kaggle/working/runs",
    name=f"ablation_{args.ablation.lower()}"
)
train_duration = time.time() - t_train_start
print(f"\nAblation {args.ablation} Training completed in {train_duration:.1f}s ({train_duration / 60:.2f} mins).")

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
    batch=args.batch,
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
print(f"  PFPD ABLATION {args.ablation} vs BASELINE VALIDATION AUDIT (384x384)")
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
shutil.copy2(export_onnx_384, artifacts_dir / f"unified_yolo11n_ablation_{args.ablation.lower()}_384.onnx")
shutil.copy2(best_pt, artifacts_dir / f"unified_yolo11n_ablation_{args.ablation.lower()}_best.pt")

# Save complete metrics report
final_report = {
    "ablation": args.ablation,
    "description": ablation_desc,
    "freeze_layers": freeze_layers,
    "train_duration_sec": train_duration,
    "baseline_metrics": base_metrics,
    "candidate_metrics": cand_metrics,
    "hyperparameters": {
        "epochs": args.epochs,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "mosaic": args.mosaic,
        "close_mosaic": 3,
        "scale": 0.20
    }
}
with open(artifacts_dir / f"ablation_{args.ablation.lower()}_metrics.json", "w") as f:
    json.dump(final_report, f, indent=2)

print(f"\nAblation {args.ablation} Artifacts successfully saved to {artifacts_dir}")
print(f"Total Pipeline Runtime: {time.time() - start_time:.1f} seconds.")
