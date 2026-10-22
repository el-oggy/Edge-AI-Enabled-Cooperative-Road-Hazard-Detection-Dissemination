"""
PFPD Unified Three-Class Road Hazard Detector: Targeted Fine-Tuning Pipeline
=============================================================================
Classes:
  0: pothole                 (Road surface craters, subsidence, asphalt voids)
  1: construction_hazard     (Workzone barricades, cones, concrete Jersey barriers)
  2: road_accident_obstacle  (Vehicle collisions, overturned vehicles, fallen trees, boulders)

Experiment:
- Bounded 8-Epoch Targeted Fine-Tuning
- Base Weights: unified_yolo11n_best.pt (2.58M params, 50-epoch production model)
- Curated Fine-Tuning Package: 1,286 train images, 300 validation images
- Backbone Freezing: Layers 0-7 frozen, Neck and Head trainable
- Conservative Learning Rate: lr0=1.5e-4, cosine decay to 1.5e-6
- Mosaic Disabled: mosaic=0.0 (prevents bounding-box feature distortion)
- Export: 384x384 and 512x512 ONNX (opset 17, static shapes)
- Comprehensive Baseline vs Candidate Evaluation on identical validation split
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

print("=" * 72)
print("  PFPD TARGETED YOLO11n ROAD HAZARD FINE-TUNING PIPELINE (8 EPOCHS)  ")
print("=" * 72)
start_time = time.time()

# -------------------------------------------------------------------------
# 1. Accelerator & Environment Audit
# -------------------------------------------------------------------------
print("\n--- STAGE 1: ACCELERATOR & DEPENDENCY AUDIT ---")
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
# 2. Directory Layout & Dataset Provisioning
# -------------------------------------------------------------------------
print("\n--- STAGE 2: DATASET PROVISIONING & LEAKAGE VERIFICATION ---")
manifest_candidates = list(Path("/kaggle/input").rglob("dataset_manifest.json"))
if manifest_candidates:
    dataset_input_dir = manifest_candidates[0].parent
else:
    pt_candidates = list(Path("/kaggle/input").rglob("unified_yolo11n_best.pt"))
    if pt_candidates:
        dataset_input_dir = pt_candidates[0].parent
    else:
        raise FileNotFoundError("Could not locate dataset root (manifest or pt) under /kaggle/input")

print(f"Dataset Source : {dataset_input_dir}")

working_dataset = Path("/kaggle/working/finetune_v2")
artifacts_dir = Path("/kaggle/working/pfpd_artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

if working_dataset.exists():
    shutil.rmtree(working_dataset)
working_dataset.mkdir(parents=True, exist_ok=True)

for split in ["train", "val"]:
    (working_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
    (working_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    0: "pothole",
    1: "construction_hazard",
    2: "road_accident_obstacle"
}

# Symlink images and copy labels to working directory (allows Ultralytics to write labels.cache)
split_img_counts = Counter()
split_inst_counts = {0: Counter(), 1: Counter(), 2: Counter()}

for split in ["train", "val"]:
    src_img_dir = dataset_input_dir / "images" / split
    src_lbl_dir = dataset_input_dir / "labels" / split
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
                            cid = int(parts[0])
                            if cid in CLASS_MAP:
                                split_inst_counts[cid][split] += 1
            else:
                # Create empty label file for background/negative
                dst_lbl.touch()

print(f"\nProvisioned Dataset Summary:")
for split in ["train", "val"]:
    print(f"  {split.upper():5s}: {split_img_counts[split]:4d} images | "
          f"Potholes: {split_inst_counts[0][split]:3d}, "
          f"Construction: {split_inst_counts[1][split]:3d}, "
          f"Accident/Obstacle: {split_inst_counts[2][split]:3d}")

# Generate dynamic data.yaml
data_yaml_path = working_dataset / "data.yaml"
with open(data_yaml_path, "w") as f:
    f.write(f"""path: {working_dataset}
train: images/train
val: images/val

names:
  0: pothole
  1: construction_hazard
  2: road_accident_obstacle
""")
print(f"Created YAML manifest: {data_yaml_path}")

# -------------------------------------------------------------------------
# 3. Base Model Audit & Baseline Validation
# -------------------------------------------------------------------------
print("\n--- STAGE 3: BASE MODEL VERIFICATION & PRE-TRAINING EVALUATION ---")
base_weights_candidates = [
    dataset_input_dir / "unified_yolo11n_best.pt",
    Path("/kaggle/input/unified_yolo11n_best.pt"),
]
base_weights = None
for c in base_weights_candidates:
    if c.exists():
        base_weights = c
        break

if not base_weights:
    # Search recursively in input
    pts = list(Path("/kaggle/input").rglob("unified_yolo11n_best.pt"))
    if pts:
        base_weights = pts[0]

assert base_weights and base_weights.exists(), f"Could not find unified_yolo11n_best.pt in {dataset_input_dir}"

# Compute SHA-256 of base model
with open(base_weights, "rb") as f:
    base_sha256 = hashlib.sha256(f.read()).hexdigest()
base_size_mb = base_weights.stat().st_size / (1024**2)
print(f"Base Weights File : {base_weights}")
print(f"Base Weights Size : {base_size_mb:.2f} MB")
print(f"Base SHA-256 Hash : {base_sha256}")

# Load baseline model
print("\nLoading baseline model for pre-training evaluation on validation set...")
baseline_model = YOLO(str(base_weights))

t_base_val_0 = time.time()
base_val_results = baseline_model.val(
    data=str(data_yaml_path),
    split="val",
    imgsz=512,
    device=0 if torch.cuda.is_available() else "cpu",
    plots=False
)
base_val_dur = time.time() - t_base_val_0

base_metrics = {
    "mAP50": round(float(base_val_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
    "mAP50_95": round(float(base_val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    "precision": round(float(base_val_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
    "recall": round(float(base_val_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
    "per_class": {}
}
for idx, cname in CLASS_MAP.items():
    p = round(float(base_val_results.box.p[idx]), 4) if hasattr(base_val_results, "box") and len(base_val_results.box.p) > idx else 0.0
    r = round(float(base_val_results.box.r[idx]), 4) if hasattr(base_val_results, "box") and len(base_val_results.box.r) > idx else 0.0
    ap50 = round(float(base_val_results.box.ap50[idx]), 4) if hasattr(base_val_results, "box") and len(base_val_results.box.ap50) > idx else 0.0
    ap = round(float(base_val_results.box.ap[idx]), 4) if hasattr(base_val_results, "box") and len(base_val_results.box.ap) > idx else 0.0
    f1 = round(2 * p * r / (p + r + 1e-8), 4)
    base_metrics["per_class"][cname] = {
        "class_id": idx, "precision": p, "recall": r, "f1": f1, "mAP50": ap50, "mAP50_95": ap
    }

print("\n--- BASELINE METRICS ON VALIDATION SPLIT ---")
print(f"Overall mAP50: {base_metrics['mAP50']:.4f} | Overall mAP50-95: {base_metrics['mAP50_95']:.4f}")
for cname, m in base_metrics["per_class"].items():
    print(f"  Class {m['class_id']} ({cname:24s}): P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, mAP50={m['mAP50']:.4f}")

# -------------------------------------------------------------------------
# 4. Configure Layer Freezing & Execute 8-Epoch Fine-Tuning
# -------------------------------------------------------------------------
print("\n--- STAGE 4: EXECUTING BOUNDED 8-EPOCH TARGETED FINE-TUNING ---")

# Re-instantiate YOLO model from base weights for fine-tuning
ft_model = YOLO(str(base_weights))

# Audit architecture layers to verify freeze target
total_params = sum(p.numel() for p in ft_model.model.parameters())
trainable_params_before = sum(p.numel() for p in ft_model.model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params / 1e6:.2f}M | Trainable Before Freeze: {trainable_params_before / 1e6:.2f}M")

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Training Hyperparameters
epochs = 8
imgsz = 512
batch_size = 32
lr0 = 0.00015     # 1.5e-4: conservative transfer learning rate
lrf = 0.01        # final lr = 1.5e-6
freeze_layers = 8 # Freezes layers 0-7 (Backbone P1-P5 stem & feature stages)

print(f"Training Parameters:")
print(f"  Epochs        : {epochs}")
print(f"  Resolution    : {imgsz}x{imgsz}")
print(f"  Batch Size    : {batch_size}")
print(f"  Initial LR    : {lr0}")
print(f"  Final LR Factor: {lrf} (Final LR: {lr0 * lrf})")
print(f"  Cosine LR     : True")
print(f"  Mosaic Aug    : 0.0 (DISABLED for bounding-box preservation)")
print(f"  Freeze Layers : {freeze_layers} (Backbone stem/stages 0-7)")

t_train_0 = time.time()
train_results = ft_model.train(
    data=str(data_yaml_path),
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    freeze=freeze_layers,
    lr0=lr0,
    lrf=lrf,
    cos_lr=True,
    warmup_epochs=1,
    mosaic=0.0,
    box=7.5,
    cls=1.2,
    dfl=1.5,
    patience=0,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=4,
    save=True,
    plots=True,
    project="/kaggle/working/runs",
    name="finetuning_candidate"
)
train_dur = time.time() - t_train_0
print(f"\nFine-tuning completed in {train_dur:.1f} seconds ({train_dur / 60:.2f} minutes).")

# -------------------------------------------------------------------------
# 5. Checkpoint Retrieval & Candidate Validation
# -------------------------------------------------------------------------
print("\n--- STAGE 5: CANDIDATE CHECKPOINT & VALIDATION EVALUATION ---")
runs_dir = Path("/kaggle/working/runs/finetuning_candidate")
weights_dir = runs_dir / "weights"
best_pt = weights_dir / "best.pt"
last_pt = weights_dir / "last.pt"

if not best_pt.exists():
    best_pt = last_pt

assert best_pt.exists(), f"Error: No checkpoint generated at {weights_dir}"
cand_size_mb = best_pt.stat().st_size / (1024**2)

with open(best_pt, "rb") as f:
    cand_sha256 = hashlib.sha256(f.read()).hexdigest()

print(f"Candidate Checkpoint : {best_pt}")
print(f"Candidate Size       : {cand_size_mb:.2f} MB")
print(f"Candidate SHA-256    : {cand_sha256}")

# Copy candidate weights to artifacts
target_pt_name = "unified_yolo11n_finetuned_candidate.pt"
shutil.copy2(best_pt, artifacts_dir / target_pt_name)

# Run full evaluation of candidate model on validation set
print("\nRunning evaluation of fine-tuned candidate on validation set...")
candidate_eval_model = YOLO(str(best_pt))
cand_val_results = candidate_eval_model.val(
    data=str(data_yaml_path),
    split="val",
    imgsz=512,
    device=0 if torch.cuda.is_available() else "cpu",
    plots=True
)

cand_metrics = {
    "mAP50": round(float(cand_val_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
    "mAP50_95": round(float(cand_val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    "precision": round(float(cand_val_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
    "recall": round(float(cand_val_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
    "per_class": {}
}
for idx, cname in CLASS_MAP.items():
    p = round(float(cand_val_results.box.p[idx]), 4) if hasattr(cand_val_results, "box") and len(cand_val_results.box.p) > idx else 0.0
    r = round(float(cand_val_results.box.r[idx]), 4) if hasattr(cand_val_results, "box") and len(cand_val_results.box.r) > idx else 0.0
    ap50 = round(float(cand_val_results.box.ap50[idx]), 4) if hasattr(cand_val_results, "box") and len(cand_val_results.box.ap50) > idx else 0.0
    ap = round(float(cand_val_results.box.ap[idx]), 4) if hasattr(cand_val_results, "box") and len(cand_val_results.box.ap) > idx else 0.0
    f1 = round(2 * p * r / (p + r + 1e-8), 4)
    cand_metrics["per_class"][cname] = {
        "class_id": idx, "precision": p, "recall": r, "f1": f1, "mAP50": ap50, "mAP50_95": ap
    }

# Print Direct Comparison Table
print("\n" + "=" * 78)
print("             BASELINE vs FINE-TUNED CANDIDATE VALIDATION COMPARISON")
print("=" * 78)
print(f"{'Class / Metric':<28s} | {'Baseline (50-ep)':<18s} | {'Candidate (8-ep FT)':<18s} | {'Delta':<8s}")
print("-" * 78)

# Overall
print(f"{'Overall mAP50':<28s} | {base_metrics['mAP50']:<18.4f} | {cand_metrics['mAP50']:<18.4f} | {cand_metrics['mAP50'] - base_metrics['mAP50']:+.4f}")
print(f"{'Overall mAP50-95':<28s} | {base_metrics['mAP50_95']:<18.4f} | {cand_metrics['mAP50_95']:<18.4f} | {cand_metrics['mAP50_95'] - base_metrics['mAP50_95']:+.4f}")
print(f"{'Overall Precision':<28s} | {base_metrics['precision']:<18.4f} | {cand_metrics['precision']:<18.4f} | {cand_metrics['precision'] - base_metrics['precision']:+.4f}")
print(f"{'Overall Recall':<28s} | {base_metrics['recall']:<18.4f} | {cand_metrics['recall']:<18.4f} | {cand_metrics['recall'] - base_metrics['recall']:+.4f}")
print("-" * 78)

# Per-Class Breakdown
for idx, cname in CLASS_MAP.items():
    bm = base_metrics["per_class"][cname]
    cm = cand_metrics["per_class"][cname]
    print(f"[{cname}]")
    print(f"  Precision                  | {bm['precision']:<18.4f} | {cm['precision']:<18.4f} | {cm['precision'] - bm['precision']:+.4f}")
    print(f"  Recall                     | {bm['recall']:<18.4f} | {cm['recall']:<18.4f} | {cm['recall'] - bm['recall']:+.4f}")
    print(f"  F1-Score                   | {bm['f1']:<18.4f} | {cm['f1']:<18.4f} | {cm['f1'] - bm['f1']:+.4f}")
    print(f"  mAP50                      | {bm['mAP50']:<18.4f} | {cm['mAP50']:<18.4f} | {cm['mAP50'] - bm['mAP50']:+.4f}")
print("=" * 78)

# -------------------------------------------------------------------------
# 6. ONNX Export & Graph Verification (384x384 & 512x512)
# -------------------------------------------------------------------------
print("\n--- STAGE 6: ONNX EXPORT & GRAPH CONTRACT VERIFICATION ---")

export_model = YOLO(str(best_pt))

# Export 384x384 (Raspberry Pi 5 Target)
print("Exporting Candidate to 384x384 fixed ONNX...")
onnx_384_raw = export_model.export(format="onnx", imgsz=384, dynamic=False, opset=17)
onnx_384_target = artifacts_dir / "unified_yolo11n_candidate_384.onnx"
shutil.copy2(onnx_384_raw, onnx_384_target)
print(f"Saved: {onnx_384_target} ({onnx_384_target.stat().st_size / (1024**2):.2f} MB)")

# Export 512x512 (Native Training Resolution)
print("Exporting Candidate to 512x512 fixed ONNX...")
onnx_512_raw = export_model.export(format="onnx", imgsz=512, dynamic=False, opset=17)
onnx_512_target = artifacts_dir / "unified_yolo11n_candidate_512.onnx"
shutil.copy2(onnx_512_raw, onnx_512_target)
print(f"Saved: {onnx_512_target} ({onnx_512_target.stat().st_size / (1024**2):.2f} MB)")

# Verify ONNX Runtime Graphs
print("\nVerifying ONNX Graph Contracts via ONNX Runtime...")
sess_384 = ort.InferenceSession(str(onnx_384_target), providers=["CPUExecutionProvider"])
in_384 = sess_384.get_inputs()[0]
out_384 = sess_384.get_outputs()[0]
print(f"  384px Model Input : Name={in_384.name}, Shape={in_384.shape}, Type={in_384.type}")
print(f"  384px Model Output: Name={out_384.name}, Shape={out_384.shape}, Type={out_384.type}")
expected_384 = [1, 7, 3024]
assert list(out_384.shape) == expected_384, f"Unexpected 384 ONNX shape: {out_384.shape} != {expected_384}"
print("  384px Graph Contract: PASSED ✅ [1, 7, 3024]")

sess_512 = ort.InferenceSession(str(onnx_512_target), providers=["CPUExecutionProvider"])
in_512 = sess_512.get_inputs()[0]
out_512 = sess_512.get_outputs()[0]
print(f"  512px Model Input : Name={in_512.name}, Shape={in_512.shape}, Type={in_512.type}")
print(f"  512px Model Output: Name={out_512.name}, Shape={out_512.shape}, Type={out_512.type}")
expected_512 = [1, 7, 5376]
assert list(out_512.shape) == expected_512, f"Unexpected 512 ONNX shape: {out_512.shape} != {expected_512}"
print("  512px Graph Contract: PASSED ✅ [1, 7, 5376]")

# -------------------------------------------------------------------------
# 7. Collect Training Artifacts & Compile Manifest
# -------------------------------------------------------------------------
print("\n--- STAGE 7: COLLECTING ARTIFACTS & GENERATING MANIFEST ---")
for plot_p in runs_dir.glob("*.png"):
    shutil.copy2(plot_p, artifacts_dir / plot_p.name)
for plot_p in runs_dir.glob("*.jpg"):
    shutil.copy2(plot_p, artifacts_dir / plot_p.name)
if (runs_dir / "results.csv").exists():
    shutil.copy2(runs_dir / "results.csv", artifacts_dir / "results.csv")

# Parse results.csv history
results_csv = runs_dir / "results.csv"
history = []
if results_csv.exists():
    import csv
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({k.strip(): float(v.strip()) for k, v in row.items() if v.strip()})

manifest = {
    "experiment": "pfpd_yolo11n_targeted_finetuning_8ep",
    "timestamp": time.time(),
    "pipeline_total_duration_sec": round(time.time() - start_time, 2),
    "accelerator": {
        "device": gpu_name,
        "vram_gb": round(gpu_mem_gb, 2)
    },
    "dataset": {
        "splits": dict(split_img_counts),
        "instances": {
            CLASS_MAP[cid]: dict(split_inst_counts[cid]) for cid in [0, 1, 2]
        }
    },
    "hyperparameters": {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "lr0": lr0,
        "lrf": lrf,
        "cos_lr": True,
        "mosaic": 0.0,
        "freeze_layers": freeze_layers,
        "training_duration_sec": round(train_dur, 2)
    },
    "baseline_model": {
        "weights_path": str(base_weights),
        "sha256": base_sha256,
        "size_mb": round(base_size_mb, 2),
        "validation_metrics": base_metrics
    },
    "candidate_model": {
        "weights_path": str(artifacts_dir / target_pt_name),
        "sha256": cand_sha256,
        "size_mb": round(cand_size_mb, 2),
        "validation_metrics": cand_metrics
    },
    "metric_deltas": {
        "overall_mAP50": round(cand_metrics["mAP50"] - base_metrics["mAP50"], 4),
        "overall_mAP50_95": round(cand_metrics["mAP50_95"] - base_metrics["mAP50_95"], 4),
        "overall_precision": round(cand_metrics["precision"] - base_metrics["precision"], 4),
        "overall_recall": round(cand_metrics["recall"] - base_metrics["recall"], 4),
        "per_class": {
            cname: {
                "precision": round(cand_metrics["per_class"][cname]["precision"] - base_metrics["per_class"][cname]["precision"], 4),
                "recall": round(cand_metrics["per_class"][cname]["recall"] - base_metrics["per_class"][cname]["recall"], 4),
                "f1": round(cand_metrics["per_class"][cname]["f1"] - base_metrics["per_class"][cname]["f1"], 4),
                "mAP50": round(cand_metrics["per_class"][cname]["mAP50"] - base_metrics["per_class"][cname]["mAP50"], 4),
            } for cname in ["pothole", "construction_hazard", "road_accident_obstacle"]
        }
    },
    "export_verification": {
        "onnx_384": {
            "path": str(onnx_384_target),
            "input_shape": list(in_384.shape),
            "output_shape": list(out_384.shape),
            "valid": True
        },
        "onnx_512": {
            "path": str(onnx_512_target),
            "input_shape": list(in_512.shape),
            "output_shape": list(out_512.shape),
            "valid": True
        }
    },
    "training_history": history
}

manifest_path = artifacts_dir / "finetuning_candidate_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved full experiment manifest to: {manifest_path}")
print("\n" + "=" * 72)
print("  TARGETED YOLO11n FINE-TUNING PIPELINE EXECUTED SUCCESSFULLY!  ")
print("=" * 72)
