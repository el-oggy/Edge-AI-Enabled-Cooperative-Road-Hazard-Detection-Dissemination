import os
import json
import time
import shutil
import torch
import pandas as pd
from ultralytics import YOLO

print("==================================================")
print("STARTING YOLO11n ROAD HAZARD TRAINING & EVALUATION")
print("==================================================")

DATA_YAML = "/content/Road_Hazard_AI/datasets/potholes_rdd2022_merged_v2/data.yaml"
PROJECT_DIR = "/content/Road_Hazard_AI/models"
RUN_NAME = "pothole_yolo11n_rdd_v2"
OUTPUT_RUN_DIR = os.path.join(PROJECT_DIR, RUN_NAME)
REPORTS_DIR = "/content/Road_Hazard_AI/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Initialize YOLO11n model
print("\n--- 1. Loading YOLO11n ---")
model = YOLO("yolo11n.pt")
print(f"Loaded YOLO11n architecture. Device available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. Train on the merged dataset
print("\n--- 2. Starting Training (50 Epochs, batch=16, imgsz=640, seed=42) ---")
start_time = time.time()
train_results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else "cpu",
    project=PROJECT_DIR,
    name=RUN_NAME,
    save=True,
    plots=True,
    workers=2,
    seed=42,
    patience=15,
    optimizer="auto",
    verbose=True
)
training_duration = time.time() - start_time
print(f"Training completed in {training_duration/60:.2f} minutes.")

# 3. Path to best checkpoint
best_pt = os.path.join(OUTPUT_RUN_DIR, "weights", "best.pt")
print(f"\n--- 3. Canonical Best Checkpoint: {best_pt} ---")
best_model = YOLO(best_pt)

# 4. Validation evaluation
print("\n--- 4. Evaluating on Validation Set ---")
val_results = best_model.val(data=DATA_YAML, split="val", imgsz=640, batch=16, device=0 if torch.cuda.is_available() else "cpu")
val_precision = float(val_results.results_dict.get("metrics/precision(B)", 0))
val_recall = float(val_results.results_dict.get("metrics/recall(B)", 0))
val_map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0))
val_map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0))
print(f"Validation: Precision={val_precision:.4f}, Recall={val_recall:.4f}, mAP50={val_map50:.4f}, mAP50-95={val_map50_95:.4f}")

# 5. Held-Out Test evaluation (Frozen 67 Andrew MVD images)
print("\n--- 5. Evaluating on Held-Out Frozen Test Set (67 images, 161 boxes) ---")
test_results = best_model.val(data=DATA_YAML, split="test", imgsz=640, batch=16, device=0 if torch.cuda.is_available() else "cpu")
test_precision = float(test_results.results_dict.get("metrics/precision(B)", 0))
test_recall = float(test_results.results_dict.get("metrics/recall(B)", 0))
test_map50 = float(test_results.results_dict.get("metrics/mAP50(B)", 0))
test_map50_95 = float(test_results.results_dict.get("metrics/mAP50-95(B)", 0))
print(f"Held-Out Test: Precision={test_precision:.4f}, Recall={test_recall:.4f}, mAP50={test_map50:.4f}, mAP50-95={test_map50_95:.4f}")

# 6. Export to ONNX for edge deployment (Jetson Nano / Raspberry Pi 3 / PYNQ-Z2)
print("\n--- 6. Exporting to ONNX (opset=12, dynamic=False, imgsz=640) ---")
onnx_path = best_model.export(format="onnx", opset=12, dynamic=False, imgsz=640)
print(f"ONNX Model Exported: {onnx_path}")

# Measure model file sizes
pt_size_mb = os.path.getsize(best_pt) / (1024 * 1024)
onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"PyTorch Best Checkpoint Size: {pt_size_mb:.2f} MB")
print(f"ONNX Export Size: {onnx_size_mb:.2f} MB")

# Benchmark latency
print("\n--- 7. Benchmarking Latency ---")
import numpy as np
dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
# GPU warmup & benchmark
if torch.cuda.is_available():
    t_dummy = torch.from_numpy(dummy_input).cuda()
    for _ in range(10): _ = best_model(t_dummy, verbose=False)
    latencies = []
    for _ in range(50):
        t0 = time.time()
        _ = best_model(t_dummy, verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)
    gpu_latency_ms = np.median(latencies)
    gpu_fps = 1000.0 / gpu_latency_ms
else:
    gpu_latency_ms, gpu_fps = 0.0, 0.0

# CPU benchmark
cpu_model = YOLO(best_pt)
t_cpu = torch.from_numpy(dummy_input).cpu()
for _ in range(5): _ = cpu_model(t_cpu, device="cpu", verbose=False)
cpu_latencies = []
for _ in range(20):
    t0 = time.time()
    _ = cpu_model(t_cpu, device="cpu", verbose=False)
    cpu_latencies.append((time.time() - t0) * 1000)
cpu_latency_ms = np.median(cpu_latencies)
cpu_fps = 1000.0 / cpu_latency_ms

print(f"GPU Latency (Tesla T4): {gpu_latency_ms:.2f} ms ({gpu_fps:.1f} FPS)")
print(f"CPU Latency (Host x86): {cpu_latency_ms:.2f} ms ({cpu_fps:.1f} FPS)")

# 8. Historical Baseline comparison
baseline_metrics = {
    "test_precision": 0.8959,
    "test_recall": 0.6522,
    "test_map50": 0.7926,
    "test_map50_95": 0.5108
}

delta_p = test_precision - baseline_metrics["test_precision"]
delta_r = test_recall - baseline_metrics["test_recall"]
delta_map50 = test_map50 - baseline_metrics["test_map50"]
delta_map95 = test_map50_95 - baseline_metrics["test_map50_95"]

# 9. Save comprehensive experiment metadata and registry
experiment_data = {
    "experiment_id": "EXP_002_YOLO11N_RDD_V2",
    "dataset_version": "potholes_rdd2022_merged_v2",
    "model": "YOLO11n",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "auto",
    "seed": 42,
    "val_precision": val_precision,
    "val_recall": val_recall,
    "val_map50": val_map50,
    "val_map50_95": val_map50_95,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_map50": test_map50,
    "test_map50_95": test_map50_95,
    "baseline_test_map50": baseline_metrics["test_map50"],
    "delta_map50": delta_map50,
    "baseline_test_recall": baseline_metrics["test_recall"],
    "delta_recall": delta_r,
    "weights_path": best_pt,
    "onnx_path": onnx_path,
    "pt_size_mb": pt_size_mb,
    "onnx_size_mb": onnx_size_mb,
    "gpu_latency_ms": gpu_latency_ms,
    "gpu_fps": gpu_fps,
    "cpu_latency_ms": cpu_latency_ms,
    "cpu_fps": cpu_fps,
    "training_time_min": training_duration / 60
}

with open(os.path.join(REPORTS_DIR, "experiment_results.json"), "w") as f:
    json.dump(experiment_data, f, indent=2)

# Update / write experiment_registry.csv
reg_path = os.path.join(REPORTS_DIR, "experiment_registry.csv")
reg_rows = [
    {
        "experiment_id": "EXP_001_BASELINE",
        "dataset_version": "potholes_final_v1 (Andrew MVD 665)",
        "model": "YOLO11n",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "auto",
        "learning_rate": "auto",
        "seed": 42,
        "precision": baseline_metrics["test_precision"],
        "recall": baseline_metrics["test_recall"],
        "mAP50": baseline_metrics["test_map50"],
        "mAP50-95": baseline_metrics["test_map50_95"],
        "model_size_mb": "6.5 MB",
        "notes": "Historical Baseline on 665-image Andrew MVD dataset"
    },
    {
        "experiment_id": "EXP_002_YOLO11N_RDD_V2",
        "dataset_version": "potholes_rdd2022_merged_v2",
        "model": "YOLO11n",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "auto",
        "learning_rate": "auto",
        "seed": 42,
        "precision": test_precision,
        "recall": test_recall,
        "mAP50": test_map50,
        "mAP50-95": test_map50_95,
        "model_size_mb": f"{pt_size_mb:.2f} MB",
        "notes": f"Clean RDD2022 India D40 merged; recall delta: {delta_r:+.4f}, mAP50 delta: {delta_map50:+.4f}"
    }
]
pd.DataFrame(reg_rows).to_csv(reg_path, index=False)
print(f"Updated {reg_path}")

print("\n==================================================")
print("TRAINING, EVALUATION & EXPORT COMPLETE!")
print(f"Test Recall: {test_recall:.4f} (Baseline: {baseline_metrics['test_recall']:.4f} -> Delta: {delta_r:+.4f})")
print(f"Test mAP50:  {test_map50:.4f} (Baseline: {baseline_metrics['test_map50']:.4f} -> Delta: {delta_map50:+.4f})")
print(f"Test mAP50-95: {test_map50_95:.4f} (Baseline: {baseline_metrics['test_map50_95']:.4f} -> Delta: {delta_map95:+.4f})")
print("==================================================")
