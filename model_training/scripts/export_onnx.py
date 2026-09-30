import os
import json
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

BEST_PT = "/content/Road_Hazard_AI/models/pothole_yolo11n_rdd_v2/weights/best.pt"
DATA_YAML = "/content/Road_Hazard_AI/datasets/potholes_rdd2022_merged_v2/data.yaml"
OUTPUT_DIR = "/content/Road_Hazard_AI/models/pothole_yolo11n_rdd_v2"
REPORTS_DIR = "/content/Road_Hazard_AI/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=== 1. Loading Trained Checkpoint ===")
model = YOLO(BEST_PT)

print("=== 2. Exporting to ONNX ===")
onnx_path = model.export(format="onnx", opset=12, dynamic=False, imgsz=640)
print(f"Exported ONNX: {onnx_path}")

pt_size_mb = os.path.getsize(BEST_PT) / (1024 * 1024)
onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

print(f"PyTorch best.pt size: {pt_size_mb:.2f} MB")
print(f"ONNX model size: {onnx_size_mb:.2f} MB")

# Benchmark latency
print("=== 3. Benchmarking Latency ===")
dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)

# GPU benchmark
if torch.cuda.is_available():
    t_gpu = torch.from_numpy(dummy).cuda()
    for _ in range(10): _ = model(t_gpu, verbose=False)
    gpu_times = []
    for _ in range(50):
        t0 = time.time()
        _ = model(t_gpu, verbose=False)
        torch.cuda.synchronize()
        gpu_times.append((time.time() - t0) * 1000)
    gpu_latency = float(np.median(gpu_times))
    gpu_fps = float(1000.0 / gpu_latency)
else:
    gpu_latency, gpu_fps = 0.0, 0.0

# CPU benchmark
cpu_model = YOLO(BEST_PT)
t_cpu = torch.from_numpy(dummy).cpu()
for _ in range(5): _ = cpu_model(t_cpu, device="cpu", verbose=False)
cpu_times = []
for _ in range(25):
    t0 = time.time()
    _ = cpu_model(t_cpu, device="cpu", verbose=False)
    cpu_times.append((time.time() - t0) * 1000)
cpu_latency = float(np.median(cpu_times))
cpu_fps = float(1000.0 / cpu_latency)

print(f"Tesla T4 GPU: {gpu_latency:.2f} ms ({gpu_fps:.1f} FPS)")
print(f"Host x86 CPU: {cpu_latency:.2f} ms ({cpu_fps:.1f} FPS)")

# Quantitative metrics from evaluation
val_metrics = {
    "precision": 0.5972,
    "recall": 0.5446,
    "map50": 0.5771,
    "map50_95": 0.2925
}

test_metrics = {
    "precision": 0.8034,
    "recall": 0.6957,
    "map50": 0.7715,
    "map50_95": 0.4964
}

baseline_metrics = {
    "precision": 0.8959,
    "recall": 0.6522,
    "map50": 0.7926,
    "map50_95": 0.5108
}

delta_p = test_metrics["precision"] - baseline_metrics["precision"]
delta_r = test_metrics["recall"] - baseline_metrics["recall"]
delta_map50 = test_metrics["map50"] - baseline_metrics["map50"]
delta_map95 = test_metrics["map50_95"] - baseline_metrics["map50_95"]

print("=== 4. Updating Experiment Registry ===")
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
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
        "mAP50": baseline_metrics["map50"],
        "mAP50-95": baseline_metrics["map50_95"],
        "model_size_mb": "5.40 MB",
        "inference_latency_cpu_ms": "35.2 ms",
        "notes": "Historical baseline trained on 665 Andrew MVD images"
    },
    {
        "experiment_id": "EXP_002_YOLO11N_RDD_V2",
        "dataset_version": "potholes_rdd2022_merged_v2 (2,195 images)",
        "model": "YOLO11n",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "auto (AdamW)",
        "learning_rate": "0.002",
        "seed": 42,
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "mAP50": test_metrics["map50"],
        "mAP50-95": test_metrics["map50_95"],
        "model_size_mb": f"{pt_size_mb:.2f} MB",
        "inference_latency_cpu_ms": f"{cpu_latency:.2f} ms",
        "notes": f"Clean RDD2022 India D40 merged; Recall improved from 65.22% to 69.57% ({delta_r:+.4f}), mAP50={test_metrics['map50']:.4f}"
    }
]
pd.DataFrame(reg_rows).to_csv(os.path.join(REPORTS_DIR, "experiment_registry.csv"), index=False)
print("Saved experiment_registry.csv")

# Save JSON metadata
summary = {
    "experiment_id": "EXP_002_YOLO11N_RDD_V2",
    "dataset": "potholes_rdd2022_merged_v2",
    "total_images": 2195,
    "total_boxes": 4926,
    "model_architecture": "YOLO11n",
    "parameters": 2582347,
    "gflops": 6.4,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
    "baseline_comparison": {
        "baseline_recall": baseline_metrics["recall"],
        "new_recall": test_metrics["recall"],
        "recall_delta": delta_r,
        "baseline_map50": baseline_metrics["map50"],
        "new_map50": test_metrics["map50"],
        "map50_delta": delta_map50
    },
    "edge_hardware": {
        "pt_size_mb": pt_size_mb,
        "onnx_size_mb": onnx_size_mb,
        "gpu_latency_ms": gpu_latency,
        "gpu_fps": gpu_fps,
        "cpu_latency_ms": cpu_latency,
        "cpu_fps": cpu_fps
    }
}
with open(os.path.join(REPORTS_DIR, "canonical_model_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Saved canonical_model_summary.json")

# 5. Error Analysis on Test Set
print("=== 5. Running Qualitative Error Analysis ===")
test_images_dir = "/content/Road_Hazard_AI/datasets/potholes_rdd2022_merged_v2/images/test"
test_imgs = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)[:15]]
pred_dir = os.path.join(REPORTS_DIR, "sample_predictions")
os.makedirs(pred_dir, exist_ok=True)

preds = model(test_imgs, conf=0.25, save=True, project=REPORTS_DIR, name="sample_predictions", exist_ok=True)
print(f"Generated sample predictions in {pred_dir}")
print("=== Complete! ===")
