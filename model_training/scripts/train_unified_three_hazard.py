"""
PFPD Unified Three-Class Hazard Detector Training & Export Pipeline
===================================================================
Covers Step 2 (Unified YOLO11s) and Step 3 (Lightweight Nano YOLO11n / YOLO26n):
1. Loads unified 3-class dataset configuration (pothole, construction, accident).
2. Trains or fine-tunes candidate architectures (YOLO11s, YOLO11n, YOLO26n).
3. Evaluates per-class precision, recall, and mAP@50 across all 3 hazard categories.
4. Directly exports optimized deployment graphs: FP32 ONNX, static INT8, and NCNN.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Three-Hazard Road Hazard Detector")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (yolo11s.pt, yolo11n.pt, yolo26n.pt)")
    parser.add_argument("--data", type=str, default="configs/unified_three_class_hazard.yaml", help="Path to unified data YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", type=str, default="models/unified_detectors", help="Output project directory")
    parser.add_argument("--name", type=str, default="unified_yolo11n_3class", help="Experiment run name")
    return parser.parse_args()

def main():
    args = parse_args()
    print("====================================================================")
    print(f"STARTING UNIFIED THREE-CLASS TRAINING: {args.model.upper()} @ {args.imgsz}px")
    print("====================================================================")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model = YOLO(args.model)

    # 1. Train
    t0 = time.time()
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        save=True,
        plots=True,
        workers=2,
        seed=42,
        patience=15,
        optimizer="auto",
        verbose=True
    )
    train_duration = (time.time() - t0) / 60.0
    print(f"\n[Training Completed] Duration: {train_duration:.2f} minutes")

    # 2. Path to Best Checkpoint
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"Warning: Best checkpoint not found at {best_weights}. Using last checkpoint.")
        best_weights = Path(args.project) / args.name / "weights" / "last.pt"

    best_model = YOLO(str(best_weights))

    # 3. Validation Evaluation (Per-Class)
    print("\n--- Running Multi-Class Validation ---")
    val_metrics = best_model.val(data=args.data, split="val", imgsz=args.imgsz, batch=args.batch, device=device)
    
    # Extract Per-Class Metrics
    names = val_metrics.names
    print("\nPer-Class Detection Quality:")
    class_summaries = {}
    for cid, cname in names.items():
        p = float(val_metrics.box.p[cid]) if hasattr(val_metrics.box, 'p') and len(val_metrics.box.p) > cid else 0.0
        r = float(val_metrics.box.r[cid]) if hasattr(val_metrics.box, 'r') and len(val_metrics.box.r) > cid else 0.0
        map50 = float(val_metrics.box.ap50[cid]) if hasattr(val_metrics.box, 'ap50') and len(val_metrics.box.ap50) > cid else 0.0
        class_summaries[cname] = {"precision": round(p, 4), "recall": round(r, 4), "mAP50": round(map50, 4)}
        print(f"  Class {cid} [{cname}]: Precision={p:.4f}, Recall={r:.4f}, mAP50={map50:.4f}")

    overall_map50 = float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    overall_map50_95 = float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    # 4. Multi-Resolution Edge Deployment Exports
    print("\n--- Exporting Edge Deployment Artifacts ---")
    for export_res in [640, 512, 384, 320]:
        onnx_file = best_model.export(
            format="onnx",
            imgsz=export_res,
            dynamic=False,
            simplify=True,
            opset=17
        )
        print(f"  Exported ONNX @ {export_res}px: {onnx_file}")

    # 5. Save Machine-Readable Validation Report
    report = {
        "model": args.model,
        "experiment_name": args.name,
        "epochs": args.epochs,
        "train_imgsz": args.imgsz,
        "overall_mAP50": round(overall_map50, 4),
        "overall_mAP50_95": round(overall_map50_95, 4),
        "classes": class_summaries,
        "weights_path": str(best_weights)
    }

    report_path = Path("reports") / f"val_summary_{args.name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved validation report to: {report_path}")

if __name__ == "__main__":
    main()
