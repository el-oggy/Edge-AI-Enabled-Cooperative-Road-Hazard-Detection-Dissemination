"""
PFPD Physical Pi 5 Acceptance Validation Script
==============================================
Executes genuine end-to-end perception across 30 real benchmark images (10 per hazard class)
using the authentic specialist models with safe 2-thread pacing and thermal monitoring.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pfpd_perception.end_to_end_pipeline import PFPDEndToEndPipeline

def get_vcgencmd(cmd):
    try:
        out = subprocess.check_output(["vcgencmd", cmd]).decode().strip()
        return out
    except Exception as e:
        return f"unavailable ({e})"

def run_physical_acceptance_benchmark():
    print("==================================================================")
    print("  PFPD PHYSICAL PI 5 ACCEPTANCE VALIDATION (PACED RUN - 2 THREADS)")
    print("==================================================================")

    temp_start = get_vcgencmd("measure_temp")
    throttled_start = get_vcgencmd("get_throttled")
    print(f"Starting System State: Temp = {temp_start} | Throttled = {throttled_start}")

    # Gather benchmark images recursively
    base_data = Path("data/benchmark_150")
    domains = {
        "potholes": list((base_data / "potholes").rglob("*.jpg")) + list((base_data / "potholes").rglob("*.png")),
        "construction": list((base_data / "construction").rglob("*.jpg")) + list((base_data / "construction").rglob("*.png")),
        "accidents_obstacles": list((base_data / "accidents_obstacles").rglob("*.jpg")) + list((base_data / "accidents_obstacles").rglob("*.png"))
    }

    selected_images = []
    for d, img_list in domains.items():
        if not img_list:
            print(f"Warning: No images found in {base_data / d}")
            continue
        # Take up to 10 images per category
        selected = sorted(img_list)[:10]
        for img_p in selected:
            selected_images.append((d, img_p))

    print(f"Total benchmark images selected: {len(selected_images)} (10 per hazard category)")
    if not selected_images:
        print("Error: No images found to benchmark.")
        return

    # Initialize Pipeline (safe 2 threads)
    print("\nInitializing PFPDEndToEndPipeline (authentic 3-specialist ONNX INT8, 2 threads)...")
    t0_init = time.perf_counter()
    pipeline = PFPDEndToEndPipeline(rsu_id="RSU_PI5_ACCEPTANCE", threads=2)
    init_ms = (time.perf_counter() - t0_init) * 1000.0
    print(f"Pipeline initialized in {init_ms:.1f} ms.")

    frame_records = []
    
    print("\nExecuting Paced End-to-End Processing...")
    for idx, (cat, img_path) in enumerate(selected_images, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [Frame {idx:02d}] Failed to load {img_path}")
            continue

        res = pipeline.process_frame(img_bgr)
        lat = res["total_latency_ms"]
        raw = res["raw_detections"]
        conf = res["confirmed_hazards"]
        denm_cnt = len(res["denm_messages"])

        print(f"  [Frame {idx:02d}/{len(selected_images):02d}] Cat: {cat[:12]:12s} | Lat: {lat:7.1f} ms | Raw: {raw} | Confirmed: {conf} | DENM: {denm_cnt}")

        frame_records.append({
            "frame_idx": idx,
            "category": cat,
            "file": img_path.name,
            "latency_ms": lat,
            "raw_detections": raw,
            "confirmed_hazards": conf,
            "timings_ms": res["timings_ms"],
            "denm_messages": res["denm_messages"]
        })

        # Pacing sleep to prevent core heat buildup
        time.sleep(0.05)

    temp_end = get_vcgencmd("measure_temp")
    throttled_end = get_vcgencmd("get_throttled")
    print(f"\nEnding System State: Temp = {temp_end} | Throttled = {throttled_end}")

    # Compute Statistics
    lats = [r["latency_ms"] for r in frame_records]
    p50 = float(np.percentile(lats, 50))
    p95 = float(np.percentile(lats, 95))
    mean_lat = float(np.mean(lats))
    min_lat = float(np.min(lats))
    max_lat = float(np.max(lats))
    pure_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    telemetry = pipeline.get_summary_telemetry()

    summary = {
        "timestamp": time.time(),
        "hardware": "Raspberry Pi 5 Model B Rev 1.0 (8GB RAM, Quad Cortex-A76 @ 2.4 GHz)",
        "configuration": {
            "execution_mode": "Sequential Authentic 3-Specialist Pipeline",
            "models": [
                "quantization/pothole_yolo11s_int8.onnx",
                "quantization/construction_yolo11s_int8.onnx",
                "quantization/accident_obstacle_yolo11s_int8.onnx"
            ],
            "input_resolution": [768, 768],
            "threads_per_session": 2,
            "runtime": "ONNX Runtime (CPUExecutionProvider)",
            "safety_pacing_delay_ms": 50
        },
        "system_health": {
            "temp_start": temp_start,
            "temp_end": temp_end,
            "throttled_start": throttled_start,
            "throttled_end": throttled_end
        },
        "metrics": {
            "total_frames_evaluated": len(frame_records),
            "mean_latency_ms": round(mean_lat, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "pure_inference_fps": round(pure_fps, 2),
            "total_raw_detections": sum(r["raw_detections"] for r in frame_records),
            "total_confirmed_events": telemetry.get("events_disseminated", 0),
            "resident_memory_rss_mb": telemetry.get("rss_mb", 0.0)
        },
        "frames": frame_records
    }

    out_path = Path("results/step4_physical_acceptance_e2e.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAudit results successfully saved to {out_path}")
    print("------------------------------------------------------------------")
    print(f"  Mean Latency : {mean_lat:.1f} ms  (p50: {p50:.1f} ms, p95: {p95:.1f} ms)")
    print(f"  Sustained FPS: {pure_fps:.2f} FPS (3-specialist full pass)")
    print(f"  RSS Memory   : {telemetry.get('rss_mb', 0.0)} MB")
    print(f"  Core Temp    : {temp_start} -> {temp_end}")
    print(f"  Throttling   : {throttled_end}")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    run_physical_acceptance_benchmark()
