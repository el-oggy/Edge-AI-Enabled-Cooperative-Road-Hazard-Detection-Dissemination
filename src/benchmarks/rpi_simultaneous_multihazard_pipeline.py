"""
RSU Multi-Hazard Simultaneous Perception & LoRa V2X Transmission Pipeline
========================================================================
Target: Raspberry Pi 5 RSU (Quad Cortex-A76 @ 2.4GHz, LoRa SX1262 HAT)
Evaluates all 3 INT8 quantized models simultaneously across all 3 hazard streams:
1. Pothole Model on test_user_pothole.png
2. Construction Model on test_user_construction.png
3. Accident Model on test_user_accident.png
Generates unified ETSI EN 302 637-3 DENM JSON beacon for LoRaWAN / V2X transmission.
"""

import os
import time
import json
from pathlib import Path
import numpy as np
import cv2
import psutil
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
QUANT_DIR = ROOT_DIR / "quantization"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HAZARDS = [
    {
        "id": "pothole",
        "title": "Pothole Detector",
        "model": QUANT_DIR / "pothole_yolo11s_int8.onnx",
        "image": DATA_DIR / "test_user_pothole.png",
        "hazard_type": "POTHOLE",
        "cause_code": 97,
        "severity": "MEDIUM",
        "advisory": "REDUCE_SPEED",
        "speed_kmh": 25,
        "conf_thresh": 0.25
    },
    {
        "id": "construction",
        "title": "Construction Zone Detector",
        "model": QUANT_DIR / "construction_yolo11s_int8.onnx",
        "image": DATA_DIR / "test_user_construction.png",
        "hazard_type": "CONSTRUCTION_HAZARD",
        "cause_code": 96,
        "severity": "HIGH",
        "advisory": "MERGE_LANE",
        "speed_kmh": 20,
        "conf_thresh": 0.25
    },
    {
        "id": "accident",
        "title": "Accident & Debris Detector",
        "model": QUANT_DIR / "accident_obstacle_yolo11s_int8.onnx",
        "image": DATA_DIR / "test_user_accident.png",
        "hazard_type": "ROAD_ACCIDENT_OBSTACLE",
        "cause_code": 95,
        "severity": "CRITICAL",
        "advisory": "MANDATORY_LANE_CHANGE",
        "speed_kmh": 0,
        "conf_thresh": 0.35
    }
]

def preprocess(img_path, input_size=768):
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        raise ValueError(f"Could not load image: {img_path}")
    h0, w0 = img0.shape[:2]
    scale = min(input_size / h0, input_size / w0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    dw, dh = (input_size - nw) / 2, (input_size - nh) / 2
    resized = cv2.resize(img0, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, (h0, w0), (scale, dw, dh), img0

def decode_detections(preds, orig_shape, ratio_pad, hazard_info, conf_thresh=0.25, iou_thresh=0.45):
    predictions = np.squeeze(preds).T
    h0, w0 = orig_shape
    scale, dw, dh = ratio_pad
    boxes = []
    scores = []
    for row in predictions:
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        score = float(row[4])
        if score >= conf_thresh:
            xmin = (cx - w / 2 - dw) / scale
            ymin = (cy - h / 2 - dh) / scale
            boxes.append([int(xmin), int(ymin), int(w / scale), int(h / scale)])
            scores.append(score)
    detections = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        if len(indices) > 0:
            for idx in indices.flatten():
                b = boxes[idx]
                xmin = max(0, b[0])
                ymin = max(0, b[1])
                xmax = min(w0, b[0] + b[2])
                ymax = min(h0, b[1] + b[3])
                detections.append({
                    "hazard_type": hazard_info["hazard_type"],
                    "cause_code": hazard_info["cause_code"],
                    "severity": hazard_info["severity"],
                    "advisory": hazard_info["advisory"],
                    "speed_kmh": hazard_info["speed_kmh"],
                    "confidence": float(scores[idx]),
                    "box": [xmin, ymin, xmax, ymax]
                })
    return detections

def main():
    print("=" * 80)
    print("  SIMULTANEOUS RSU MULTI-HAZARD PERCEPTION & LORA V2X ENGINE")
    print("  Target Profile: Raspberry Pi 5 (Quad Cortex-A76 @ 2.4 GHz, 8GB RAM)")
    print("=" * 80)

    base_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"\n[Phase 1] Process Baseline RSS: {base_ram:.1f} MB")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("\n[Phase 2] Loading all 3 models simultaneously into RAM...")
    sessions = {}
    for h in HAZARDS:
        t0 = time.perf_counter()
        session = ort.InferenceSession(str(h["model"]), opts, providers=["CPUExecutionProvider"])
        t_load = (time.perf_counter() - t0) * 1000.0
        sessions[h["id"]] = session
        sz_mb = os.path.getsize(h["model"]) / (1024 * 1024)
        print(f"  ✓ [{h['id'].upper():<12}] Size: {sz_mb:.2f} MB | Initialized: {t_load:.1f} ms")

    loaded_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"\n[Phase 3] All 3 models active in memory:")
    print(f"  Total Process RSS : {loaded_ram:.1f} MB (Overhead: +{loaded_ram - base_ram:.1f} MB)")
    print(f"  Available Headroom: {(8192 - loaded_ram)/8192*100:.1f}% free on 8GB RPi 5")

    print("\n[Phase 4] Executing multi-hazard inference across all 3 streams...")
    all_hazards = []
    total_time = 0.0

    for h in HAZARDS:
        session = sessions[h["id"]]
        in_name = session.get_inputs()[0].name
        blob, orig_shape, ratio_pad, orig_img = preprocess(h["image"], input_size=768)

        t0 = time.perf_counter()
        preds = session.run(None, {in_name: blob})[0]
        t_infer = (time.perf_counter() - t0) * 1000.0
        total_time += t_infer

        dets = decode_detections(preds, orig_shape, ratio_pad, h, conf_thresh=h["conf_thresh"])
        all_hazards.extend(dets)

        print(f"  ✓ Stream [{h['id'].upper():<12}] Inference: {t_infer:.1f} ms | Found {len(dets)} hazard(s):")
        for d in dets:
            print(f"      -> {d['hazard_type']} | Conf: {d['confidence']*100:.1f}% | Box: {d['box']}")

    print(f"\n  Cumulative Perception Latency: {total_time:.1f} ms")
    print(f"  Total Verified Hazards       : {len(all_hazards)}")

    now_ms = int(time.time() * 1000)
    unified_beacon = {
        "header": {
            "protocolVersion": 2,
            "messageID": 1,
            "stationID": "RSU_GATEWAY_NODE_01",
            "timestampMs": now_ms,
            "loraModulation": {
                "frequencyMhz": 868.1,
                "spreadingFactor": "SF7",
                "bandwidthKhz": 125,
                "txPowerDbm": 14
            }
        },
        "denm": {
            "management": {
                "actionID": f"RSU_GATEWAY_{now_ms}",
                "detectionTime": now_ms,
                "referenceTime": now_ms,
                "validityDurationSec": 300,
                "transmissionIntervalMs": 100
            },
            "situation": {
                "totalHazardsDetected": len(all_hazards),
                "hazardList": [
                    {
                        "trackID": idx,
                        "hazardType": hz["hazard_type"],
                        "causeCode": hz["cause_code"],
                        "severity": hz["severity"],
                        "confidence": round(hz["confidence"], 3),
                        "boundingBox": hz["box"],
                        "advisoryAction": hz["advisory"],
                        "advisorySpeedKmh": hz["speed_kmh"]
                    }
                    for idx, hz in enumerate(all_hazards, 1)
                ]
            }
        }
    }

    out_file = RESULTS_DIR / "unified_rsu_lora_denm_beacon.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(unified_beacon, f, indent=2)
    print(f"\n[Phase 5] Disseminated Unified LoRa/V2X DENM Beacon to: {out_file.name}")
    print("=" * 80)

if __name__ == "__main__":
    main()
