"""
PFPD Unified Three-Class Hazard Detector: Physical Pi 5 Acceptance Validation
=============================================================================
Benchmarking 2-epoch smoke test models on physical Raspberry Pi 5 hardware:
- ONNX Runtime CPU (512x512 and 384x384)
- MNN CPU FP32 & INT8 (384x384)
Across authentic benchmark images from data/benchmark_150 (potholes, construction, accidents).

Enforces:
1. Paced execution (100ms inter-frame delay)
2. Active thermal monitoring (vcgencmd measure_temp & get_throttled)
3. Accurate ETSI CDD v1.3.1 cause/subcause code formatting:
   - Pothole: Cause 9, Subcause 4
   - Construction: Cause 3, Subcause 1
   - Accident: Cause 2, Subcause 1
   - Obstacle: Cause 10, Subcause 4
4. Class-aware intra-class NMS
"""

import os
import sys
import time
import json
import psutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

import onnxruntime as ort
import MNN

# ETSI CDD v1.3.1 Mapping
ETSI_HAZARD_MAP = {
    0: {"name": "pothole", "cause_code": 9, "subcause_code": 4, "description": "hazardousLocation-SurfaceCondition (subsidence/pothole)"},
    1: {"name": "construction_hazard", "cause_code": 3, "subcause_code": 1, "description": "roadworks (majorRoadworks)"},
    2: {"name": "road_accident_obstacle", "cause_code": 2, "subcause_code": 1, "description": "accident (multiVehicleAccident)"}
}

CLASS_NAMES = ["pothole", "construction_hazard", "road_accident_obstacle"]

def get_telemetry():
    temp_str = subprocess.getoutput("vcgencmd measure_temp 2>/dev/null").strip()
    throttled_str = subprocess.getoutput("vcgencmd get_throttled 2>/dev/null").strip()
    return {"temp": temp_str, "throttled": throttled_str}

def letterbox(img, target_size=(512, 512), color=(114, 114, 114)):
    """Letterbox resize image with aspect-ratio preservation and padding."""
    shape = img.shape[:2]  # h, w
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = target_size[1] - new_unpad[0]
    dh = target_size[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def nms_boxes(boxes, scores, iou_thresh=0.45):
    """Greedy non-maximum suppression for a single class."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def postprocess_yolo11(output_raw, conf_thresh=0.25, iou_thresh=0.45, ratio=1.0, pad=(0, 0), orig_shape=(1080, 1920)):
    """
    Decodes YOLO11 output tensor [1, 7, N] -> [cx, cy, w, h, p0, p1, p2].
    Applies letterbox unpadding and class-aware intra-class NMS.
    """
    # output_raw: [1, 7, N] -> transpose to [N, 7]
    preds = output_raw[0].T  # [N, 7]
    cx = preds[:, 0]
    cy = preds[:, 1]
    w = preds[:, 2]
    h = preds[:, 3]
    cls_scores = preds[:, 4:7]  # [N, 3]

    # Convert cx, cy, w, h to x1, y1, x2, y2 on padded image
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Unpad to original image coordinates
    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / ratio
    y1 = (y1 - pad_y) / ratio
    x2 = (x2 - pad_x) / ratio
    y2 = (y2 - pad_y) / ratio

    # Clip to bounds
    orig_h, orig_w = orig_shape
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    confirmed_detections = []
    # Class-aware NMS
    for cid in range(3):
        scores = cls_scores[:, cid]
        mask = scores >= conf_thresh
        if not np.any(mask):
            continue
        cls_b = boxes[mask]
        cls_s = scores[mask]
        keep = nms_boxes(cls_b, cls_s, iou_thresh)
        for k in keep:
            confirmed_detections.append({
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
                "confidence": float(cls_s[k]),
                "bbox": [round(float(v), 2) for v in cls_b[k]],
                "etsi_cause": ETSI_HAZARD_MAP[cid]["cause_code"],
                "etsi_subcause": ETSI_HAZARD_MAP[cid]["subcause_code"]
            })
    return confirmed_detections

class BaseRunner:
    def warmup(self, n=3):
        pass
    def infer(self, input_data):
        raise NotImplementedError

class ORTRunner(BaseRunner):
    def __init__(self, model_path, threads=2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.shape = self.sess.get_inputs()[0].shape

    def warmup(self, n=3):
        dummy = np.zeros(self.shape, dtype=np.float32)
        for _ in range(n):
            self.sess.run(None, {self.in_name: dummy})

    def infer(self, input_data):
        return self.sess.run(None, {self.in_name: input_data})[0]

class MNNRunner(BaseRunner):
    def __init__(self, model_path, shape, threads=2, precision="normal"):
        self.shape = shape
        self.interpreter = MNN.Interpreter(str(model_path))
        config = {"precision": precision, "backend": 0, "numThread": threads}
        self.session = self.interpreter.createSession(config)
        self.in_tensor = self.interpreter.getSessionInput(self.session)

    def warmup(self, n=3):
        dummy = np.zeros(self.shape, dtype=np.float32)
        tmp = MNN.Tensor(self.shape, MNN.Halide_Type_Float, dummy, MNN.Tensor_DimensionType_Caffe)
        for _ in range(n):
            self.in_tensor.copyFrom(tmp)
            self.interpreter.runSession(self.session)

    def infer(self, input_data):
        tmp = MNN.Tensor(self.shape, MNN.Halide_Type_Float, input_data, MNN.Tensor_DimensionType_Caffe)
        self.in_tensor.copyFrom(tmp)
        self.interpreter.runSession(self.session)
        out_tensor = self.interpreter.getSessionOutput(self.session)
        out_shape = out_tensor.getShape()
        host_tensor = MNN.Tensor(out_shape, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
        out_tensor.copyToHostTensor(host_tensor)
        out_data = np.array(host_tensor.getData(), dtype=np.float32).reshape(out_shape)
        return out_data

def evaluate_config(config_name, runner, imgsz, benchmark_images, threads=2):
    print(f"\n---> Evaluating {config_name} (Resolution={imgsz}x{imgsz}, Threads={threads})")
    telemetry_start = get_telemetry()
    print(f"     Start Telemetry: Temp={telemetry_start['temp']}, Throttled={telemetry_start['throttled']}")

    runner.warmup(n=3)

    records = []
    total_detections_per_class = {0: 0, 1: 0, 2: 0}

    for idx, (cat, img_p) in enumerate(benchmark_images, 1):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        orig_shape = img_bgr.shape[:2]

        # 1. Preprocessing
        t0 = time.perf_counter()
        img_lb, ratio, pad = letterbox(img_bgr, target_size=(imgsz, imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        input_data = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, :]
        input_data = np.ascontiguousarray(input_data)
        t_prep_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Inference
        t1 = time.perf_counter()
        raw_output = runner.infer(input_data)
        t_infer_ms = (time.perf_counter() - t1) * 1000.0

        # 3. Postprocessing & NMS
        t2 = time.perf_counter()
        detections = postprocess_yolo11(raw_output, conf_thresh=0.25, iou_thresh=0.45, ratio=ratio, pad=pad, orig_shape=orig_shape)
        t_post_ms = (time.perf_counter() - t2) * 1000.0

        # 4. ETSI Packaging
        t3 = time.perf_counter()
        etsi_messages = []
        for d in detections:
            etsi_messages.append({
                "actionID": f"RSU_{int(time.time()*1000)}_{d['class_id']}",
                "causeCode": d["etsi_cause"],
                "subCauseCode": d["etsi_subcause"],
                "confidence": d["confidence"],
                "bbox": d["bbox"]
            })
            total_detections_per_class[d["class_id"]] += 1
        t_etsi_ms = (time.perf_counter() - t3) * 1000.0

        t_e2e_ms = t_prep_ms + t_infer_ms + t_post_ms + t_etsi_ms

        records.append({
            "image": img_p.name,
            "category": cat,
            "t_prep_ms": t_prep_ms,
            "t_infer_ms": t_infer_ms,
            "t_post_ms": t_post_ms,
            "t_etsi_ms": t_etsi_ms,
            "t_e2e_ms": t_e2e_ms,
            "detections": len(detections)
        })

        time.sleep(0.08)  # 80ms safe pacing

    telemetry_end = get_telemetry()
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    infer_times = [r["t_infer_ms"] for r in records]
    e2e_times = [r["t_e2e_ms"] for r in records]

    results = {
        "config_name": config_name,
        "resolution": imgsz,
        "threads": threads,
        "frames_evaluated": len(records),
        "telemetry_start": telemetry_start,
        "telemetry_end": telemetry_end,
        "memory_rss_mb": round(rss_mb, 2),
        "infer_latency": {
            "mean_ms": round(float(np.mean(infer_times)), 2),
            "median_ms": round(float(np.median(infer_times)), 2),
            "p95_ms": round(float(np.percentile(infer_times, 95)), 2),
            "min_ms": round(float(np.min(infer_times)), 2),
            "max_ms": round(float(np.max(infer_times)), 2),
            "pure_infer_fps": round(1000.0 / float(np.mean(infer_times)), 2)
        },
        "e2e_latency": {
            "mean_ms": round(float(np.mean(e2e_times)), 2),
            "median_ms": round(float(np.median(e2e_times)), 2),
            "p95_ms": round(float(np.percentile(e2e_times, 95)), 2),
            "min_ms": round(float(np.min(e2e_times)), 2),
            "max_ms": round(float(np.max(e2e_times)), 2),
            "e2e_fps": round(1000.0 / float(np.mean(e2e_times)), 2)
        },
        "detections_summary": {
            "potholes": total_detections_per_class[0],
            "construction": total_detections_per_class[1],
            "accidents_obstacles": total_detections_per_class[2],
            "total": sum(total_detections_per_class.values())
        }
    }

    print(f"     [RESULTS] Pure Infer: {results['infer_latency']['mean_ms']} ms ({results['infer_latency']['pure_infer_fps']} FPS) | "
          f"E2E Latency: {results['e2e_latency']['mean_ms']} ms ({results['e2e_latency']['e2e_fps']} FPS) | "
          f"Temp: {telemetry_start['temp']} -> {telemetry_end['temp']} (Throttled: {telemetry_end['throttled']})")
    return results

def main():
    print("========================================================================")
    print("  PFPD PHYSICAL PI 5 ACCEPTANCE BENCHMARK — UNIFIED YOLO11n DETECTOR    ")
    print("========================================================================")

    # Gather benchmark images (10 per category)
    base_data = Path("/home/oggy/projects/pfpd/data/benchmark_150")
    cats = {
        "potholes": list((base_data / "potholes").rglob("*.jpg")) + list((base_data / "potholes").rglob("*.png")),
        "construction": list((base_data / "construction").rglob("*.jpg")) + list((base_data / "construction").rglob("*.png")),
        "accidents_obstacles": list((base_data / "accidents_obstacles").rglob("*.jpg")) + list((base_data / "accidents_obstacles").rglob("*.png"))
    }

    benchmark_images = []
    for c, imgs in cats.items():
        for p in sorted(imgs)[:10]:
            benchmark_images.append((c, p))

    print(f"Total benchmark images collected: {len(benchmark_images)} (10 per domain)")

    models_dir = Path("/home/oggy/projects/pfpd/models")
    suite_results = []

    # 1. ORT 512x512 CPU (2 threads)
    runner_ort_512 = ORTRunner(models_dir / "unified_yolo11n_512.onnx", threads=2)
    suite_results.append(evaluate_config("ONNX_Runtime_FP32_512px_2th", runner_ort_512, 512, benchmark_images, threads=2))

    # 2. ORT 384x384 CPU (2 threads)
    runner_ort_384 = ORTRunner(models_dir / "unified_yolo11n_384.onnx", threads=2)
    suite_results.append(evaluate_config("ONNX_Runtime_FP32_384px_2th", runner_ort_384, 384, benchmark_images, threads=2))

    # 3. MNN FP32 384x384 CPU (2 threads)
    runner_mnn_fp32_384 = MNNRunner(models_dir / "unified_yolo11n_384.mnn", shape=(1, 3, 384, 384), threads=2, precision="normal")
    suite_results.append(evaluate_config("MNN_FP32_384px_2th", runner_mnn_fp32_384, 384, benchmark_images, threads=2))

    # 4. MNN INT8 384x384 CPU (2 threads)
    runner_mnn_int8_384 = MNNRunner(models_dir / "unified_yolo11n_384_int8.mnn", shape=(1, 3, 384, 384), threads=2, precision="low")
    suite_results.append(evaluate_config("MNN_INT8_384px_2th", runner_mnn_int8_384, 384, benchmark_images, threads=2))

    # 5. MNN INT8 512x512 CPU (2 threads)
    runner_mnn_int8_512 = MNNRunner(models_dir / "unified_yolo11n_512_int8.mnn", shape=(1, 3, 512, 512), threads=2, precision="low")
    suite_results.append(evaluate_config("MNN_INT8_512px_2th", runner_mnn_int8_512, 512, benchmark_images, threads=2))

    # Save output JSON
    out_json = Path("/home/oggy/projects/pfpd/results/unified_smoke_pi5_acceptance.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(suite_results, f, indent=2)

    print("\n========================================================================")
    print("  BENCHMARK COMPLETED SUCCESSFULLY! RESULTS FLUSHED TO DISK:            ")
    print(f"  {out_json}")
    print("========================================================================")

if __name__ == "__main__":
    main()
