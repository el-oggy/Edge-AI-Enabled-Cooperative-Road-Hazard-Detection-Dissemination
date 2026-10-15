#!/usr/bin/env python3
"""
Edge RSU Inference Runner for Multi-Hazard Road Safety Detection (YOLO11n ONNX INT8)
Target Platform: Raspberry Pi 3 Model B (ARM Cortex-A53 @ 1.2 GHz, 1GB RAM)
Supported Hazards:
  Class 0: pothole
  Class 1: construction_hazard (barricades, work zone barriers, diversion markers)
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort

CLASS_NAMES = ["pothole", "construction_hazard"]
CLASS_COLORS = {
    0: (255, 165, 0),   # Orange / Cyan in BGR: (0, 165, 255)
    1: (0, 0, 255),     # Bright Red for Construction Hazard: (0, 0, 255)
}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess(img_bgr, input_shape=(640, 640)):
    """Preprocess image for YOLO11 ONNX inference."""
    img_padded, r, (dw, dh) = letterbox(img_bgr, new_shape=input_shape)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch, r, (dw, dh)

def postprocess(output, orig_shape, r, dw_dh, conf_thresh=0.25, iou_thresh=0.45):
    """Postprocess raw YOLO11 output tensor [1, 4 + num_classes, 8400]."""
    preds = np.squeeze(output[0])  # [num_classes + 4, 8400]
    preds = np.transpose(preds, (1, 0))  # [8400, num_classes + 4]

    boxes = preds[:, :4]
    scores_all = preds[:, 4:]

    class_ids = np.argmax(scores_all, axis=1)
    confidences = np.max(scores_all, axis=1)

    mask = confidences > conf_thresh
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], [], []

    dw, dh = dw_dh
    cx = boxes[:, 0] - dw
    cy = boxes[:, 1] - dh
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = (cx - w / 2) / r
    y1 = (cy - h / 2) / r
    x2 = (cx + w / 2) / r
    y2 = (cy + h / 2) / r

    h_orig, w_orig = orig_shape[:2]
    x1 = np.clip(x1, 0, w_orig)
    y1 = np.clip(y1, 0, h_orig)
    x2 = np.clip(x2, 0, w_orig)
    y2 = np.clip(y2, 0, h_orig)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confidences.tolist(), conf_thresh, iou_thresh)

    final_boxes, final_confs, final_classes = [], [], []
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        for idx in indices:
            final_boxes.append([x1[idx], y1[idx], x2[idx], y2[idx]])
            final_confs.append(float(confidences[idx]))
            final_classes.append(int(class_ids[idx]))

    return final_boxes, final_confs, final_classes

def run_inference(model_path, image_path, output_path=None, conf_thresh=0.25, benchmark=False, runs=30):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4  # Match Pi 3's 4 CPU cores
    
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    orig_shape = img.shape
    input_tensor, r, dw_dh = preprocess(img)

    if benchmark:
        print(f"Running benchmark on {runs} iterations...")
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = session.run(None, {input_name: input_tensor})
            latencies.append((time.perf_counter() - t0) * 1000)
        
        avg_lat = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        print(f"Benchmark Results:")
        print(f"  Average Latency: {avg_lat:.2f} ms ({1000/avg_lat:.1f} FPS)")
        print(f"  P50 Latency:     {p50:.2f} ms")
        print(f"  P95 Latency:     {p95:.2f} ms")

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    inf_time_ms = (time.perf_counter() - t0) * 1000

    boxes, confs, clss = postprocess(outputs, orig_shape, r, dw_dh, conf_thresh=conf_thresh)
    print(f"\nInference completed in {inf_time_ms:.2f} ms. Detected {len(boxes)} road hazards:")

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
        print(f"  Hazard [{i+1}]: {name.upper()} (Confidence: {conf*100:.1f}%) at [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")

        if output_path:
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name}: {conf*100:.1f}%"
            cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visual detection to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Hazard Edge RSU Inference Runner (Raspberry Pi 3)")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX / INT8 model file")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="detection_result.jpg", help="Path to save annotated output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--runs", type=int, default=30, help="Number of benchmark iterations")
    args = parser.parse_args()

    run_inference(args.model, args.source, args.output, args.conf, args.benchmark, args.runs)

if __name__ == "__main__":
    main()
