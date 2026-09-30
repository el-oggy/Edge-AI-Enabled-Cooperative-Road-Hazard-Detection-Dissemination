"""
Edge-AI Road Hazard Perception — Lightweight Raspberry Pi 3 INT8 Inference Script
Optimized for ARM Cortex-A53 / Quad-Core 1.2GHz / 1GB RAM

Usage on Raspberry Pi 3:
  1. pip3 install onnxruntime opencv-python-headless numpy
  2. python3 edge_pi3_inference.py --model ../models/pothole_yolo11n_rdd_v2/weights/best_int8.onnx --image test.jpg
"""

import os
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort

def preprocess(img_path, input_size=640):
    img0 = cv2.imread(img_path)
    if img0 is None:
        raise ValueError(f"Could not read image: {img_path}")
    h0, w0 = img0.shape[:2]

    # Letterbox resize maintaining aspect ratio
    r = min(input_size / h0, input_size / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = (input_size - new_unpad[0]) / 2, (input_size - new_unpad[1]) / 2

    resized = cv2.resize(img0, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Convert BGR to RGB, normalize [0, 1], add batch dimension BCHW
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, (h0, w0), (r, dw, dh), img0

def postprocess(preds, orig_shape, ratio_pad, conf_thresh=0.35, iou_thresh=0.45):
    # preds shape: (1, 5, 8400) -> transpose to (8400, 5) [x, y, w, h, conf]
    predictions = np.squeeze(preds).T
    h0, w0 = orig_shape
    r, dw, dh = ratio_pad

    boxes = []
    scores = []
    for row in predictions:
        score = row[4]
        if score >= conf_thresh:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            xmin = (cx - w / 2 - dw) / r
            ymin = (cy - h / 2 - dh) / r
            xmax = (cx + w / 2 - dw) / r
            ymax = (cy + h / 2 - dh) / r
            boxes.append([int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)])
            scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            b = boxes[idx]
            detections.append({
                "class": "pothole",
                "confidence": scores[idx],
                "box": [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            })
    return detections

def main():
    parser = argparse.ArgumentParser(description="Run YOLO11n INT8 Inference on Raspberry Pi 3")
    parser.add_argument("--model", type=str, default="../models/pothole_yolo11n_rdd_v2/weights/best_int8.onnx", help="Path to best_int8.onnx")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--benchmark", action="store_true", help="Run 20 dummy inference benchmark loops")
    args = parser.parse_args()

    print(f"Loading Quantized INT8 ONNX Model: {args.model}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4 # Quad-core Cortex-A53
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if args.benchmark or not args.image:
        print("Running hardware latency benchmark on 640x640 input (20 iterations)...")
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        # warmup
        for _ in range(3): _ = session.run(None, {input_name: dummy})
        latencies = []
        for _ in range(20):
            t0 = time.time()
            _ = session.run(None, {input_name: dummy})
            latencies.append((time.time() - t0) * 1000)
        median_lat = np.median(latencies)
        print(f"Benchmark Result: {median_lat:.1f} ms per frame (~{1000.0/median_lat:.1f} FPS)")

    if args.image and os.path.exists(args.image):
        blob, orig_shape, ratio_pad, img0 = preprocess(args.image)
        t0 = time.time()
        preds = session.run(None, {input_name: blob})[0]
        dt = (time.time() - t0) * 1000
        detections = postprocess(preds, orig_shape, ratio_pad, conf_thresh=args.conf)
        print(f"Inference latency: {dt:.1f} ms | Potholes detected: {len(detections)}")
        for d in detections:
            print(f"  -> Pothole conf={d['confidence']:.2f} box={d['box']}")

if __name__ == "__main__":
    main()
