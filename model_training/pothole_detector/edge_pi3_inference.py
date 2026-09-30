"""
Edge-AI Road Hazard Perception — Lightweight Raspberry Pi 3 INT8 Inference Script
Optimized for ARM Cortex-A53 / Quad-Core 1.2GHz / 1GB RAM

Usage on Raspberry Pi 3 / Local PC:
  1. pip3 install onnxruntime opencv-python-headless numpy
  2. python3 edge_pi3_inference.py --model ../models/pothole_yolo11n_rdd_v2/weights/best_int8.onnx --image ../data/sample_pothole_prominent.jpg
  3. python3 edge_pi3_inference.py --benchmark
"""

import os
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort

def preprocess(img_path_or_array, input_size=640):
    if isinstance(img_path_or_array, str):
        img0 = cv2.imread(img_path_or_array)
        if img0 is None:
            raise ValueError(f"Could not read image: {img_path_or_array}")
    else:
        img0 = img_path_or_array
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

def postprocess(preds, orig_shape, ratio_pad, conf_thresh=0.25, iou_thresh=0.45):
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
            xmin = max(0, b[0])
            ymin = max(0, b[1])
            xmax = min(w0, b[0] + b[2])
            ymax = min(h0, b[1] + b[3])
            detections.append({
                "class": "pothole",
                "confidence": scores[idx],
                "box": [xmin, ymin, xmax, ymax]
            })
    return detections

def draw_detections(img, detections):
    annotated = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["confidence"]
        label = f"Pothole {conf:.2f}"
        # Draw bounding box (Orange-Red: safety alert)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
        # Draw label background
        (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + w_txt + 4, max(20, y1)), (0, 140, 255), -1)
        cv2.putText(annotated, label, (x1 + 2, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated

def main():
    parser = argparse.ArgumentParser(description="Run YOLO11n INT8 Inference on Raspberry Pi 3")
    parser.add_argument("--model", type=str, default="models/pothole_yolo11n_rdd_v2/weights/best_int8.onnx", help="Path to best_int8.onnx")
    parser.add_argument("--image", type=str, default="data/sample_pothole_prominent.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="data/output_detected.jpg", help="Path to save annotated image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--benchmark", action="store_true", help="Run hardware latency benchmark loops")
    args = parser.parse_args()

    # Find model file
    model_path = args.model
    if not os.path.exists(model_path):
        alt_path = os.path.join("..", args.model)
        if os.path.exists(alt_path):
            model_path = alt_path

    print("=======================================================")
    print(f"Loading Quantized INT8 ONNX Model: {model_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4 # Quad-core Cortex-A53
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Model initialized successfully! Model input: {input_name}")
    print("=======================================================")

    if args.benchmark:
        print("\n--- Running Latency Benchmark on 640x640 Input (20 Iterations) ---")
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        for _ in range(3): _ = session.run(None, {input_name: dummy})
        latencies = []
        for _ in range(20):
            t0 = time.time()
            _ = session.run(None, {input_name: dummy})
            latencies.append((time.time() - t0) * 1000)
        median_lat = np.median(latencies)
        print(f"Median Inference Latency: {median_lat:.1f} ms (~{1000.0/median_lat:.1f} FPS)")

    if args.image and os.path.exists(args.image):
        print(f"\n--- Running Inference on Image: {args.image} ---")
        blob, orig_shape, ratio_pad, img0 = preprocess(args.image)
        t0 = time.time()
        preds = session.run(None, {input_name: blob})[0]
        dt = (time.time() - t0) * 1000
        detections = postprocess(preds, orig_shape, ratio_pad, conf_thresh=args.conf)
        print(f"Inference latency: {dt:.1f} ms | Potholes detected: {len(detections)}")
        for i, d in enumerate(detections, 1):
            print(f"  [Detection #{i}] Class: {d['class']} | Conf: {d['confidence']*100:.1f}% | BBox: {d['box']}")

        if detections:
            annotated = draw_detections(img0, detections)
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            cv2.imwrite(args.output, annotated)
            print(f"Saved annotated detection output to: {args.output}")

if __name__ == "__main__":
    main()
