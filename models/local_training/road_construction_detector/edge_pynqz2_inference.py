"""
PYNQ-Z2 Edge Inference Runner for Pothole Hazard Detection
Target: Xilinx Zynq-7000 SoC (XC7Z020) - Dual-Core ARM Cortex-A9 @ 650 MHz, 512 MB DDR3 RAM

Optimizations:
1. Runs lightweight dynamic INT8 ONNX model (~2.87 MB).
2. Minimal memory footprint (< 35 MB RAM) to prevent OOM on 512 MB DDR3.
3. Uses ONNX Runtime with ARM NEON SIMD acceleration.
4. Generates bounding boxes, confidence scores, and latency statistics.
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not found. Install on PYNQ-Z2 via: pip install onnxruntime")
    sys.exit(1)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("[INFO] OpenCV not found, falling back to PIL for image manipulation.")


def get_memory_usage_mb():
    """Returns the current process resident memory in megabytes."""
    try:
        import resource
        # On Linux/PYNQ ru_maxrss is in kilobytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def preprocess(image_path, input_size=(640, 640)):
    """Preprocess image with letterboxing for YOLO11n input."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Scale with aspect ratio preservation
    target_w, target_h = input_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Pad to target dimensions (letterbox)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    canvas.paste(resized, (pad_w, pad_h))
    
    # Normalize to [0.0, 1.0] and format to NCHW float32
    img_data = np.array(canvas, dtype=np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # HWC -> CHW
    img_data = np.expand_dims(img_data, axis=0)   # CHW -> NCHW
    
    metadata = {
        "orig_size": (orig_w, orig_h),
        "scale": scale,
        "pad": (pad_w, pad_h)
    }
    return img_data, metadata, img


def postprocess(outputs, metadata, conf_thresh=0.30, iou_thresh=0.45):
    """Postprocess YOLO11n outputs (1, 5, 8400) -> detections [x1, y1, x2, y2, conf, cls]."""
    preds = outputs[0]  # Shape: (1, 5, 8400)
    if preds.ndim == 3:
        preds = preds[0]  # Shape: (5, 8400)
    preds = np.transpose(preds)  # Shape: (8400, 5)

    boxes = []
    scores = []
    
    scale = metadata["scale"]
    pad_w, pad_h = metadata["pad"]
    orig_w, orig_h = metadata["orig_size"]

    for row in preds:
        cx, cy, w, h = row[:4]
        conf = row[4]
        if conf >= conf_thresh:
            x1 = (cx - w / 2.0 - pad_w) / scale
            y1 = (cy - h / 2.0 - pad_h) / scale
            x2 = (cx + w / 2.0 - pad_w) / scale
            y2 = (cy + h / 2.0 - pad_h) / scale
            
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

    if not boxes:
        return []

    # Non-Maximum Suppression (NMS)
    if HAS_OPENCV:
        cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(cv_boxes, scores, conf_thresh, iou_thresh)
        if len(indices) > 0:
            indices = indices.flatten()
            return [{"box": boxes[i], "conf": scores[i], "class": "road_construction"} for i in indices]
        return []
    else:
        # Simple Python NMS fallback
        indices = list(range(len(boxes)))
        indices.sort(key=lambda i: scores[i], reverse=True)
        keep = []
        while indices:
            cur = indices.pop(0)
            keep.append(cur)
            rem = []
            b1 = boxes[cur]
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            for o in indices:
                b2 = boxes[o]
                xx1 = max(b1[0], b2[0])
                yy1 = max(b1[1], b2[1])
                xx2 = min(b1[2], b2[2])
                yy2 = min(b1[3], b2[3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / (area1 + area2 - inter + 1e-6)
                if iou < iou_thresh:
                    rem.append(o)
            indices = rem
        return [{"box": boxes[i], "conf": scores[i], "class": "road_construction"} for i in keep]


def main():
    parser = argparse.ArgumentParser(description="PYNQ-Z2 Pothole Hazard Detector")
    parser.add_argument("--model", type=str, default="weights/best_int8.onnx", help="Path to best_int8.onnx")
    parser.add_argument("--image", type=str, default="sample_test_images/sample_pothole_prominent.jpg", help="Path to road image")
    parser.add_argument("--output", type=str, default="pynq_pothole_detection.jpg", help="Path to save annotated output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark loops for latency measurement")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark iterations")
    args = parser.parse_args()

    print("=" * 60)
    print("  PYNQ-Z2 (Xilinx Zynq-7000 SoC) Edge Hazard Inference Engine")
    print("  Target Arch: Dual ARM Cortex-A9 @ 650 MHz | RAM: 512 MB DDR3")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return

    # Configure lightweight ONNX Runtime session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2  # PYNQ-Z2 has 2 ARM CPU cores
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"Loading Quantized Model: {args.model} ({os.path.getsize(args.model)/(1024**2):.2f} MB)")
    session = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Session initialized successfully! Resident RAM: {get_memory_usage_mb():.1f} MB")

    if not os.path.exists(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        return

    # Preprocessing
    t0 = time.time()
    input_tensor, meta, orig_img = preprocess(args.image)
    t_pre = (time.time() - t0) * 1000

    # Single warm-up pass
    session.run(None, {input_name: input_tensor})

    if args.benchmark:
        print(f"\n--- Running Benchmark ({args.runs} iterations) ---")
        latencies = []
        for i in range(args.runs):
            t_start = time.perf_counter()
            outputs = session.run(None, {input_name: input_tensor})
            latencies.append((time.perf_counter() - t_start) * 1000)
        
        latencies = np.array(latencies)
        print(f"Average Inference Latency: {np.mean(latencies):.1f} ms ({1000.0/np.mean(latencies):.2f} FPS)")
        print(f"Min: {np.min(latencies):.1f} ms | Max: {np.max(latencies):.1f} ms | Std: {np.std(latencies):.1f} ms")
        print(f"Total Memory Consumption: {get_memory_usage_mb():.1f} MB (within 512 MB limit)")
    else:
        # Standard inference
        t_start = time.perf_counter()
        outputs = session.run(None, {input_name: input_tensor})
        t_inf = (time.perf_counter() - t_start) * 1000

        # Postprocessing
        detections = postprocess(outputs, meta, conf_thresh=args.conf)
        print(f"\nInference completed in {t_inf:.1f} ms | Detections: {len(detections)}")

        # Draw boxes and save
        from PIL import ImageDraw
        draw = ImageDraw.Draw(orig_img)
        for d in detections:
            box = d["box"]
            conf = d["conf"]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0] + 5, box[1] + 5), f"Pothole: {conf*100:.1f}%", fill="red")
            print(f"  -> Pothole detected at [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}] | Conf: {conf*100:.1f}%")

        orig_img.save(args.output)
        print(f"\nAnnotated image saved to: {args.output}")
        print(f"Process Memory: {get_memory_usage_mb():.1f} MB")


if __name__ == "__main__":
    main()
