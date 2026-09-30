# Pothole Hazard Detector (YOLO11n Edge Optimized)

This folder contains the production weights, INT8 quantized model, and edge deployment scripts for the **Pothole Road Hazard Detector**.

---

## 🎯 Model Weights Overview

| File | Format | File Size | Target Device | Precision / Quantization |
| :--- | :--- | :--- | :--- | :--- |
| **[`weights/best_int8.onnx`](weights/best_int8.onnx)** | **ONNX INT8** | **2.87 MB** | **Raspberry Pi 3 / 4 / 5 Edge RSUs** | **Dynamic INT8 (QInt8)** |
| [`weights/best.onnx`](weights/best.onnx) | ONNX FP32 | 10.11 MB | Desktop / Laptop / Jetson | Full Precision (Opset 12) |
| [`weights/best.pt`](weights/best.pt) | PyTorch | 5.47 MB | Training / Evaluation / PyTorch Runtime | FP32 |

---

## 🚀 Raspberry Pi 3 Quickstart Guide

To run inference directly on your **Raspberry Pi 3** using the lightweight INT8 model:

### 1. Requirements
Install the required lightweight runtime on your Raspberry Pi:
```bash
pip install onnxruntime opencv-python-headless numpy pillow
```

### 2. Run Inference with Visual Bounding Boxes
```bash
python edge_pi3_inference.py \
  --model weights/best_int8.onnx \
  --source sample_test_images/sample_pothole_prominent.jpg \
  --output detection_result.jpg \
  --conf 0.25
```

### 3. Run Benchmark Mode (Latency & Memory Profiling)
```bash
python edge_pi3_inference.py \
  --model weights/best_int8.onnx \
  --source sample_test_images/sample_pothole_prominent.jpg \
  --benchmark \
  --runs 50
```

---

## 📊 Benchmark Metrics (Held-out Frozen Test Split)

- **Recall**: **69.57%** (112 true hazard detections / 161 ground-truth potholes)
- **Precision**: **80.34%** (Only 27 false alarms across 67 diverse road test images)
- **mAP@50**: **77.15%**
- **mAP@50-95**: **49.64%**
- **Compression Ratio**: **3.52x reduction** (10.11 MB $\to$ 2.87 MB)
