# Road Construction Hazard Detector (YOLO11n Edge Optimized)

This folder contains the production weights, INT8 quantized model, and edge deployment scripts for the **Road Construction Hazard Detector** (barricades, construction barriers, and work zone obstacles).

---

## 🎯 Model Weights Overview

| File | Format | File Size | Target Device | Precision / Quantization |
| :--- | :--- | :--- | :--- | :--- |
| **[`weights/best_int8.onnx`](weights/best_int8.onnx)** | **ONNX INT8** | **2.87 MB** | **Raspberry Pi 3 / 4 / 5 Edge RSUs** | **Dynamic INT8 (QInt8)** |
| [`weights/best.onnx`](weights/best.onnx) | ONNX FP32 | 10.11 MB | Desktop / Laptop / Jetson | Full Precision (Opset 12) |
| [`weights/best.pt`](weights/best.pt) | PyTorch | 5.22 MB | Training & Validation Runtime | FP32 |

---

## 📊 Benchmark Metrics (Held-out Frozen Test Split — 122 Images)

- **Precision**: **89.84%** (Under 10% false alarms on complex road scenes)
- **Recall**: **77.33%** (High sensitivity to road barricades and work zones)
- **mAP@50**: **89.20%**
- **mAP@50-95**: **60.71%**
- **Compression Ratio**: **3.52x reduction** (10.11 MB $\to$ 2.87 MB)

---

## 🚀 Raspberry Pi 3 Quickstart Guide

### 1. Requirements
Install the lightweight runtime on your Raspberry Pi:
```bash
pip install onnxruntime opencv-python-headless numpy pillow
```

### 2. Run Inference with Visual Bounding Boxes
```bash
python edge_pi3_inference.py \
  --model weights/best_int8.onnx \
  --source sample_construction.jpg \
  --output detection_result.jpg \
  --conf 0.25
```

### 3. Run Benchmark Mode (Latency & Core Utilization)
```bash
python edge_pi3_inference.py \
  --model weights/best_int8.onnx \
  --source sample_construction.jpg \
  --benchmark \
  --runs 50
```
