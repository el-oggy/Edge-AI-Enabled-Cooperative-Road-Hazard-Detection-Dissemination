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

## 📊 Benchmark Metrics (Boosted 3,126-Image Test Split)

- **Dataset Scale**: 3,126 total road scenes (2,509 Train, 305 Val, 315 Held-out Test) combining Andrew MVD + Hyderabad Dashcam + 171 clean hard-negative road images.
- **Precision**: **79.56%**
- **Recall**: **71.85%**
- **mAP@50**: **79.48%** (Highest overall detection accuracy)
- **mAP@50-95**: **52.17%**
- **Model Parameters**: 2.58M (6.4 GFLOPs)
- **INT8 File Size**: **2.87 MB** (3.52x compression from 10.11 MB FP32)
- **Target Hardware**: Raspberry Pi 3 Model B (ARM Cortex-A53 via ARM NEON SIMD) & compatible edge devices.
