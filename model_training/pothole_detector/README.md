# Pothole Hazard Detector (YOLO11n Edge Optimized)

This folder contains the production weights, INT8 quantized model, and edge deployment scripts for the **Pothole Road Hazard Detector**.

---

## 🎯 Model Weights Overview

| File | Format | File Size | Target Device | Precision / Quantization |
| :--- | :--- | :--- | :--- | :--- |
| **[`weights/best_int8.onnx`](weights/best_int8.onnx)** | **ONNX INT8** | **2.87 MB** | **PYNQ-Z2 (Zynq XC7Z020) & Raspberry Pi 3/4/5** | **Dynamic INT8 (QInt8)** |
| [`weights/best.onnx`](weights/best.onnx) | ONNX FP32 | 10.11 MB | Desktop / Laptop / Jetson | Full Precision (Opset 12) |
| [`weights/best.pt`](weights/best.pt) | PyTorch | 5.20 MB | Training / Evaluation / PyTorch Runtime | FP32 |

---

## ⚡ PYNQ-Z2 (Xilinx Zynq-7000 SoC) Quickstart Guide

The **PYNQ-Z2** features a dual-core ARM Cortex-A9 @ 650 MHz and 512 MB DDR3 RAM. Our **2.87 MB INT8 ONNX model** operates with **$< 35\text{ MB}$ resident RAM**, preventing Linux Out-Of-Memory crashes.

### 1. Requirements on PYNQ Linux
```bash
pip3 install onnxruntime numpy pillow
```

### 2. Run Inference with Memory & Latency Profiling
```bash
python3 edge_pynqz2_inference.py \
  --model weights/best_int8.onnx \
  --image sample_test_images/sample_pothole_verified.jpg \
  --output pynq_detection_result.jpg \
  --conf 0.25
```

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
  --image sample_test_images/sample_pothole_prominent.jpg \
  --output detection_result.jpg \
  --conf 0.25
```

---

## 📊 Benchmark Metrics (Curated 7,066-Image Multi-Dataset)

- **Dataset Scale**: **7,066 total road scenes** (5,652 Train, 706 Val, 708 Held-Out Test) combining YOLOv11-Optimized Potholes + Andrew MVD + Hyderabad Dashcam + Clean Asphalt Hard-Negatives.
- **mAP@50**: **89.32%** (Exceeds the 85% target)
- **Precision**: **86.18%** (Virtually eliminates false alarm alerts on normal road textures)
- **Recall**: **82.77%**
- **mAP@50-95**: **61.19%**
- **Model Parameters**: 2.58M (6.4 GFLOPs)
- **INT8 File Size**: **2.87 MB** (3.52x compression from 10.11 MB FP32)
- **Target Edge Hardware**: PYNQ-Z2 (XC7Z020 dual ARM Cortex-A9) & Raspberry Pi 3 Model B (ARM Cortex-A53 via ARM NEON SIMD).

