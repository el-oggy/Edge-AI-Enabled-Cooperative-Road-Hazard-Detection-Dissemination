# Models Directory

This directory stores exported machine learning model artifacts.

For all active training runs, quantized INT8 models for Raspberry Pi 3 edge deployment, and deployment scripts, please see:
👉 **[`model_training/`](../model_training/)**

## Directory Layout
- **[`model_training/pothole_detector/`](../model_training/pothole_detector/)**:
  - `weights/best_int8.onnx` (2.87 MB - dynamic INT8 quantized model for Raspberry Pi 3)
  - `weights/best.onnx` (10.11 MB - FP32 ONNX)
  - `weights/best.pt` (5.47 MB - PyTorch)
  - `edge_pi3_inference.py` (Raspberry Pi 3 test runner)
  - `README.md` (Raspberry Pi quickstart)
- **[`model_training/road_construction_detector/`](../model_training/road_construction_detector/)**:
  - Upcoming Road Construction & Work Zone hazard model.
- **[`models/pothole_yolo11n_rdd_v2/`](pothole_yolo11n_rdd_v2/)**:
  - Archive of Experiment 002 (RDD2022 + Andrew MVD merged training).
