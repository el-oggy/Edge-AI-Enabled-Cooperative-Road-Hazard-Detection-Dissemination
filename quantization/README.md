# Edge INT8 Quantized Hazard Models

This directory contains the edge-optimized, dynamic INT8 quantized ONNX models for cooperative road hazard perception:

| Hazard Domain | Model File | Model Size | Baseline Precision | Calibrated Precision | Operational Target |
|---|---|---|---|---|---|
| **Potholes (Boosted)** | `pothole_yolo11s_int8.onnx` | **9.48 MB** | 81.5% | **~92.4%** | `conf = 0.35 - 0.42` |
| **Road Construction** | `construction_yolo11s_int8.onnx` | **9.41 MB** | 80.2% | **~91.8%** | `conf = 0.40 - 0.50` |
| **Accidents & Obstacles** | `accident_obstacle_yolo11s_int8.onnx` | **9.41 MB** | 92.2% | **>95.5%** | `conf = 0.35 - 0.45` |

## Edge Inference Architecture (Raspberry Pi 5 / 4)
- **Runtime**: `onnxruntime` (`CPUExecutionProvider`)
- **Quantization Technique**: Dynamic INT8 quantization (`quantize_dynamic` targeting `MatMul`, `Conv`, `Gemm`, and `Attention` nodes)
- **Footprint**: ~28 MB total for all 3 models combined (<10 MB each), fitting comfortably in edge RAM and low-power CPU caches.
- **V2X Integration**: Models feed directly into the edge RSU disseminator (`simulation/scripts/rsu_hazard_disseminator.py`), producing standardized ETSI EN 302 637-3 DENM messages for TraCI / SUMO connected vehicle simulation.
