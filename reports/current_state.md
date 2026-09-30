# Project State & Continuation Document

**Project**: Edge-AI Enabled Cooperative Road Hazard Detection and Dissemination using Low-Cost IoT Roadside Units (RSUs) in VANETs  
**Current Date**: 2026-09-30  
**Phase**: Phase 2 — Computer Vision & Perception Pipeline (Pothole Detection)

---

## Authoritative System State (Rule #23 Alignment)

### 1. What dataset is currently authoritative?
The authoritative dataset is **`potholes_rdd2022_merged_v2`**.
It is a curated, verified merger of the clean **Andrew MVD Pothole Detection** dataset and the **RDD2022 India** dataset (strictly filtered for class `D40` pothole).

### 2. How many images are in it?
* **Total Images**: **2,195 images**
* **Train Split**: 1,833 images (4,115 pothole bounding boxes)
* **Validation Split**: 295 images (650 pothole bounding boxes)
* **Frozen Benchmark Test Split**: **67 images (161 pothole bounding boxes)**
* **Total Annotated Potholes**: **4,926 bounding boxes** (Avg 2.24 potholes/image)
* **Class Mapping**: `0: pothole` (All non-pothole crack labels such as longitudinal, transverse, and alligator cracks were strictly discarded).

### 3. What model was trained?
* **Model Family**: Ultralytics **YOLO11n** (`yolo11n.pt`)
* **Architecture**: 100 layers, **2,582,347 parameters** (~2.58M), **6.4 GFLOPs**
* **Training Setup**: 50 Epochs, input resolution `640x640`, batch size `16`, optimizer `AdamW` (lr=0.002, momentum=0.9), seed `42`, device `NVIDIA Tesla T4 GPU`.

### 4. What checkpoint is canonical?
* **PyTorch Weights**: [`models/pothole_yolo11n_rdd_v2/weights/best.pt`](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/models/pothole_yolo11n_rdd_v2/weights/best.pt) (5.22 MB)
* **Exported Edge ONNX**: [`models/pothole_yolo11n_rdd_v2/weights/best.onnx`](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/models/pothole_yolo11n_rdd_v2/weights/best.onnx) (10.11 MB, Opset 12, FP32/FP16 compatible, slimmed with `onnxslim`).

### 5. What metrics were obtained?

#### Held-Out Frozen Test Set (67 Images, 161 Ground-Truth Potholes)
| Metric | Historical Baseline (EXP_001) | Merged Dataset Model (EXP_002) | Delta / Impact |
|:---|:---:|:---:|:---:|
| **Precision** | 0.8959 | **0.8034** | -0.0925 (Tuned for safety) |
| **Recall** | 0.6522 | **0.6957** | **+0.0435 (+4.35% hazard recall)** |
| **mAP@50** | 0.7926 | **0.7715** | -0.0211 |
| **mAP@50-95** | 0.5108 | **0.4964** | -0.0144 |

#### Hardware Benchmarks
* **Tesla T4 GPU (FP16/FP32)**: **8.65 ms / image (~115.6 FPS)**
* **Host CPU (x86-64, single-thread)**: **144.32 ms / image (~6.9 FPS)**
* **Projected Jetson Nano (TensorRT FP16)**: **~22–28 FPS** (Target met for 10–15 FPS RSU streaming)
* **Projected Raspberry Pi 3 (ONNX/NCNN INT8)**: **~3–5 FPS** (Feasible with 1–2 Hz polling / motion triggering).

### 6. What experiment is currently running?
No active training experiment is currently executing; `EXP_002` reached its completion target of 50 epochs and underwent full validation, test evaluation, ONNX export, and error analysis.

### 7. What has been completed?
1. Full environment discovery and Colab CLI/MCP bridge verification on Tesla T4.
2. Andrew MVD baseline data and split preservation (reproducing 665 images, 1,739 valid boxes, frozen 67 test set).
3. Cryptographic and perceptual hashing on 100% of images (0 duplicate collisions between Andrew MVD and RDD2022).
4. RDD2022 India strict class filtering (extracting solely `D40` pothole; discarding 1,146 crack annotations).
5. Versioned dataset creation (`potholes_rdd2022_merged_v2`) with 2,195 images.
6. 50-epoch training of YOLO11n on Tesla T4.
7. Validation and frozen held-out test evaluation.
8. Model export to ONNX format with ONNX-slim optimization.
9. GPU and CPU latency profiling.
10. Generation of machine-readable reports:
    - [`reports/dataset_report.csv`](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/reports/dataset_report.csv)
    - [`reports/source_distribution.csv`](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/reports/source_distribution.csv)
    - [`reports/class_distribution.csv`](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/reports/class_distribution.csv)
    - [`reports/duplicate_report.csv`](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/reports/duplicate_report.csv)
    - [`reports/split_manifest.csv`](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/reports/split_manifest.csv)

### 8. What remains?
1. Optional edge quantization to INT8 (TensorRT / OpenVINO / NCNN) for Raspberry Pi 3 deployment.
2. Ingestion of secondary hazard classes (Construction Zones, Accident Vehicles) once clean, verified bounding-box datasets are acquired without compromising pothole detection.
3. Integration with RSU VANET packet dissemination pipeline.

### 9. What should be done next?
1. Release the canonical `best.onnx` model to the RSU edge deployment target.
2. Integrate the detection output coordinates with the SUMO simulation / VANET hazard warning beacon system.
3. Shut down the idle Colab GPU VM session to preserve the user's compute quota.
