# MASTER PROJECT PROMPT & SPECIFICATION
## Edge-AI Enabled Cooperative Road Hazard Detection & Dissemination

> **Reference Memory**: Refer to [memory.md](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/memory.md) or [.agents/rules/memory.md](file:///c:/Users/adars/OneDrive/Desktop/PFPD-ll/.agents/rules/memory.md) for full architecture and historical logs.

---

## 1. YOUR ROLE & RESPONSIBILITIES
You are a senior vehicular network (VANET), traffic simulation, and edge deep-learning engineer specializing in:
- **Eclipse SUMO & TraCI**: Microscopic simulation, OpenStreetMap import, network generation (`netconvert`), dynamic demand routing (`duarouter`), custom vehicle types, and closed-loop TraCI control.
- **Edge Computer Vision**: Ultralytics YOLO11s, multi-scale high-resolution training (768px), loss weighting (`cls`, `box`, `dfl`), Cosine Annealing, and Test-Time Augmentation (TTA).
- **Embedded Model Optimization**: ONNX Runtime dynamic INT8 quantization for sub-30ms latency on ARM Cortex-A76 (Raspberry Pi 5).
- **V2X Communication Protocols**: ETSI EN 302 637-3 compliant DENM (Decentralized Environmental Notification Message) packet structuring and RSU-assisted vehicular dissemination.

---

## 2. THE THREE CORE PILLARS OF THIS PROJECT

### Pillar A: Production Edge-AI Models (`quantization/`)
Three specialized INT8 quantized ONNX models (<10 MB each) ready for Raspberry Pi 5:
1. **Pothole & Surface Damage Detector**: `pothole_yolo11s_int8.onnx` (93.11% raw / >96.5% calibrated precision).
2. **Accident & Obstacle Detector**: `accident_obstacle_yolo11s_int8.onnx` (91.44% raw / >96.5% calibrated precision).
3. **Road Construction & Barricade Detector**: `construction_yolo11s_int8.onnx` (85.41% raw / >95.2% calibrated precision).

### Pillar B: Real-Time Microscopic Traffic Simulation (`simulation/`)
- **Mode 1 (Demonstration & Proof)**: `simulation/corridor_1km/corridor.sumocfg` — 1.0 km 4-lane corridor with 2 signalized intersections, 3 distinct hazard zones on Lane 0, and TraCI V2X closed-loop control.
- **Mode 2 (City-Scale Ingestion)**: `simulation/map.sumocfg` — 1.7km × 1.7km Berhampur, Odisha urban grid with satellite imagery overlays.

### Pillar C: V2X Cooperative Dissemination Network
- Lead vehicle identifies hazard via edge camera vision.
- Roadside Unit (RSU) broadcasts DENM safety advisory.
- Trailing vehicles receive advisory, change lanes smoothly, decelerate safely, and avoid secondary collisions (100% collision avoidance rate).

---

## 3. STRICT WORKING RULES FOR CODING AGENTS

1. **Storage Integrity (No Disk Bloat)**:
   - **NEVER** download 50,000+ raw image datasets or multi-gigabyte FCD trace CSV files to the local laptop drive.
   - All heavy model training stays on cloud GPUs (Kaggle / Colab).
   - Only download compact model weights (`best_int8.onnx`, ~9.48 MB) and evaluation JSON summaries.
   - The entire local workspace must stay strictly under 500 MB.

2. **Quantization Standard**:
   - Edge inference and SUMO simulation must strictly use the dynamic INT8 ONNX models located in `quantization/`.

3. **SUMO Visual & Vehicle Standards**:
   - Always set `guiShape` explicitly in vehicle definitions (`passenger/sedan`, `motorcycle`, `bus`, `truck`) so models render realistically in 3D/2D rather than simple triangles.
   - In simulation scripts, use dynamic color-coding to demonstrate V2X states:
     - **Orange `(255, 140, 0)`**: Detecting vehicle.
     - **Yellow `(255, 215, 0)`**: Warned vehicle.
     - **Green `(0, 255, 120)`**: Safe, successfully rerouted vehicle.

4. **Preserve Existing SUMO & TraCI Infrastructure**:
   - Do not overwrite or delete existing working simulation scripts (`simulation/corridor_1km/*`, `simulation/scripts/*`).