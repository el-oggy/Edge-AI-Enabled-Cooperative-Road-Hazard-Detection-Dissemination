# Edge-AI Road Hazard Perception: 24–30 FPS Optimization Roadmap
**Target Platform:** Raspberry Pi 5 (Broadcom BCM2712, Quad-Core Cortex-A76 @ 2.4 GHz, VideoCore VII GPU)  
**Budgetary Constraint:** $0.00 (Zero-Cost / Native Board Only — No External Hailo-8L NPU HATs or PCIe Accelerators)  
**Target Performance:** **24.0 to 30.0 FPS Continuous Perception** with $\ge 85\%$ Precision & Recall  
**Document Status:** Architectural Specification & Future Research Plan (Ready for Codex Enhancement)

---

## 1. Executive Summary & Problem Formulation

In the current baseline deployment on Raspberry Pi 5:
- 3 independent, heavy YOLO11s INT8 models are deployed ($768\times 768$ resolution, ~9.4M parameters, 28.6 GFLOPs each).
- Executing all 3 models per frame on 4 Cortex-A76 CPU threads consumes **~1,830 ms per frame**, yielding a brute-force throughput of **~0.55 FPS**.
- Deploying the **Robinson (Round-Robin) Multiplexer** with a 30-frame temporal TTL tracker reduces inference to 1 model per frame (~609 ms), boosting throughput to **~1.64 FPS** ($3\times$ gain).
- While Robinson multiplexing successfully stabilizes HUD advisories and avoids flickering, **1.64 FPS remains insufficient for highway-speed vehicle navigation (which requires $\ge 24\text{ FPS}$ to maintain $<1.5\text{ m}$ reaction latency at 60 km/h)**.

Because purchasing hardware accelerators (e.g., Hailo-8L AI Kit, Coral TPU) is ruled out due to project funding constraints, **the required $15\times$ to $18\times$ speedup must be achieved entirely through pure algorithmic, architectural, and software compilation optimizations on native Raspberry Pi 5 hardware.**

---

## 2. Baseline vs. Target Performance Gap Analysis

| Metric | Current Brute-Force | Current Robinson Engine | Target Native Goal (24–30 FPS) | Required Gap Factor |
| :--- | :---: | :---: | :---: | :---: |
| **Active Models per Pass** | 3 Separate Passes | 1 Pass (Multiplexed) | 1 Unified Model (All Hazards) | $3\times$ compute reduction |
| **Input Resolution** | $768 \times 768$ | $768 \times 768$ | $384 \times 384$ to $416 \times 416$ | $3.4\times$ to $4.0\times$ FLOP cut |
| **Model Size / Params** | Small (9.4M, 28.6 GF) | Small (9.4M, 28.6 GF) | Nano (2.6M, 6.4 GF) or Optimized S | $4.5\times$ FLOP reduction |
| **Compute Engine** | 4-Thread CPU (ONNX) | 4-Thread CPU (ONNX) | VideoCore VII (Vulkan NCNN) + NEON | Hardware offload |
| **Inference Latency** | ~1,830 ms | ~609 ms | **33.3 ms – 41.6 ms** | **$15\times$ to $18\times$ faster** |
| **Perception Rate** | 0.55 FPS | 1.64 FPS | **24.0 – 30.0 FPS** | **Real-Time Driving Safety** |

---

## 3. Four-Pillar Optimization Architecture (Zero-Cost Software Acceleration)

### Pillar 1: Model Consolidation (Single Unified Multi-Hazard Model)
* **The Bottleneck:** Currently, the system runs 3 separate neural networks:
  1. `pothole_yolo11s_int8.onnx`
  2. `construction_yolo11s_int8.onnx`
  3. `accident_obstacle_yolo11s_int8.onnx`
  Even under Robinson scheduling, each hazard class is only updated once every 3 frames ($1/3$ duty cycle).
* **The Solution:** Retrain into a single unified multi-class model with 3 output heads/classes:
  - Class 0: `Pothole`
  - Class 1: `Road Construction / Work Zone`
  - Class 2: `Accident / Road Obstacle`
* **Impact:** 1 single inference pass detects **all hazards simultaneously** in every single frame, instantly eliminating redundant backbone convolutions and delivering a **$3\times$ speedup**.

---

### Pillar 2: Dynamic Resolution Scaling ($768\times 768 \to 384\times 384$)
* **The Bottleneck:** Inference FLOPs in 2D convolutional networks scale **quadratically** with spatial resolution:
  $$\text{FLOPs} \propto W \times H$$
  At $768\times 768$, each image contains $589,824$ pixels.
* **The Solution:** Downscale the input dimension to $384\times 384$ ($147,456$ pixels) or $416\times 416$:
  $$\text{Scale Factor} = \left(\frac{384}{768}\right)^2 = \frac{1}{4} = 0.25$$
* **Impact:** Exactly **$4\times$ fewer matrix multiplication operations**. Latency on Cortex-A76 drops from ~600 ms down to **~150 ms** on CPU alone, without changing model weights.

---

### Pillar 3: Architecture Exploration (YOLO11s vs. YOLO11n Trade-Off)
* **The Bottleneck:** YOLO11s possesses 9.4M parameters. On embedded ARM CPUs, memory bandwidth and L2/L3 cache misses constrain throughput.
* **The Solution:** Evaluate YOLO11n (Nano architecture):
  - Parameters: 2.58M ($3.6\times$ lighter)
  - Theoretical FLOPs: 6.4 GFLOPs at $640\times 640$, dropping to **~2.3 GFLOPs at $384\times 384$**
* **Trade-Off Investigation (Research Required):**
  - Verify whether YOLO11n maintains acceptable recall ($\ge 80\%$) on small, distant potholes and dark asphalt cracks compared to YOLO11s.
  - Explore hybrid downscaled Small backbones with reduced width/depth multipliers (e.g., width=0.50, depth=0.33).

---

### Pillar 4: Embedded GPU Acceleration via VideoCore VII & Vulkan NCNN
* **The Hardware Opportunity:** Raspberry Pi 5 features an integrated **Broadcom VideoCore VII GPU** clocked at 800–900 MHz.
  - VideoCore VII supports **Vulkan 1.2** via the open-source Mesa `v3dv` driver.
  - While it lacks CUDA and official OpenCL runtimes, it provides compute shader pipelines through Vulkan SPIR-V.
* **The Software Pipeline:**
  1. Convert ONNX weights to **Tencent NCNN** (`.param` and `.bin` format).
  2. Enable NCNN Vulkan compute pipeline: `ncnn::create_gpu_instance()` and `net.opt.use_vulkan_compute = true`.
  3. Offload convolution and matrix multiply layers directly to the VideoCore VII GPU, freeing up the Cortex-A76 CPU cores for camera I/O, V2X messaging, and vehicle HUD rendering.
* **Expected Throughput:** At $384\times 384$, NCNN Vulkan on VideoCore VII is projected to deliver **22 to 32 FPS**.

---

### Pillar 5: Vectorized Compilation & Runtime Tuning (ARM NEON & OpenVINO)
* **ARM NEON SIMD:** Compile ONNX Runtime / NCNN with native ARMv8-A NEON 128-bit vectorization flags (`-march=armv8.2-a+fp16+dotprod -O3`).
* **Int8 Quantization Scheme:** Transition from symmetric min-max quantization to per-channel symmetric INT8 with calibrate-dataset quantization to preserve tiny pothole gradients.
* **OpenVINO ARM Plugin:** Benchmark Intel OpenVINO ARM runtime against ONNX Runtime CPU execution provider for optimized assembly kernel scheduling.

---

## 4. Cumulative Speedup Matrix (Compounding Gains)

| Optimization Step | Technique | Cumulative Latency | Cumulative FPS | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline** | 3 Separate Small Models ($768\times 768$, CPU) | 1,830 ms | 0.55 FPS | Implemented |
| **Step 1** | Robinson Round-Robin Multiplexer (Pi 5 CPU) | 609 ms | 1.64 FPS | Implemented & Verified |
| **Step 2** | Dynamic Resolution Scaling ($384\times 384$) | 152 ms | 6.5 FPS | Ready for Integration |
| **Step 3** | Single Unified Multi-Hazard Model (1 Pass) | 51 ms | 19.6 FPS | Phase 1 Research |
| **Step 4** | VideoCore VII Vulkan Offload (NCNN) + ARM NEON | **33 – 40 ms** | **25.0 – 30.0 FPS** | Phase 2 Research |

$$\mathbf{\text{Total Speedup: } \approx 18.3\times \text{ over Robinson, } 54.5\times \text{ over Brute-Force (Zero Hardware Cost)}}$$

---

## 5. Phased Research & Implementation Plan (For Future Codex Execution)

### Phase 1: Rapid Resolution & Input Optimization (Code Level)
- [ ] Implement multi-scale inference in `edge_continuous_camera_perception.py` allowing dynamic switching between $768\times 768$, $512\times 512$, and $384\times 384$.
- [ ] Measure mAP degradation curve on `data/benchmark_150` across the 3 resolutions.
- [ ] Profile memory bandwidth and CPU temperature on Pi 5 under continuous 10-minute loops.

### Phase 2: Unified Multi-Hazard Model Training
- [ ] Merge label manifests from `pothole`, `construction`, and `accident_obstacle` into a single combined multi-class dataset format.
- [ ] Train a unified model (`yolo11n_multihazard_384.pt` and `yolo11s_multihazard_384.pt`).
- [ ] Export to ONNX with dynamic batching and evaluate accuracy on test splits.

### Phase 3: NCNN Vulkan Embedded Deployment
- [ ] Install `libvulkan-dev` and Mesa Vulkan drivers (`mesa-vulkan-drivers`) on Raspberry Pi OS 64-bit.
- [ ] Build Tencent NCNN with Vulkan compute enabled (`-DNCNN_VULKAN=ON`).
- [ ] Convert the unified multi-hazard ONNX model into NCNN `.param` / `.bin`.
- [ ] Implement the C++ / Python NCNN inference wrapper and benchmark FPS on VideoCore VII.

### Phase 4: Field Validation & Driving Hazard Advisory
- [ ] Validate detection distance at 25–30 FPS (target: detect potholes at $\ge 15\text{ m}$ and construction zones at $\ge 35\text{ m}$).
- [ ] Confirm zero missed hazards in the 150-image benchmark suite.
- [ ] Package final deployment script for autonomous startup on Raspberry Pi 5.
