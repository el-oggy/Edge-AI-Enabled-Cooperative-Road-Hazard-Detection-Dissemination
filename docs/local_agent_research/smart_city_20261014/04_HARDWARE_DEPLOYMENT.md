# 4 — Hardware and deployment feasibility

Date: 15 October 2026. Host inspection is recorded in [HOST_PROJECT_AUDIT.json](HOST_PROJECT_AUDIT.json); transient available memory is a snapshot, not a fixed specification. No major engine was installed, no new performance trial was run, and no remote service/account/credential was used.

## Observed local environment

- Ryzen 5 5500U, six physical/twelve logical cores, confirmed through registry and psutil.
- Usable physical RAM: 7.38 GiB; about 1.30 GiB available and 82.4% used at the initial audit.
- Registry graphics: integrated AMD Radeon; 512 MiB reported dedicated memory. Shared memory and WebGL/API performance were not measured.
- Windows build 26300 platform string; it does not independently establish the Windows marketing edition. CIM queries were unavailable; registry/psutil supplied the evidence instead.
- About 159 GB decimal free disk (about 148 GiB). Executables absent from PATH do not prove applications are not installed elsewhere.
- Project Python 3.11.16 and ONNX Runtime 1.31.0. Available ORT providers: Azure and CPU; reviewed model sessions used CPU. The three graph hashes and static contracts matched; no new forwards in this research audit.

Recorded M1 per-image inference observations are about 2.24/2.03/2.01 s. Serial sweep sum 6.28 s; three-frame/all-model arithmetic 18.84 s/camera, excluding readback, queue, temporal, transport and control. At 40 km/h, 6.28 s is approximately 69.8 m of travel and 18.84 s is 209 m. These calculations explain delay risks; they are not throughput or safety measurements.

## Published baselines versus this host

| Stack | Published baseline | Engineering assessment |
|---|---|---|
| Three.js/WebGL2 | Browser/API-dependent, no universal scene RAM/FPS guarantee | Best first bounded local experiment; profile graphics capability, scene memory and readback |
| Godot Compatibility | Simple native editor: 4 GB minimum, 8 GB recommended RAM; compatible integrated graphics API | Plausible native alternative, but combined inference/scene load untested |
| Unity 6.0 | Editor guidance recommends at least 8 GB RAM and supported GPU/API drivers | Current available RAM leaves little editor/SUMO/ORT headroom; a scoped exported player must be assessed separately |
| Unreal editor | Recommended 32 GB RAM and 8 GB graphics memory | Remote/workstation development tier |
| CARLA UE4 0.9.16 | Dedicated RTX2070-equivalent or better, ≥8 GB VRAM in quickstart | Current integrated graphics does not match recommendation |
| CARLA UE5 | Recommended RTX3000+ with ≥16 GB VRAM; 130 GB storage, Windows11/Ubuntu22.04 | Not a local-first choice; exact package/map/import/bridge feasibility still needs a pilot |
| Blender | 8 GB minimum/32 GB recommended RAM and 2/8 GB graphics-memory figures | Small optional offline asset authoring; avoid concurrent heavy rendering during local runs |

Sources: [Godot requirements](https://docs.godotengine.org/en/stable/about/system_requirements.html), [Unity 6.0](https://docs.unity3d.com/6000.0/Documentation/Manual/system-requirements.html), [Epic hardware](https://dev.epicgames.com/documentation/en-us/unreal-engine/hardware-and-software-specifications-for-unreal-engine), [CARLA 0.9.16](https://carla.readthedocs.io/en/0.9.16/start_quickstart/), [CARLA UE5](https://carla-ue5.readthedocs.io/en/latest/start_quickstart/), [Blender](https://www.blender.org/download/requirements/). These requirements are not guaranteed scene FPS.

CARLA ordinary 0.9.16 and UE5 0.10.0 are separate release tracks. The UE5 0.10.0 announcement lists map/import limitations; later development may differ. Do not mix the older SUMO bridge's claims with a different latest package. Pin server, Python API, map/export pipeline and bridge together. [UE5 release announcement](https://carla.org/2024/12/19/release-0.10.0/), [CARLA release list](https://carla.readthedocs.io/en/0.9.16/download/).

## Execution choices

**Local first:** headless SUMO, small Python worker pool, browser 3D client and a selected sensor target. Reuse M1 venv without training packages; future renderer dependencies are project-local and pinned. Proposed art budget: instanced traffic/trees, modest textures, simple roofs, one limited shadow region, no mandatory ray tracing/water reflections. Test total process memory and camera readback concurrently with genuine forwards. Closing unrelated applications can increase headroom, but the proposal does not promise success.

**Hybrid:** local SUMO plus remote inference is the simplest offload; local render quality remains bounded. Send timestamped actual images and model hash expectation, receive predictions/service receipts, and measure roundtrip/queue delay. Local SUMO with remote renderer adds clock/acknowledgment and frame bandwidth complexity. For tight camera experiments, co-locating SUMO, rendering and inference remotely is cleaner; laptop receives presentation video and compact telemetry. That requires a graphics-capable remote environment, not merely a compute GPU allocation.

**Cloud/workstation:** richer Unity/CARLA scenes, more cameras and provider trials become plausible, but versions, graphics drivers, headless EGL/Vulkan/OpenGL, stream responsiveness, source artifact licensing and operator support need actual checks. No account access, free credits, capacity or region availability is presumed. A free notebook can support offline experiments but is not a guaranteed long-lived interactive server.

## Provider optimization without changing the models

Current CPU sessions are verified. DirectML is not installed here, is in sustained engineering and has execution/concurrency restrictions; Microsoft now points new work toward WinML. CUDA is also absent. A GPU provider can leave unsupported operators on CPU, and existing dynamic INT8 CNN graphs do not guarantee GPU acceleration. Use a separately approved environment and inspect node placement and complete timings; leave weights unchanged. [ORT DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html), [CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Bandwidth and illustrative costs

One 768×768 RGB uint8 frame is 1,769,472 bytes (1.6875 MiB). Two frames/s imply about 28.31 Mbit/s per camera before overhead; float32 input tensors are 6.75 MiB each. Fifty pose records at 10 Hz and an assumed 200 bytes/record give about 0.8 Mbit/s before serialization overhead. These are arithmetic planning estimates; measure actual serializer/compression and archive exactly the pixels consumed by the model.

| Public rate example observed by specialist | Displayed GPU hourly price | 10 compute hours | Caveat |
|---|---:|---:|---|
| Runpod A5000 24 GB | $0.27 | $2.70 | Extracted tier selector ambiguous; quotation/availability required |
| Runpod L4 24 GB | $0.59 | $5.90 | Compute GPU does not prove suitable interactive graphics stack |
| Runpod RTX4090 24 GB | $0.89 | $8.90 | Storage/tax/other charges additional |
| Paperspace A4000 | $0.76 | $7.60 | Plan restrictions/subscription/storage additional |
| Google T4 component | $0.35 | $3.50 GPU component | Excludes host VM/storage; not an all-in comparison |

[Runpod public pricing](https://www.runpod.io/pricing), [Paperspace pricing](https://docs.digitalocean.com/products/paperspace/pricing/), [Google GPU pricing](https://cloud.google.com/products/compute/gpus-pricing?hl=en). These are source-observed illustrative prices, not account quotes or guaranteed availability. The public Runpod page showed standard retained network storage at $0.07/GB/month below 1 TB; 150 GB arithmetic is $10.50/month. UE5's 130 GB base footprint leaves little headroom in that example volume. Stopped volumes may still incur cost. Requote the chosen region/tier and all storage before approval.

Open-source software cost can be zero while assets, hardware and compute still cost money. Three.js/Godot licenses are permissive; Unity Personal and Unreal eligibility/terms require the relevant context, particularly non-game industry usage. [Unity Personal](https://unity.com/products/unity-personal), [Unreal licensing](https://www.unrealengine.com/license), [Godot license](https://godotengine.org/license/). No paid account or licensing eligibility was assumed.

## Proposed resource gate

After approval, run a small daylit junction/road asset slice at a declared 1280×720 display resolution, one 768×768 sensor target, initially 10 then 50 SUMO-controlled actors, actual visible TLS and one genuine inference worker. Target 30 display FPS; provisional minimum 20 sustained display FPS, p95 presentation interval ≤50 ms and simulation real-time factor ≥0.9 over a declared 20-minute session. Log rendering CPU/GPU duration separately from presentation interval. No unbounded queue, memory growth or persistent paging is acceptable. Record RSS, RAM headroom, frame/readback timing, service delay and dropped captures. These are pass targets, not results. If the local gate fails, present the measured blocker and a scoped Unity/Godot/remote comparison before expanding the city.
