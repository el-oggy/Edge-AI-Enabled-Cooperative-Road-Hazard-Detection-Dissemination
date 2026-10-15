# PFPD: SUMO simulation research

Research date: **14 October 2026, Asia/Calcutta**. Status: **proposed architecture, awaiting user review; no simulation features implemented**.

Workspace: `C:\Users\adars\OneDrive\Desktop\PFPD-ll`  
Repository: [Edge-AI Enabled Cooperative Road Hazard Detection and Dissemination](https://github.com/el-oggy/Edge-AI-Enabled-Cooperative-Road-Hazard-Detection-Dissemination)  
Audited checkout: `main`, `34dad0a3d8683ebb9db548eb105604d8e29cb355`; tracked working tree was clean before research.

## 1. Decision

Start with **real road footage or images, genuine execution of the three existing ONNX models, explicit scenario-location mapping, and one Python TraCI controller controlling the existing 1 km corridor**. Prefer a fixed sensing RSU for the first integration. A moving lead vehicle requires additional pose/time alignment and is a later extension of the same architecture.

Support three clearly identified operating modes:

1. **Real inference during the run, slower than real time:** frames actually execute through ONNX Runtime; simulation playback may wait in wall time. Simulated eligibility still includes processing and delivery delay.
2. **Replay of genuine inference:** an immutable ledger produced by a documented real inference run drives paired traffic experiments. Show the original frames, predictions and provenance. Replay is not live inference.
3. **Mock perception:** structured events exercise communication/control failure cases. Label this mode prominently and prohibit it from satisfying real-model acceptance criteria.

CARLA is unnecessary for the first proof and poorly matched to the observed laptop resources. Keep it as an optional extension after perception, event timing and traffic evaluation work.

## 2. Evidence boundaries

This investigation used independent ONNX, SUMO/API and V2X/safety specialists; the principal agent reconciled their findings. The complete project memory was read and checked against deployed files. Architecture and codebase-discovery skill methods were applied, with their output adapted to the user's exact three-document limit.

| Evidence category | What it means here |
|---|---|
| Verified locally | Current source/XML inspection, file hashes, runtime metadata, in-memory inference and headless traffic probes |
| Historical record | A persisted report or telemetry file, whose measurement method and scope may be incomplete |
| External research | Official documentation, tagged original source, standards documents, or supplied research papers |
| Recommendation | A proposed engineering choice; not an implementation or successful experiment |
| Unresolved | Representative positive footage, model accuracy lineage, target-board performance and real radio operation |

Only the three requested Markdown documents were written. Models, scenario files, existing outputs and project memory were not modified. No dependencies, CARLA packages or datasets were installed/downloaded.

## 3. SUMO fundamentals relevant to this project

### Individual vehicles and road topology

SUMO is a microscopic simulator: individual road users follow car-following, lane-changing and junction models. `sumo` supplies headless execution; `sumo-gui` adds visual inspection. It does not produce a photorealistic dashcam feed suitable for these road-image detectors. Its normal visual scene is a representation of traffic state. [SUMO application](https://eclipse.dev/sumo/docs/sumo.html), [SUMO-GUI](https://eclipse.dev/sumo/docs/sumo-gui.html).

A network contains directed edges, lanes, junctions and lane-to-lane connections. Opposite directions use different edges. Lane index 0 is the rightmost lane in the direction of travel. Internal junction lanes and network coordinate offsets matter when calculating distances and placing overlays. Use the loaded network geometry rather than nominal road-node coordinates. [Road-network structure](https://eclipse.dev/sumo/docs/Networks/SUMO_Road_Networks.html).

### Configuration, demand and vehicle dynamics

A `.sumocfg` chooses network, routes, additional entities, timing and outputs. Routes list connected edge sequences; vehicles and flows supply demand; `vType` parameters describe dimensions, vehicle class and dynamics. Traffic lights arbitrate junction movements. `guiShape` changes appearance; it does not by itself define vehicle class or physical behavior. Inspect actual class permissions when considering a closure. [Vehicle, type and route definitions](https://eclipse.dev/sumo/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html).

Retain the corridor's **0.1 s simulation step** initially. Simulation step size, driver decision interval, GUI refresh rate and real wall-clock throughput are distinct. Pin SUMO version, route/configuration hashes, seeds and event schedule. Use explicit seeds rather than `--random`. A 0.1 s step does not demonstrate human reaction latency; driver response delays must be explicit experimental assumptions. [Basic definition](https://eclipse.dev/sumo/docs/Simulation/Basic_Definition.html), [randomness](https://eclipse.dev/sumo/docs/Simulation/Randomness.html), [safety and reaction timing](https://eclipse.dev/sumo/docs/Simulation/Safety.html).

### Hazards and restrictions

POIs and polygons are visual objects with no inherent obstacle interaction. SUMO also does not simulate pothole suspension damage. A pothole study needs a separately defined lane-region exposure/speed metric. For workzones or accident obstructions, a stopped vehicle can supply a physical traffic obstacle. Keep its existence independent of whether AI detects it. [Shapes](https://eclipse.dev/sumo/docs/Simulation/Shapes.html), [SUMO accident FAQ](https://eclipse.dev/sumo/docs/FAQ.html#how_to_simulate_an_accident).

Lane permission changes affect an entire lane. Closing `edge_2_0` changes roughly 485.6 m of roadway, not just the painted workzone. A localized blocker plus a visible workzone is the first preserved-network experiment. A true short segment closure requires an approved derivative network with split edges. Native closures can change baseline behavior themselves; both comparison arms must share the same physical restriction. [Rerouters and closures](https://eclipse.dev/sumo/docs/Simulation/Rerouter.html), [lane state changes](https://eclipse.dev/sumo/docs/TraCI/Change_Lane_State.html).

## 4. Current environment and scenario

### Runtime and hardware

| Item | Current observation |
|---|---|
| Project Python | `.venv\Scripts\python.exe`, Python 3.11.16 |
| Default PATH Python | `C:\iverilog\gtkwave\bin\python.exe`; unsuitable as an implicit project interpreter |
| SUMO binary | `C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe`, version **1.27.1** |
| Wheel packages | `traci`/`sumolib` 1.28.0 in `.venv`; use the installed binary's `tools` path first to match 1.27.1 |
| Inference dependencies | ONNX Runtime 1.31.0, NumPy 2.4.6, OpenCV headless 5.0.0.93, psutil 7.2.2 |
| Absent project packages | `onnx`, `ultralytics`, `torch`, `libsumo`, `pytest`; these absences do not prevent direct ORT inference |
| CPU | AMD Ryzen 5 5500U, 6 physical / 12 logical cores |
| RAM | 7,928,741,888 bytes, about **7.38 GiB**; about 0.9 GiB available at one observation |
| GPU evidence | Registry reports AMD Radeon(TM) Graphics; no NVIDIA adapter/tool was found by the probes; dedicated VRAM was not measured |
| Storage | Approximately 166.7 GB free on C: at audit time |

The current official SUMO changelog lists **1.28.0, released 8 October 2026**. It includes dynamic lane-permission fixes relevant to this design. Do not silently upgrade the baseline; test observed restriction behavior on 1.27.1 and request an upgrade if required. [SUMO changelog](https://eclipse.dev/sumo/docs/ChangeLog.html).

### What the preserved corridor supplies

- A nominal 1 km arterial, two lanes per direction, signals at nominal x=300/800 m, opposing/cross traffic, connected cars, buses and trucks.
- Deployed continuous flows and a 3,600 s configuration, native environment polygons, three hazard overlays, an RSU marker and saved viewsettings.
- A controller with position-triggered hazard registration, JSON-like warnings, vehicle colors, lane-change/speed commands and a telemetry writer.
- A reproducible starting network. It has no inspected forward bypass around its middle edge.

The current `build_corridor.py` contains older templates. Executing it would overwrite deployed files with an earlier, simpler scenario. Preserve the deployed XML as the starting point.

The controller does **not** execute ONNX inference. Printed confidence/latency/model names are constants. Receiver selection is confined to the same edge and hazard lane within the 50–60 m detection window; the memory's 250 m warning-radius claim does not match this implementation. Its green-state predicate can fire before crossing; acceleration is sampled every second but divided by 0.1 s; logged time is shifted by one step. These need targeted remediation after approval.

Authoritative locations are `edge_2_0` at lane positions 250, 350 and 430 m. A live `simulation.convert2D` probe returned x=557.2, 657.2 and 737.2 m, all y=95.2. Existing drawings differ. The lane starts at x=307.2, has length 485.6 m and includes a network offset; do not map locations by adding 300 to lane position.

### Execution evidence and limits

A direct, output-free TraCI probe used SUMO 1.27.1, the unchanged corridor config, seed 42, strict route-error checking, 0.1 s steps and a 120 s horizon. It completed **1,200 steps, 56 departed/unique vehicles, 29 arrivals and a maximum of 29 active vehicles**. Colliding-vehicle and emergency-stop ID queries were empty. This was a **traffic-only check without the cooperative controller**, and supplies no warning-efficacy result.

The existing corridor CSV contains 17,352 records for 306 vehicle IDs over approximately 671 s. It establishes prior trajectory logging, but lacks paired experiments, warning causality and reliable acceleration. Available FCD/tripinfo files examined by the safety audit contain empty roots; one SSM file is malformed and the duplicate is empty of conflicts. The memory's 100% collision-avoidance and 60+ GUI FPS claims are not established by these artifacts. GUI rendering/performance was not tested during this task.

## 5. TraCI control architecture and API inventory

Use **one simulation owner** to launch, step, read and command SUMO. Inference workers communicate through data/event interfaces; they do not open another TraCI control client.

The following signatures were inspected in the installed 1.27.1 Python source. This is a verified interface inventory, not implemented integration code:

| Purpose | Relevant methods |
|---|---|
| Lifecycle | `traci.start(command_list)`, `traci.simulationStep()`, `traci.getVersion()`, `traci.close()` |
| Clock | `simulation.getTime()`, `simulation.getDeltaT()` |
| Vehicle state | `vehicle.getPosition(id)`, `getSpeed(id)`, `getAcceleration(id)`, `getAngle(id)`, `getRoadID(id)`, `getLaneID(id)`, `getLaneIndex(id)`, `getLanePosition(id)`, `getRoute(id)`, `getRouteIndex(id)` |
| Route distance | `vehicle.getDrivingDistance(id, edgeID, pos, laneIndex=0)` |
| Advisory | `vehicle.slowDown(id, speed, duration)`, `vehicle.changeLane(id, laneIndex, duration)`, `vehicle.setSpeed(id, -1)` to release explicit speed control |
| Maneuver checks | `vehicle.couldChangeLane(id, direction, state=None)`, `getLaneChangeState`, adjacent leader/follower queries |
| Rerouting | `simulation.findRoute(...)`, `vehicle.rerouteTraveltime(id, currentTravelTimes=True)`, `vehicle.setRoute(id, edgeList)` |
| Road mutation | `lane.getAllowed`, `getDisallowed`, `getMaxSpeed`, `setAllowed`, `setDisallowed`, `setMaxSpeed`; `vehicle.updateBestLanes(id)` |
| Coordinates | `simulation.convert2D(edgeID, pos, laneIndex=0, toGeo=False)` |
| Safety status | `simulation.getCollidingVehiclesIDList()`, `getEmergencyStoppingVehiclesIDList()` |

Sources: [Python interface](https://eclipse.dev/sumo/docs/TraCI/Interfacing_TraCI_from_Python.html), [vehicle retrieval](https://eclipse.dev/sumo/docs/TraCI/Vehicle_Value_Retrieval.html), [vehicle changes](https://eclipse.dev/sumo/docs/TraCI/Change_Vehicle_State.html), [simulation retrieval](https://eclipse.dev/sumo/docs/TraCI/Simulation_Value_Retrieval.html), [installed-version vehicle source](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/tools/traci/_vehicle.py).

Subscribe newly departed vehicles to relevant variables and read subscription results after each step; clean up vehicles that leave. Context subscriptions can reduce neighborhood queries but do not replace route/direction filtering. `simulationStep()` does not return the subscription data. A step listener runs per API call, not necessarily per internal step; use explicit one-step orchestration initially. [Object subscriptions](https://eclipse.dev/sumo/docs/TraCI/Object_Variable_Subscription.html), [context subscriptions](https://eclipse.dev/sumo/docs/TraCI/Object_Context_Subscription.html).

Multiple clients require a configured client count, connection barriers, ordered clients and coordinated stepping. This is unnecessary infrastructure for inference integration. `libsumo` can reduce socket overhead for future headless batches, but its GUI is unsupported on Windows. Keep TraCI as the presentation backend. [TraCI protocol](https://eclipse.dev/sumo/docs/TraCI/Protocol.html), [libsumo](https://eclipse.dev/sumo/docs/Libsumo.html).

Preserve default safety modes, normally speed mode 31 and lane-change mode 1621, while recording the actual values. `couldChangeLane` is advisory and a lane request is not evidence of completion. Check lane existence/permissions/route continuation, request a safe maneuver, then log observed state. Convert km/h to m/s. If a merge cannot complete, maintain a safe slowdown/stop policy. Never disable safety checks to create a desirable demonstration.

## 6. Approach comparison

This table is an engineering assessment, not a measured schedule or cost quotation.

| Criterion | A: real footage + SUMO | B: CARLA + SUMO | C: structured event tests |
|---|---|---|---|
| Complexity | Moderate: inference, explicit mapping and clocks | High: two engines, map/actor/sensor/version alignment | Low |
| Hardware | Existing CPU, accepting slow playback | Rendering GPU, more RAM, larger storage | Minimal |
| Scientific scope | Real visual perception plus simulated traffic; manual spatial mapping disclosed | Synthetic camera perception plus simulated traffic; domain shift remains | Communication/control logic with mock perception, or labelled inference replay |
| Trained-model use | Direct genuine execution | Genuine execution on synthetic images | Absent for mock events |
| Realism | Real appearance, abstract traffic environment | Rich 3D appearance if hazards/assets are credible | Abstract events |
| Latency | Measurable; current laptop is far from video rate | Inference plus rendering/sensor/bridge delay | Configurable transport/control delay |
| Reproducibility | Strong with input/model/ledger hashes | Strong only with synchronized ticks and version/asset pinning | Strong |
| Presentation | Clear causal three-panel demonstration | Impressive after significant engineering | Useful testing, limited AI evidence |
| Development scope | Bounded first milestone and existing corridor reuse | Additional project-sized integration | Shortest control-test path |
| Existing-code compatibility | Highest | Needs a matching rendered world | High |

## 7. Real inference timing, mapping and transport

All three graphs accept static float32 `[1,3,768,768]` and return float32 `[1,5,12096]`. Their class 0 metadata and weights differ. Current real-image smoke measurements on this laptop took about **3.8–5.4 s per model**; a three-worker triplet averaged about **7.8 s**. These are small, noisy measurements, not production benchmarks. Detection accuracy is unresolved; some sample predictions are visibly inconsistent with the claimed hazard.

Consequently, a 35 FPS three-model live system is not an established starting capability. A 50 m lead-vehicle trigger at 50 km/h leaves only about 3.6 s before the hazard. Slow three-model inference and temporal confirmation cannot fit that assumption. Use a stationary source and explicitly declared early observation for the first presentation, then test late observations and failures separately.

Manual fixture mapping is acceptable for a proof of integration: a clip is explicitly associated with an edge/lane/position. It does not mean YOLO estimated GPS position. Future calibrated road-plane projection needs camera pose/intrinsics and geometric validation; moving cameras additionally need synchronized ego poses. Keep camera pixel coordinates and SUMO metres separate.

Use source-frame timestamps, a monotonic wall timer for latency, and SUMO's actual clock. For deterministic service replay, compute `start=max(capture_sim_time, previous_worker_finish)`, then `finish=start+measured_service_duration`. Confirmation is available after the final distinct qualifying frame finishes; add communication delay once. Never apply an event before that eligibility time. Wall-clock pauses do not make processing instantaneous in a safety experiment. [Python clock semantics](https://docs.python.org/3.11/library/time.html).

Start with synchronous calls for the first smoke milestone, then a single inference worker and bounded `queue.Queue` for live integration. A queue of latest frames prevents unbounded backlog; event delivery must log overflow rather than silently discard confirmed hazards. Python multiprocessing can isolate heavy work but brings Windows spawn/serialization overhead. TCP becomes useful when a physical RSU/Pi is on another machine. ZeroMQ and MQTT offer broader transports but add dependencies/brokers that the local prototype does not yet need. [Python 3.11 queues](https://docs.python.org/3.11/library/queue.html), [multiprocessing](https://docs.python.org/3/library/multiprocessing.html).

Quantized file size does not establish speed or accuracy. ONNX Runtime distinguishes dynamic and static quantization and requires performance/accuracy assessment on the target platform. The inspected graphs contain dynamic integer convolution operators; no model conversion is authorized in this research task. [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [Ultralytics export options](https://docs.ultralytics.com/modes/export/).

## 8. Native visualization and the companion display decision

Use the existing corridor environment. Dynamically show a confirmed hazard at geometry-derived coordinates, a labelled assumed RSU coverage polygon, recipient highlights and states tied to observed actions. Retain grey/default traffic, cyan sensing source, amber warned vehicles, and green only after a verified crossing; a rejected/pending merge remains visibly pending. Do not label acknowledgement or reception as physical radio evidence.

Installed methods include `vehicle.setColor`, `vehicle.highlight`, `poi.add/setColor/setPosition/remove`, `polygon.add/setShape/setColor/remove`, `gui.trackVehicle`, `gui.setZoom/setOffset/setBoundary`, and `gui.screenshot(viewID, filename, width=-1, height=-1)`. Use `View #0` unless the GUI supplies another ID. A snapshot request is fulfilled on a subsequent simulation step; verify format/rendering on Windows before recording a presentation. [GUI reference](https://eclipse.dev/sumo/docs/TraCI/GUI_Value_Retrieval.html), [GUI changes](https://eclipse.dev/sumo/docs/TraCI/Change_GUI_State.html), [SUMO-GUI snapshots](https://eclipse.dev/sumo/docs/sumo-gui.html).

The useful initial layout is **annotated road frame + SUMO-GUI + compact local telemetry**. OpenCV headless cannot create GUI windows; use a small standard-library Tkinter presentation adapter if available, or an optional existing GUI dependency only after it is checked. Show image/model identity, timestamps, inference duration, event ID, recipients, commanded versus observed motion, and mode labels. Do not introduce a frontend framework now. A read-only dashboard becomes justified if faculty cannot follow three adjacent surfaces; it consumes the same telemetry and never owns TraCI.

Before/after experiments can replay the same inference ledger, traffic seed and viewsettings. Screenshots and recordings illustrate behavior; event logs and SUMO outputs substantiate it.

## 9. CARLA feasibility and future integration

Official co-simulation provides actor synchronization between matching CARLA and SUMO worlds. A future implementation needs a CARLA OpenDRIVE map, a compatible SUMO network/vehicle mapping, one traffic-light authority and one synchronized tick owner. The existing SUMO corridor does not automatically become a rendered CARLA road. [Official co-simulation guide](https://carla.readthedocs.io/en/latest/adv_sumo/).

Use tagged bridge code for coordinate offset, handedness, yaw and actor-origin conversion; do not invent a simple x/y offset. Pin server, Python API, co-simulation scripts and SUMO together. CARLA camera records include frame/time/transform; GPU sensor results can lag ticks, so match frame IDs and fixed-delta synchronous operation. Hazard meshes and their camera visibility must be validated independently of detection. [0.9.16 synchronization source](https://raw.githubusercontent.com/carla-simulator/carla/0.9.16/Co-Simulation/Sumo/run_synchronization.py), [coordinate bridge](https://raw.githubusercontent.com/carla-simulator/carla/0.9.16/Co-Simulation/Sumo/sumo_integration/bridge_helper.py), [synchronous sensors](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/), [RGB sensors](https://carla.readthedocs.io/en/latest/ref_sensors/).

Version-specific requirements differ. The UE4 quickstart recommends a dedicated RTX 2070-equivalent GPU with at least 8 GB VRAM and roughly 20 GB storage. The UE5 quickstart recommends RTX 3000-series or better with at least 16 GB VRAM and roughly 130 GB storage. Official releases include the 0.9.16 and separate 0.10.0 lines, while some quickstart text still calls 0.9.15 latest. Recheck a chosen tagged release before installation. [UE4 quickstart](https://carla.readthedocs.io/en/latest/start_quickstart/), [UE5 quickstart](https://carla-ue5.readthedocs.io/en/latest/start_quickstart/), [official releases](https://github.com/carla-simulator/carla/releases).

The observed integrated Radeon system with about 7.38 GiB RAM does not meet the cited dedicated-GPU guidance. Storage alone cannot resolve that limitation. CARLA compatibility and performance were not tested. Postpone installation; no purchase, cloud GPU spending or large package download is authorized.

## 10. Safety research and standards qualification

Use native collision, lane-change, trip and SSM outputs. A zero-collision baseline is expected under normal safe SUMO behavior and is still informative: compare hazard-entry speed/exposure, braking, merge lead distance, gap quality, warning timeliness and traffic delay. TTC concerns closing conflicts, not arbitrary distance to a drawn pothole; call the latter time-to-hazard. Report undefined measures explicitly. [Collision output](https://eclipse.dev/sumo/docs/Simulation/Output/Collisions.html), [lane changes](https://eclipse.dev/sumo/docs/Simulation/Output/Lanechange.html), [trip information](https://eclipse.dev/sumo/docs/Simulation/Output/TripInfo.html), [SSM](https://eclipse.dev/sumo/docs/Simulation/Output/SSM_Device.html).

The existing JSON is **DENM-inspired application data**, not a standards-compliant message or radio transmission. Existing cause codes 95/96/97 are incorrect for the named hazards. The implementation plan pins an authoritative CDD reference and separates accident, roadworks, adverse surface and obstacle semantics. A broad accident/obstacle detector cannot infer a specific accident subtype merely from class 0.

Supplied literature was inspected for context: Gong et al., *Edge Intelligence in Intelligent Transportation Systems: A Survey*, IEEE T-ITS (2023), DOI [10.1109/TITS.2023.3275741](https://doi.org/10.1109/TITS.2023.3275741), and Kandali et al., *A Comprehensive Survey on VANET–IoT Integration Toward the Internet of Vehicles* (2026), DOI [10.3390/futuretransp6010032](https://doi.org/10.3390/futuretransp6010032). These support research context, not this project's performance. Publisher web access was restricted; local supplied PDFs provided the inspected text.

## 11. Next handoff

The first implementation milestone is **repair and verify the real inference path only**: valid model defaults, shape-derived preprocessing, correct specialist semantics, NMS/coordinate decoding, actual `--image` execution, honest benchmark output and distinct-frame temporal tests. Construction/accident positive fixtures and videos remain missing. Do not advance to a convincing three-hazard traffic demonstration by replacing missing visual evidence with scripted detections.

Read the companion implementation plan for the exact schema, scenarios, acceptance gates and risk register; the standalone subagent system prompt contains the bounded future engineering instructions.
