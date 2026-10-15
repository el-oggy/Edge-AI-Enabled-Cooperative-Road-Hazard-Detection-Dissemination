# PFPD rendering and visual simulation research

Research date: 2026-10-14. Research and design only. No renderer was installed, no simulation was run, and no source, network, model, project memory, or Git history was changed by this specialist. The city concepts below are original proposals, not functioning simulations or measured results.

## Recommendation

Build a polished browser 3D environment with **SUMO as the traffic authority, Three.js as the initial renderer, Blender as an optional asset authoring tool, and the existing Python ONNX inference service**. Keep a small renderer adapter boundary so that Unity URP can replace the visual client later if the approved visual target cannot be achieved locally. Babylon.js is a credible alternative to Three.js; choose one browser engine rather than developing both. Native SUMO-GUI remains valuable for traffic debugging.

This recommendation is engineering judgment based on the current memory-constrained integrated-graphics host, inspectable browser/SUMO precedent, and the requirement for genuine model causality. It does not establish local frame rate. The initial browser scene must contain textured asphalt, proper junction geometry, curbs, markings, buildings, vegetation, signal heads, diverse vehicle meshes, RSUs, and vehicle-level cameras. A flat infographic would fail the visual requirement.

Three.js has a WebGL2 renderer and an instancing API for repeated geometry; neither supplies traffic behavior. The archived SUMO-Web3D implementation demonstrates the proposed separation using TraCI, WebSocket state updates and a Three.js client. Its age makes it a reference rather than a dependency to install unchanged. [Three.js renderer](https://threejs.org/docs/pages/WebGLRenderer.html), [InstancedMesh](https://threejs.org/docs/pages/InstancedMesh.html), [SUMO-Web3D](https://github.com/sidewalklabs/sumo-web3d).

**Unity is the strongest alternative for scene editing and richer presentation; CARLA is the stronger eventual sensor-simulation extension on suitable remote hardware.** They are different choices: a general engine requires a custom scientific adapter, while CARLA supplies an autonomous-driving simulator and a documented SUMO bridge. No reviewed source establishes that the existing PFPD network will automatically become a matching Unity or CARLA scene.

## Existing project and hardware context

The current local [M1 report](../M1_IMPLEMENTATION_REPORT.md) records genuine forwards on three distinct reviewed models with static float32 input `[1,3,768,768]` and output `[1,5,12096]`. Its final single-image CPU inference times were approximately 2.24, 2.03 and 2.01 seconds. That is execution evidence, not throughput statistics, detector accuracy, or synthetic-frame recognition. The supplied positives remain unannotated, and clean negatives and timestamped temporal clips are missing. This research leaves the M1 work intact.

The coordinator's current read-only hardware inspection reports Ryzen 5 5500U, six physical/twelve logical cores, 7.38 GiB usable physical RAM, integrated Radeon graphics, and about 1.30 GiB available RAM at one inspection. Registry dedicated graphics memory is 512 MiB; shared-memory capacity and graphics API capability are not established by that value. The available disk snapshot was about 159 GB. These are host observations, not renderer benchmarks. Hardware should be recorded again before any approved feasibility trial because load and available memory change.

The repository file inventory contains SUMO networks and the existing `simulation/corridor_1km/`; it did not reveal a checked-in Unity, Godot, or browser 3D project in the paths searched. Existing SUMO traffic and historical source documentation are useful baselines, while their prior choice of a simple GUI is superseded by the new visual brief. The stale corridor generator must remain untouched.

## Technology comparison

The assessments in the last column are recommendations, not measured laptop results. All candidates need geometry, camera visibility and performance validation; “50 vehicles” alone says little about texture memory, shadows, foliage, sensor passes or physics load.

| Candidate | Source-supported capabilities and constraints | Fit for this project |
|---|---|---|
| SUMO-GUI, normal 2D | Traffic visualization, inspection and snapshots in the same application as SUMO. [Official GUI documentation](https://eclipse.dev/sumo/docs/sumo-gui.html). | Keep as traffic-debug view. It does not meet the requested driving-scene presentation or validate road-image perception. |
| SUMO-GUI + OpenSceneGraph | Generates roads, curbs, junctions, crossings and signals; permits static/vehicle models and ego camera navigation. Official documentation calls it experimental, lists missing pedestrian/cyclist models, performance problems with several vehicles and reload limitations. Windows OSG is in the extra build. [GUI](https://eclipse.dev/sumo/docs/sumo-gui.html), [Downloads](https://eclipse.dev/sumo/docs/Downloads.html). | Useful bounded comparison, weaker final choice. Do not promise 50-agent performance from its built-in 3D option. The installed binary's OSG support is unverified. |
| SUMO + Three.js | WebGL2 scene rendering, repeated-mesh instancing, custom cameras, materials and render targets; SUMO-Web3D provides an inspectable precedent for pose/signal streaming. [Renderer](https://threejs.org/docs/pages/WebGLRenderer.html), [Instancing](https://threejs.org/docs/pages/InstancedMesh.html). | Preferred local feasibility candidate. Low installation burden compared with major editors; custom road generation, asset composition, capture and synchronization remain engineering work. No official universal RAM/FPS minimum for this scene. |
| SUMO + Babylon.js | Browser game/rendering engine; official repository documents WebGL/WebGPU, glTF sandbox and modular npm packages. Scene optimization guidance exists, and texture readback APIs are asynchronous. [Repository](https://github.com/BabylonJS/Babylon.js), [Optimization](https://doc.babylonjs.com/features/featuresDeepDive/scene/optimize_your_scene/), [Readback changes](https://github.com/BabylonJS/Documentation/blob/master/content/setup/support/webGPU/webGPUBreakingChanges.md). | Comparable local option with more built-in tooling. No reviewed current SUMO-specific bridge was established; same camera/domain validation burden. Prefer WebGL capability baseline over assuming WebGPU support. |
| SUMO + Unity URP | Windows Unity 6.0 editor guidance recommends at least 8 GB RAM and a supported DX10/11/12 or Vulkan GPU/driver. Cameras can render to textures; async GPU readback can return texture data. SUMO2Unity contains an importer and Unity integration scripts. [Requirements](https://docs.unity3d.com/6000.0/Documentation/Manual/system-requirements.html), [Camera target](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/camera/targettexture), [GPU readback](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Rendering.AsyncGPUReadback.Request.html). | Strongest general-engine alternate. Editor plus Python/SUMO/browser on a 7.38 GiB host is a resource concern. A standalone player with baked lighting may fit better than editor playback; this is untested. |
| SUMO + Unreal Engine | SceneCapture2D writes camera views to render targets; spline meshes support road-like geometry. Current hardware guidance recommends 32 GB RAM, 8 GB graphics memory and DX12 graphics. [Rendering components](https://dev.epicgames.com/documentation/unreal-engine/rendering-components-in-unreal-engine), [Hardware](https://dev.epicgames.com/documentation/unreal-engine/hardware-and-software-specifications-for-unreal-engine). | Excellent visual ceiling, substantial editor/build and bridge burden. No primary source inspected here establishes a maintained standalone Unreal–SUMO adapter preferable to CARLA. Defer to remote workstation tier. |
| CARLA UE4 branch + SUMO | Versioned 0.9.16 guidance recommends dedicated GPU comparable to RTX2070+, at least 8 GB VRAM and about 20 GB disk. The documented bridge supports matched OpenDRIVE/SUMO maps and a selected signal owner. RGB sensors record images. [Requirements](https://carla.readthedocs.io/en/0.9.16/start_quickstart/), [Bridge](https://carla.readthedocs.io/en/0.9.15/adv_sumo/), [Sensors](https://carla.readthedocs.io/en/0.9.16/ref_sensors/). | Mature sensor and traffic co-simulation reference; unsuitable for the observed local graphics resources. Pin server/API/bridge together. A 0.9.16 page still contains older 0.9.15 examples, so verify tagged releases before selection. |
| CARLA UE5 branch + SUMO | UE5 package guide specifies Windows11/Ubuntu22.04, recommended RTX3000-series or better with at least 16 GB VRAM, and 130 GB storage; it illustrates 0.10.0 packages. [UE5 quick start](https://carla-ue5.readthedocs.io/en/latest/start_quickstart/). | Remote high-quality tier. Do not apply legacy UE4 requirements to this branch or assume bridge compatibility from the older documentation. Custom hazard/map authoring may require a source build. |
| Godot 4 Compatibility + SUMO | Native editor baseline 4 GB RAM, recommendation 8 GB; Compatibility editor graphics require OpenGL3.3. A viewport can render its Camera3D to a texture. [Requirements](https://docs.godotengine.org/en/stable/about/system_requirements.html), [Viewport](https://docs.godotengine.org/en/stable/classes/class_viewport.html). | Good lightweight native alternative, permissive license and editable scenes. No inspected ready-made SUMO bridge here; importing geometry and maintaining a custom adapter reduce its advantage over browser stack. |
| Blender | Official requirements list 8 GB minimum/32 GB recommended RAM and 2 GB minimum/8 GB recommended graphics memory for desktop use. glTF supports mesh/PBR exchange. [Requirements](https://www.blender.org/download/requirements/), [glTF specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.pdf). | Author small reusable road furniture, hazard meshes, RSUs and vehicle LODs. Also useful for offline films. Avoid running heavy authoring/rendering concurrently with the live local demo. Blender animation alone is not live cooperative traffic. |
| sumo3Dviz / Panda3D | A recent research tool renders SUMO logs in external, driver and cinematic views. Inspected source separates trajectory replay from interactive static-world navigation. [Paper](https://arxiv.org/abs/2604.19194), [Repository](https://github.com/DerKevinRiehl/sumo3dviz). | Useful offline/replay reference and possible presentation exporter. It is not an inspected live TraCI/inference client. Dependency isolation and live adapter work are required. |

Software-license cost and compute cost differ. Three.js is MIT, Babylon.js Apache2.0, Godot MIT and sumo3Dviz GPL3.0 in their primary repositories/license pages. Unity Personal and Unreal publish eligibility-dependent licensing; do not assume eligibility or account access. Asset terms are separate. None of these pages provides the operational cost of a PFPD run; remote GPU quotations and bandwidth need the hardware/deployment study. [Three.js](https://github.com/mrdoob/three.js), [Babylon.js](https://github.com/BabylonJS/Babylon.js), [Godot license](https://godotengine.org/license/), [Unity Personal](https://unity.com/products/unity-personal), [Unreal licensing](https://www.unrealengine.com/license).

## Inspectable projects and useful visual references

The source observations below establish published code behavior or documentation. **No linked YouTube footage was watched or runtime demo benchmarked in this assignment.** A linked demo is a visual reference to review, not evidence of this laptop's performance or this project's detector success.

### 1. Sidewalk Labs SUMO-Web3D

[Repository](https://github.com/sidewalklabs/sumo-web3d), inspected commit `d89fe8014879153d30632996f35b68efb4d8c086`.

The [Python server](https://raw.githubusercontent.com/sidewalklabs/sumo-web3d/d89fe8014879153d30632996f35b68efb4d8c086/sumo_web3d/server/server.py) actually starts SUMO, calls `traci.simulationStep`, subscribes to vehicles and lights, and sends state differences over WebSocket. [Signal rendering](https://raw.githubusercontent.com/sidewalklabs/sumo-web3d/d89fe8014879153d30632996f35b68efb4d8c086/frontend/src/traffic-lights.ts) maps connection link indices to meshes and updates materials from phase state. These are reusable architecture patterns.

The repository was archived in April2023. Its [package manifest](https://raw.githubusercontent.com/sidewalklabs/sumo-web3d/d89fe8014879153d30632996f35b68efb4d8c086/package.json) uses Three.js `^0.87.1`, React15 and webpack1. Do not drop its historical package into the current project, or let its server become a second TraCI owner. Source does not establish genuine YOLO perception or photorealism.

### 2. SUMO2Unity, SimuTraffX Lab

[Repository and research links](https://github.com/SimuTraffX-Lab/SUMO2Unity), inspected commit `4379205441d2d17e58c2288b9373b8cdcb20ec1f`. README targets SUMO1.22 and Unity6000.0.53f1 and describes 0.10-second pose/signal exchange. It explicitly lacks pedestrians. Code is described as MIT and assets CC-BY; asset attribution must be retained.

The [editor version](https://raw.githubusercontent.com/SimuTraffX-Lab/SUMO2Unity/4379205441d2d17e58c2288b9373b8cdcb20ec1f/ProjectSettings/ProjectVersion.txt) and [URP17.0.4 manifest](https://raw.githubusercontent.com/SimuTraffX-Lab/SUMO2Unity/4379205441d2d17e58c2288b9373b8cdcb20ec1f/Packages/manifest.json) confirm that stack. The [road importer](https://raw.githubusercontent.com/SimuTraffX-Lab/SUMO2Unity/4379205441d2d17e58c2288b9373b8cdcb20ec1f/Assets/_Project/Scripts/RoadNetworkScripts/RoadNetworkBuilder.cs) reads SUMO lane widths/shapes, creates lane and junction meshes, applies marking decals, and uses a local origin. This is useful for geometry generation; its mapping flattens roads into Unity's XZ plane, so it does not prove elevation fidelity.

The [communication script](https://raw.githubusercontent.com/SimuTraffX-Lab/SUMO2Unity/4379205441d2d17e58c2288b9373b8cdcb20ec1f/Assets/_Project/Scripts/IntegrationScripts/ExchangeData.cs) uses NetMQ subscriber/dealer sockets and a main-thread queue. The [controller](https://raw.githubusercontent.com/SimuTraffX-Lab/SUMO2Unity/4379205441d2d17e58c2288b9373b8cdcb20ec1f/Assets/_Project/Scripts/IntegrationScripts/SimulationController.cs) handles JSON vehicle poses/types and individual signal characters. Its ego-vehicle return path and [dynamic rigidbody vehicle controller](https://raw.githubusercontent.com/SimuTraffX-Lab/SUMO2Unity/4379205441d2d17e58c2288b9373b8cdcb20ec1f/Assets/_Project/Scripts/IntegrationScripts/VehicleController.cs) need authority review before reuse in PFPD.

**Audit limitation:** the inspected [Python file](https://raw.githubusercontent.com/SimuTraffX-Lab/SUMO2Unity/4379205441d2d17e58c2288b9373b8cdcb20ec1f/RequiredFiles/Sumo2UnityTool_combined.py) is only a header/version/link stub, while the repository tree contains a roughly31.4 MB `Sumo2UnityTool.exe`. This does not provide an inspectable Python bridge counterpart. No binary was downloaded or executed. Therefore do not call the full bridge source verified or select the executable as the experiment's traffic authority.

### 3. Sumonity, TUM traffic engineering

[Repository](https://github.com/TUM-VT/Sumonity), [base project](https://github.com/TUM-VT/Sumonity-UnityBaseProject). Its README describes a Python/Unity bridge with PID/pure-pursuit vehicle control and a RoadRunner→Unity/OpenDRIVE→SUMO geometry workflow. It says setup was tested on Ubuntu20.04 and uses submodules. MIT interface code is useful precedent, but the RoadRunner workflow is an additional dependency and Windows compatibility was not demonstrated here. Do not inherit competing vehicle dynamics merely to animate meshes. [Primary README](https://github.com/TUM-VT/Sumonity).

### 4. BME SUMO–Unity3D connection

[Repository](https://github.com/BMEAutomatedDrive/SUMO-Unity3D-connection), [author-linked demonstration](https://www.youtube.com/watch?v=4RiJyDnm41Q). README describes Python3.7 TCP communication and references a 2018 vehicle-in-the-loop paper; project/executables are split archive packages. Useful historic precedent for pose bridging. Archives and footage were not inspected, so do not assert current Unity/Python/SUMO compatibility or copy its performance claims.

### 5. sumo3Dviz, ETH authors, 2026

[Paper](https://arxiv.org/abs/2604.19194), [repository](https://github.com/DerKevinRiehl/sumo3dviz), inspected commit `10d50036be8fee7ca4b09a5522a7b24e84841924`. Useful author-linked [cinematic](https://www.youtube.com/watch?v=wEUbjlqigyg), [driver-following](https://www.youtube.com/watch?v=DesOELXIWxc), [overview](https://www.youtube.com/watch?v=XvpG5cbv7Ig) and [interactive](https://www.youtube.com/watch?v=dq9pH1Cj7gA) references.

The [CLI source](https://raw.githubusercontent.com/DerKevinRiehl/sumo3dviz/10d50036be8fee7ca4b09a5522a7b24e84841924/src/sumo3Dviz/cli/render_cli.py) loads recorded trajectories/signals for non-interactive modes. Interactive mode skips those loads and vehicle/signal creation, then enables camera exploration. The [simulation manager](https://raw.githubusercontent.com/DerKevinRiehl/sumo3dviz/10d50036be8fee7ca4b09a5522a7b24e84841924/src/sumo3Dviz/tools/simulation_tools.py) advances poses from arrays, not a TraCI session. This distinction prevents mistaking its interactive label for live traffic causality.

The [manifest](https://raw.githubusercontent.com/DerKevinRiehl/sumo3dviz/10d50036be8fee7ca4b09a5522a7b24e84841924/pyproject.toml) pins Panda3D1.10.15, NumPy1.24.4, OpenCV4.9.0.80 and sumolib1.25.0. README says only Python3.9 is supported while the manifest allows `>=3.9`; compatibility with the M1 environment is unresolved. Use a separate environment if an approved trial is later selected.

### 6. CARLA infrastructure cameras → YOLO → SUMO signals

[Traffic Co-Simulation Framework Empowered by Infrastructure Camera Sensing and Reinforcement Learning](https://arxiv.org/abs/2412.03925), revised September2025, reports traffic-pole cameras in CARLA, YOLO vehicle counts and feedback to SUMO multi-junction signal control. It is relevant evidence that rendered observations can drive traffic decisions. It evaluates vehicle counting/control, not the three PFPD hazard models, fallen-tree recognition or local hardware feasibility. Only the abstract/publication record was inspected here; code and footage were not verified.

### 7. OpenCDA and Multi-V2X

[UCLA OpenCDA](https://github.com/ucla-mobility/OpenCDA) describes modular CARLA/SUMO cooperative driving with perception, localization, planning and communication modules; its release examples reference older CARLA versions. [Multi-V2X paper](https://arxiv.org/abs/2409.04980) describes CARLA/SUMO sensor collection from cars and RSUs and links [source](https://github.com/RadetzkyLi/Multi-V2X). These inform source/pose identity and RSU camera placement. They are not validation of these YOLO exports or proof of physical radio behavior; neither was run here.

### 8. Browser appearance references

[Three.js glTF scene tutorial](https://threejs.org/manual/pages/load-gltf.html) supplies an inspectable scene-loading technique; its example city is a small stylized asset and is not the proposed topology. [Babylon Sponza](https://www.babylonjs.com/Demos/Sponza/) and [official feature demos](https://babylonjs.com/featureDemos/) are relevant material/lighting/interactivity references. Their pages were read; full 3D scenes were not loaded or benchmarked. Attractive material rendering in those examples does not prove a complete smart-city simulation.

## Three independent visual concepts

All share about5 km of unique road centerline as a provisional compact-district interpretation, pending the city's final geometry study. The following descriptions define experience and engineering intent; exact junction lengths, lane counts, demand, traffic-light programs and deduplicated RSU positions need the geometry proposal. None reuses the earlier ordinary block grid.

### A. Canal and Campus Quarter — recommended local visual direction

A shallow canal bends through a low-rise research campus and a neighborhood shopping street. A planted boulevard follows the water, two bridges join opposite-bank loops, and a smaller inner campus loop creates alternative routes. Broad daylight, modest rough concrete, textured asphalt, restrained glazing and visible lane wear give depth without large night-light budgets. Trees are grouped into small planted strips so signals and sensing views remain clear.

The overview camera sees the two curved loops and bridge chokepoints; a bus-level chase camera moves under signal arms and past RSU poles. Four-way signal intersections sit where the neighborhood cross-streets meet the boulevard. Potholes appear near a worn service entrance; construction occupies a bridge approach; a stalled car interrupts the opposite loop. RSU state changes appear as a small mast status light and optional event-linked paths, with coverage assumptions shown only when requested.

This is a proposed moderate-complexity flat scene: instanced foliage, simple rooflines, baked ambient shading, one sun shadow region around the selected camera, and distant building silhouettes. Its geometry is compatible with a browser prototype, Unity URP, or Godot. Rich water reflections are optional decoration and must not consume the camera/inference budget.

Concept connection sketch, not a metric map:

```text
Neighborhood loop ── Bridge north ── Campus loop
       │                  canal          │
       └─ planted boulevard ─ Bridge south┘
               service street / workzone
```

Reference techniques: author-linked [sumo3Dviz cinematic views](https://www.youtube.com/watch?v=wEUbjlqigyg) for camera sequencing; [Three.js glTF tutorial](https://threejs.org/manual/pages/load-gltf.html) for reusable asset composition. Footage was not watched.

### B. Foundry and Market District — strongest mixed-traffic narrative

Two irregular loops interlock around a warehouse courtyard and an older market street. The freight loop has wider turning radii, loading setbacks and a service spur; the market loop contains offset streets, awnings, parking recesses and buses. A tree-lined connector gives an alternate path without turning the district into a rectangular grid. Warm late-afternoon light makes rough brick, metal doors, dusty roadworks and vehicles readable.

The opening shot follows a delivery truck through a signal junction, then rises to reveal several independent hazard locations. A work vehicle/cones closes a short service segment, a disabled van reduces available lanes, and a road-surface fixture affects a quieter street. Distinct vehicle silhouettes and signal heads explain circulation. An event ledger selection isolates the originating sensor, relay RSUs and approaching recipients instead of animating continuous decorative radio beams.

This is a proposed medium-to-high asset-composition workload, suitable for browser rendering with atlases/LODs or Unity URP. Occlusion by trucks and awnings becomes an explicit perception test. Lighting should use emissive sign faces with bounded real lights. More object diversity raises CPU/draw-call pressure even if vehicle count stays50.

```text
Freight loop ── warehouse court ── broad connector
     │                                  │
Service spur                     market loop
     └──────── tree-lined alternate ─────┘
```

Reference techniques: [SUMO2Unity's driver/bicycle/overview reference collection](https://github.com/SimuTraffX-Lab/SUMO2Unity) for perspective variety; [Babylon Sponza](https://www.babylonjs.com/Demos/Sponza/) for restrained architectural materials. Neither establishes this proposed city or perception success.

### C. Terraced Green Belt — strongest terrain and driving experience

A road follows a wooded contour above a small valley, while terraced residential streets reconnect to a lower park road. Four-way intersections occupy flat plateaus; climbing connectors link two circulation loops. A civic plaza and maintenance depot provide believable reasons for the roads. Early morning daylight and longer sight lines emphasize elevation, retaining walls, guard rails and vegetation.

A vehicle passes a clear wooded segment; a tree later falls behind it. A downstream camera captures the changed scene and the detector may or may not recognize it. If recognition fails, the physical obstacle still exists but no neural warning is invented. The alternate valley road demonstrates route relevance, while a bridge or plateau workzone supports a simultaneous construction event. Vehicle-level views reveal crest occlusion and sensor visibility limitations.

This is the highest-complexity concept. SUMO Z coordinates, slope-aware meshes, actor origins, sensor poses and shadow/foliage costs must be handled accurately. Prefer a remote Unity/CARLA tier if dense realistic vegetation is essential. A reduced local version is possible only after a bounded trial; no frame-rate promise follows from the concept.

```text
Upper contour loop ── plateau signal junctions
       \                         /
        climbing connectors / terraces
       /                         \
Lower park loop ───── valley alternative
```

Reference techniques: [CARLA RGB camera documentation](https://carla.readthedocs.io/en/0.9.16/ref_sensors/) for pose-tagged sensing and [OpenCDA](https://github.com/ucla-mobility/OpenCDA) for separating sensing and cooperative actions. No downloaded CARLA world or runtime footage was inspected.

## Camera and synchronization implications

Engine capability to render a camera is **not detector validity**. For a browser camera, render a dedicated target and read actual pixels; for Unity use `Camera.targetTexture` plus supported readback; for Godot use a Camera3D viewport texture; for CARLA use `sensor.camera.rgb`. Preserve original bytes, capture frame ID, simulation time, camera pose/intrinsics and asset/render configuration. Readback completion can lag capture; do not replace capture time with result-arrival time. [Three.js renderer methods](https://threejs.org/docs/pages/WebGLRenderer.html), [Unity camera](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/camera/targettexture), [Godot viewport](https://docs.godotengine.org/en/stable/classes/class_viewport.html), [CARLA sensors](https://carla.readthedocs.io/en/0.9.16/ref_sensors/).

Recommended ownership:

1. One coordinator advances the SUMO clock and holds all traffic-control authority. SUMO supplies vehicle poses, speeds, link states and signal states.
2. Renderer applies geometry transforms and timestamped states; display interpolation is allowed between authoritative snapshots. Sensor capture uses an exact acknowledged snapshot, with no future extrapolation used as evidence.
3. Scenario service applies a seeded physical fixture to traffic occupancy and its visual mesh on the same declared simulation tick. A separate ground-truth log records onset/clearance. It does not publish neural predictions.
4. Genuine inference reads actual footage or rendered bytes through the reviewed model registry and preprocesses the static768 input. A model can produce zero detections. A rendered fallen tree is an out-of-domain validation target until evidence establishes recognition.
5. Only accepted prediction evidence produces a warning event; the modeled RSU layer creates delivery receipts; the traffic owner applies eligible advisories and observes resulting motion. Presentation consumes those records.

The real-footage mode should remain available as an explicit manual-fixture association: its inference is real but its video appearance is not a camera recording of the rendered scene. Synthetic mode is a separate experiment requiring frozen positive/negative frames, visibility changes and domain tests. Neither mode silently substitutes the scenario scheduler for perception.

For slow local inference, choose reproducible stepped/slow-time demonstration or asynchronous wall-time execution with bounded queues and explicit measured result age. A renderer can remain responsive while a prediction is pending, but warnings cannot be displayed as received before inference and delivery finish. GPU cloud inference may improve service time; it does not remove capture, network, queue and application delays.

Do not copy Unity vehicle physics or CARLA Traffic Manager into the experiment while SUMO controls those same vehicles. CARLA's bridge documentation explicitly supports one selected traffic-light manager and warns that unsynchronized managers differ. [CARLA–SUMO ownership](https://carla.readthedocs.io/en/0.9.15/adv_sumo/).

## Proposed feasibility gates, requiring implementation approval

These are proposed acceptance targets, not current measurements or authorization to execute them.

| Approved experiment | Required observation | Proposed decision gate |
|---|---|---|
| Small representative browser scene: one four-way junction, asphalt/curbs, two building groups, signal meshes, RSUs and 50 vehicle instances | Record browser/GPU/API/version, draw calls, memory, frame-time distribution, signal correctness and camera view | Aim for30 visual FPS at1280×720; require at least20 sustained FPS and p95 frame interval≤50 ms after warmup. Revise target with user if visual quality/performance conflicts. |
| Same scene plus one actual sensor target and one M1 CPU inference worker | Save capture pixels/hash, pose/tick, measured readback/inference age and CPU/RAM competition | No renderer freeze from inference, no warnings before results, bounded queue and no silent dropped evidence. FPS and inference speed are separate metrics. |
| Approved district scene and continuous traffic | 50+ active vehicles maintained during scored interval; run15 minutes then a60-minute soak; memory/frame/pose/signal logs | No steadily growing memory, IDs or queues; timestamp/coordinate mismatch cases pass. Long-session traffic capacity needs the traffic agent's gate. |
| Frozen synthetic positive/negative frames per model | Annotation/provenance, fixed thresholds, genuine outputs, misses and likely false positives | Treat rendered perception as experimental until scene recognition is established. Do not lower thresholds to obtain a desired narrative. |
| Alternative Unity player trial only if browser target is insufficient | Same camera/asset/vehicle quality target and measured resource footprint | Select based on evidence and authoring needs, not engine reputation or a beautiful unrelated demo. |

The initial study should use small locally authored assets. It must not require a major engine install, large asset download or paid GPU session until separately approved. A remote design should keep rendering, camera capture and inference together where practical and stream compressed presentation/state results; a split local traffic/remote renderer adds round-trip synchronization that must be measured. Operational costs and actual remote capacity remain deployment decisions.

## Open uncertainties

- No renderer's local performance, Windows graphics API support, install footprint or full5 km asset budget was measured.
- 50-agent interactive scenes and multiple sensor views were not exercised.
- Existing M1 smoke fixtures do not establish synthetic-frame recognition, calibrated confidence or fallen-tree/debris recognition.
- The Unity bridge references are not drop-in compatible with the current SUMO/Python baseline; SUMO2Unity's Python counterpart is not inspectable in the reviewed file.
- The modern CARLA UE5 bridge and old UE4 bridge must not be assumed interchangeable.
- Geometry conversion, elevation, vehicle-origin/yaw transformations, signal link mapping and exact RSU counts remain approved implementation work.
- Visual references and papers support techniques; they do not establish PFPD safety gains or radio-network fidelity.

