# 1 — Technology research and comparison

Date: 15 October 2026. All engine choices below are proposals. Detailed repository inspections, pinned commit references and additional visual sources are in [the rendering specialist note](SPECIALIST_RENDERING.md).

## What the existing project supplies

The preserved 1 km corridor supplies traffic, signals and a scripted cooperative-warning demonstration. Its current controller does not execute ONNX, and its hazard drawings are cosmetic. The M1 runner now executes genuine image inference with distinct specialist identities. The current read-only model audit rechecked the three hashes, single-class names and static input/output contracts without executing new forwards. The recorded M1 observations show successful CPU forwards and questionable detection regions; they do not establish synthetic-camera recognition. [Local M1 report](../M1_IMPLEMENTATION_REPORT.md), [current audit](HOST_PROJECT_AUDIT.json).

Existing research remains useful for causality, safe TraCI ownership and evidence boundaries. Its simple SUMO-GUI presentation is not the proposed final experience. The new requirement calls for proper 3D road surfaces, curbs, buildings, vegetation, functional signal heads, vehicle meshes, hazards, RSUs and close cameras.

## Fair comparison

| Approach | Visual world and sensor frames | Traffic integration and burden | Fit and decisive limitation |
|---|---|---|---|
| SUMO-GUI 2D | Strong traffic inspection; schematic presentation | Native traffic and TLS authority | Debug/reference view; insufficient final visual target |
| SUMO OpenSceneGraph | Built-in 3D roads, junctions, signals and models | Optional build support; documented experimental limitations | Bounded comparison only; no proven 50-agent performance or suitable perception camera bridge here |
| SUMO + Three.js/WebGL2 | PBR/glTF assets, custom camera render targets, instancing | New small Python state/capture adapter; inspectable older browser precedent | **Recommended first local pilot**; art quality and FPS must be demonstrated |
| SUMO + Babylon.js | Browser engine with glTF, camera/render-target and optimization facilities | Same custom SUMO interface requirement | Credible alternative; developing both browser engines adds little value |
| SUMO + Unity URP | Mature scene editor, lighting, cameras and asset workflows | Unity/SUMO bridges exist; authority, versions and closed components need review | Best alternate for higher visual production; editor RAM competes with current host |
| SUMO + Godot Compatibility | Editable native scenes, cameras and relatively modest baseline | Custom network/pose/TLS adapter and asset pipeline | Viable lightweight native alternate; no inspected turnkey bridge here |
| SUMO + Unreal | Strong cinematic scene/camera tools | General-engine bridge and map pipeline are substantial work | Workstation/remote tier, not the current laptop's first build target |
| SUMO + CARLA UE4 | RGB sensors and driving-world co-simulation tools | Documented matched-map SUMO bridge; pin server/API/TLS owner | Strong future sensor tier; dedicated GPU requirement is unmet locally |
| SUMO + CARLA UE5 | Higher visual ceiling and sensors | Separate version track; custom-map/import limitations vary by release | Remote research option after explicit package compatibility check |
| sumo3Dviz/Panda3D | Recorded trajectory cinematic and driver views | Inspected path replays recorded arrays; interactive path explores a static world | Useful offline export reference; not an inspected live neural-feedback bridge |
| Blender | Asset authoring, baked appearance, offline visual production | Export small licensed glTF assets to selected renderer | Supporting tool, not sole live traffic/sensor authority |

Published capabilities: [SUMO-GUI and experimental OSG](https://eclipse.dev/sumo/docs/sumo-gui.html), [Three.js renderer/readback](https://threejs.org/docs/pages/WebGLRenderer.html), [instancing](https://threejs.org/docs/pages/InstancedMesh.html), [glTF loader](https://threejs.org/docs/pages/GLTFLoader.html), [Babylon specifications](https://www.babylonjs.com/specifications/), [Godot viewports](https://docs.godotengine.org/en/stable/tutorials/rendering/viewports.html), [Unity camera targets](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/camera/targettexture), [Unreal rendering components](https://dev.epicgames.com/documentation/unreal-engine/rendering-components-in-unreal-engine), [CARLA RGB sensors](https://carla.readthedocs.io/en/0.9.16/ref_sensors/), [CARLA SUMO bridge](https://carla.readthedocs.io/en/0.9.15/adv_sumo/), [sumo3Dviz source](https://github.com/DerKevinRiehl/sumo3dviz).

## Relevant projects and demonstrations

| Project/reference | What inspected evidence establishes | Reusable technique | Limits and compatibility |
|---|---|---|---|
| [SUMO-Web3D](https://github.com/sidewalklabs/sumo-web3d) | Python TraCI stepping/subscriptions, WebSocket differences, Three.js signal-link rendering | Single traffic authority, state stream, geometry and light mapping | Archived April 2023; old frontend stack. Reimplement a pinned adapter rather than install unchanged. No YOLO hazard evidence |
| [SUMO2Unity](https://github.com/SimuTraffX-Lab/SUMO2Unity) | Unity importer and JSON/NetMQ pose/TLS handling; README advertises 0.10 s exchange | Lane/junction mesh generation and main-thread state queue | Targets SUMO 1.22/Unity 6000.0.53f1. Inspected Python file is a stub and companion executable is opaque. No drop-in reproducible bridge assurance; pedestrians absent in README |
| [Sumonity](https://github.com/TUM-VT/Sumonity) and [paper](https://www.tib-op.org/ojs/index.php/scp/article/download/1115/1242/28087) | Published Unity/SUMO coupling approach | Unity integration/coordinate architecture to compare | Not executed here; current-version and complete camera pipeline validation needed |
| [BME SUMO–Unity connection](https://github.com/BMEAutomatedDrive/SUMO-Unity3D-connection) | Published older Python TCP/Unity example | Simple process separation and pose exchange | Older Python 3.7 packaging; large split archives not downloaded |
| [sumo3Dviz paper](https://arxiv.org/abs/2604.19194) / [code](https://github.com/DerKevinRiehl/sumo3dviz) | Recorded traffic and signal replay; source inspected by specialist | Driver/overview/cinematic camera presentation | Python/NumPy/sumolib pins conflict with M1 environment; isolate if later selected |
| [Infrastructure cameras + YOLO + SUMO control](https://arxiv.org/abs/2412.03925) | Publication reports rendered infrastructure camera vehicle counts feeding signal control | Demonstrates observation-to-traffic-control research pattern | Abstract inspected; source/footage not verified. Vehicle counting differs from PFPD hazard categories |
| [OpenCAMS](https://arxiv.org/abs/2507.09186) / [repository](https://github.com/minhaj6/carla-sumo-omnetpp-cosim) | Published CARLA/SUMO/OMNeT++ division of responsibilities | Future detailed communications co-simulation | Publication/repository record, not a local execution or current compatibility proof |
| [OpenCDA](https://github.com/ucla-mobility/OpenCDA) | Modular cooperative driving architecture with perception and planning | Sensor/source identity and module separation | Version-sensitive CARLA stack; not validation of existing PFPD weights |
| [Multi-V2X](https://arxiv.org/abs/2409.04980) / [code](https://github.com/RadetzkyLi/Multi-V2X) | Published vehicle and RSU sensor-data approach | Pose-aligned multi-source recording | Sensor dataset/cooperation evidence, not physical radio or PFPD detector accuracy |

Visual references to review: [RoadwayVR SUMO-to-Unity driving tour](https://www.youtube.com/watch?v=im94-gCvA1E), [truck/traffic demonstration](https://www.youtube.com/watch?v=lmsBN3Rdmdw), [Babylon Sponza materials](https://www.babylonjs.com/Demos/Sponza/), [Three.js glTF scene example](https://threejs.org/manual/pages/load-gltf.html). Video descriptions and related code were inspected; footage was not watched. The graphics examples were not loaded or benchmarked. They inform scene composition and camera experience, not expected performance.

## Choice and revisit rules

**Proposed:** start with Three.js + WebGL2, glTF/PBR assets, instanced vegetation/traffic, limited dynamic shadows, and one selected camera capture target. This is a full 3D scene proposal, not a colored map. It avoids a major editor installation and keeps a readable adapter around existing Python inference. The visual target is daylight realism: believable scale, road wear, curbs, occlusion, signal furniture and controlled camera composition.

If the approved first slice fails the visual review or resource gate, compare a small Unity URP exported-player slice on a suitable machine. Godot Compatibility remains the native alternative if browser readback or driver behavior is the decisive blocker. CARLA becomes appropriate when realistic sensor/environment research and remote GPU resources justify its larger stack. Do not quietly substitute a flat GUI if the scene misses its target.

No renderer FPS, synthetic hazard accuracy or engine compatibility was measured in this assignment. Public requirements are simple-project/editor baselines, not proof that this scene works. Version-sensitive and cloud details are in [hardware feasibility](04_HARDWARE_DEPLOYMENT.md).
