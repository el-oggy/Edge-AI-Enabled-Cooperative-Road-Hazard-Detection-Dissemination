# Native SUMO Foundry-and-Market plan

15 October 2026, Asia/Calcutta. The user selected Foundry-and-Market, clarified the dry/desert-industrial theme, and explicitly rejected a webpage as the simulation product. This revision specifies a native SUMO-GUI launch and preserves the other agent's implementation. It is a planning deliverable; source/XML changes wait for the next implementation instruction.

## The intended experience

Click **Run SUMO Simulation** in a local launcher or shortcut. It opens the installed SUMO-GUI directly, loads the Foundry map, applies the supplied visual settings, starts visible mixed traffic and retains SUMO's play/pause, pan, zoom, object inspection and signal controls. No web server is part of the product. A close native tracking view follows a selected vehicle from above; the overview remains available.

The map should feel like a dry working district: sandy lots, charcoal asphalt, worn shoulders, rust and brick foundry roofs, warehouse aprons, small market awnings, bus shelters and sparse olive trees. Preserve the existing diagonal boulevard and irregular freight/market loops. Use the approximately 5 km of total unique streets already present; do not enlarge the world to a 5 km radius or replace the current network merely to match a new drawing. Traffic density should suit the scene. Retain the earlier 50+ active-vehicle demonstration target unless the user changes it, but tune it through measured flow and queue checks rather than forcing 72 actors into an unsuitable layout.

The installed 1.27.1 binary has standard GUI support and no OSG feature. The achievable first visual target is a carefully composed, textured aerial simulation, with convincing scale and motion. A close overhead track does not provide a driver's perspective. SUMO's optional native 3D is experimental and requires a different capable build; it is excluded from this baseline. [Native SUMO-GUI capabilities](https://eclipse.dev/sumo/docs/sumo-gui.html#3d-visualization).

## Reuse first

The current project already has a native network, routes, scenery, hazard generator, controller, perception adapter and tests. The [inspection findings](INSPECTION_FINDINGS.md) identify exactly what exists. Keep those files and reconcile their behavior before adding replacements. The current input centerlines measure 5000.006476929 m, with seven nodes, twelve two-way streets and three signal controllers. Keep network changes out of a visual polish task unless a specific traffic defect requires one.

The prior conceptual graph calculates 27 dense and 11 sparse RSUs from seven mandatory junction anchors and equal chainage subdivision. That is a useful design reference, not proof about the current 27 hand-placed POIs. Audit the actual network offset, roadside setback, chainages and longest gaps. Spacing is distinct from camera coverage or radio reach.

## How native SUMO can look better

| Element | Concrete design | Native mechanism and check |
|---|---|---|
| Terrain | Sand `#C5B391`, dusty yards `#A99A80`, sparse olive `#72785A` | Background/negative-layer polygons; keep road markings visible |
| Roads | Charcoal `#464747`, pale curbs `#D8CDB7`, restrained wear | Export a named scheme from installed SUMO-GUI; retain actual lane/junction geometry |
| Foundry | Rust `#875843`, steel `#61605A`, roof seams, aprons and tanks | Small local roof textures and layered footprint polygons; inspect road clearances |
| Market | Ochre `#C18D40` awnings, smaller buildings, loading courts | Repeated native polygon/sprite assets outside traffic lanes |
| Vehicles | Recognizable car/bus/truck/motorcycle silhouettes, muted body colors | Explicit supported `guiShape` or raster images with the correct drawing-quality setting |
| Hazards | Recognizable physical artwork plus a small nearby category badge | Transparent local PNG POIs or simple polygons; verify size, angle, tint and layer |
| Traffic signals | Visible stop-line movement states at all three intersections | Native TLS state bars; any extra signal-head icons must follow actual controlled links |
| RSUs | Small consistent mast/cabinet symbol with ID on selection | Native POIs; highlight only real application receipts when that service exists |

Native shapes support fill, layers, images and dimensions; naming a POI does not create a cone or tree. Shapes are visual objects without automatic traffic interaction. [Shape declarations](https://eclipse.dev/sumo/docs/Simulation/Shapes.html). Vehicle visual shape and road-user class are separate. [Vehicle visualization](https://eclipse.dev/sumo/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#visualization).

Use a small portable asset set and the existing Python environment; no new rendering engine is needed. Baked illustrative shadows may add depth to roofs, but should not cover lanes. Avoid full-map labels, fluorescent ordinary vehicles, perpetual flashing and oversized hazard craters. A one-meter hole is barely a pixel at district scale: use a nearby overview badge and a close view, while keeping the physical footprint accurate.

The native GUI supports reusable view schemes, background decals and multiple map views. Save a whole-district overview and an approximately 80–150 m close tracking view. Portable view files must not depend on a developer's registry preferences. [GUI configuration](https://eclipse.dev/sumo/docs/sumo-gui.html#configuration-files), [GUI control API](https://eclipse.dev/sumo/docs/TraCI/Change_GUI_State.html).

## Category distinctions the demonstration must show

The user confirmed that ordinary comparisons include puddles, manhole covers and repaired patches. These are required visual and inference-test cases. Map category labels come from scenario metadata; detector results are separately identified.

| Category | Visual identity in SUMO | Traffic meaning | Neural evidence required |
|---|---|---|---|
| Pothole | Jagged asphalt cavity, rough rim and cracks | Passable slowdown or reviewed avoidance; no suspension-damage physics claim | Actual pothole-model prediction and temporal evidence |
| Puddle | Smooth blue-gray patch and soft reflection | Ordinary shallow comparison initially; no hazard action just because it resembles a hole | Genuine negative testing; no claim the detector has a puddle class |
| Manhole / repaired patch | Regular metal cover or flush asphalt repair | Normal road detail unless a separate physical defect is declared | Genuine hard-negative testing |
| Fallen tree / debris | Branching trunk or disconnected angular pieces crossing a lane | Defined blocked interval, safe stop/diversion as appropriate | Accident/obstacle recognition is unproven for these subtypes; misses remain misses |
| Accident / disabled vehicle | Vehicle body/glazing in an incident pose, triangle and optional debris | Actual occupancy/closure or stopped vehicle behavior | Accident/obstacle-model evidence; ordinary parked/stopped vehicles are negatives |
| Construction | Cone taper, striped barriers, work patch, maintenance vehicle | Reviewed work-zone restrictions, merging or diversion | Construction-model evidence |
| Ordinary traffic | Correct proportions and heading, distinct buses/trucks/bikes | Native traffic behavior and signals | No automatic hazard tag for stopping at lights, parking or congestion |

All three existing ONNX heads are independent single-class detectors. The accident head is not a verified tree-versus-debris-versus-crash classifier. Neither puddle/manhole classification nor every obstruction subtype can be promised without new capability evidence. Preserve the existing weights; testing establishes the actual limits.

## Genuine inference in a native SUMO presentation

Use actual real images or timestamped footage with explicit clip-to-lane/location mapping as the first defensible perception input. The source should be visible through clear mode text, receipts and optional saved annotated images; a new dashboard is unnecessary. Native map drawings illustrate that mapped scenario. Do not feed colored map symbols to real-road detectors and assume the result validates road perception. A future screenshot-input experiment must be labeled separately and allowed to fail.

```text
independent physical scenario → native scenery + actual traffic occupancy
independent sensor schedule → real footage/image → actual ONNX execution
  → per-model evidence → valid temporal confirmation → explicit lane mapping
  → event eligible after actual service delay → modeled RSU delivery
  → relevant approaching vehicle → safe TraCI request → observed outcome
```

The first line cannot inject a positive detection into the second. The current runner selects photos by scheduled hazard and controls vehicles directly from that hazard list; those are inspected gaps, not completed AI causation. Reuse the established M1 input/model/provenance and tracking logic where practical instead of maintaining a second weaker detector implementation. Keep source-model identity, actual thresholds, frame IDs, capture times, raw outputs and timings.

Single still images establish inference execution only. Repeated copies cannot establish temporal persistence. For real clips, retain capture/PTS order and gaps even when processing is slow. Bound queued work and log dropped/stale predictions. Traffic must not receive an event before actual inference/confirmation completes; a paused or time-dilated demonstration must declare its time mapping. No fabricated latency or real-time claim follows from a GUI delay value.

Keep physical blockage the same in warnings-on/off comparisons. A hard lane closure may independently inform routing; record that source and separate it from advance cooperative warnings. A speed limit alone does not model a solid tree or wreck. Lane permissions close a whole lane, not an arbitrary short segment. Choose a supported obstruction representation and document that approximation. [Rerouter closure behavior](https://eclipse.dev/sumo/docs/Simulation/Rerouter.html), [speed-sign behavior](https://eclipse.dev/sumo/docs/Simulation/Variable_Speed_Signs.html), [TraCI vehicle actions](https://eclipse.dev/sumo/docs/TraCI/Change_Vehicle_State.html).

## Implementation sequence after approval

| Package | Scope | Completion evidence |
|---|---|---|
| N0 — preservation and audit | Completed planning baseline; read current uncommitted work | Starting Git state, file hashes, static geometry/route checks and disclosed gaps |
| N1 — native appearance and launch | Reuse Foundry; tailored scheme, necessary local artwork, distinct benign/hazard symbols, two native views; convenient local launcher | Actual SUMO-GUI overview and close screenshots; correct signals; portable launch from another working directory; no authored asset overwrite |
| N2 — scenarios and sustained traffic | Seeded activation/clearance, after-A-passes case, passable/blocked policies, route lifecycle and traffic load | Measured active-count timeline and queues; class-valid routes; no hidden errors/teleports; repeatable scenario logs |
| N3 — genuine causal warnings | Reuse immutable models/M1 evidence, real footage, temporal validity, RSU event lifecycle, relevant safe response | Frame→prediction→confirmation→delivery→command→observation IDs; failure/negative/stale controls; measured service delay |
| N4 — comparison and presentation | 200/500 m placement audit, fixed sensing inventory, paired warning experiments, long-run stability | Placement/gap tables, seeded delivery logs and traffic/safety outcomes; honest measured limits |

Do N1 before broad art work. Start at one representative junction and extend the same scheme across the existing map after review. Once N1 is approved, its scoped launch adjustment belongs to the coordinator; it does not authorize silently replacing perception or adding new simulation engines. N2 and N3 may overlap only under clear ownership after their scope is approved.

The existing runner rewrites hazard XML on launch. The launch repair should generate any run-specific hazards/config/output under a new results run directory and leave authored files intact. Preserve relative asset resolution when relocating configuration; copying a `.sumocfg` alone is insufficient. GUI mode should be the normal user entry, errors should remain visible, and no silent headless or mock fallback should hide failure.

Acceptance includes directly launched native SUMO-GUI, visually distinguishable required categories at suitable zoom, functional traffic signals, maintained traffic movement, source preservation, and honest source mode. The first extended traffic check should log a declared 20-minute simulated window and wall-clock duration. The earlier 50+ target remains an acceptance target, not a measured result. A 30-simulated-second default launch cannot establish extended circulation.

## Cost-aware assignments

Completed research assignments: **Luna / medium** inspected current files and located Ponytail; **Sol / high** researched native visuals and installed build/API limits. The principal agent handles architecture and reconciliation. These actual assignments satisfy the current research delegation; implementation prompts remain gated.

For approved implementation, use Luna for bounded scenery/scheme edits and later screenshot/artifact checking; use a capable Sol specialist for network/TLS correctness and another for perception/integration risks. Assign one owner to the runtime and shared contracts. With four concurrency slots, run two or three independent contributors, then a separate verification pass. Do not keep agents polling while idle or split a one-file task into a team. Model routing is a practical allocation, not a measured price/performance benchmark.

## Sources and reusable examples

The [native visual specialist report](SPECIALIST_NATIVE_VISUALS.md) links official version-pinned 1.27.1 schema, loader, drawing and client/API sources. This is important because the public documentation describes development versions and some older drawing-layer notes are stale. Verify accepted settings against the installed build.

[MoST](https://github.com/lcodeca/MoSTScenario) and [LuST](https://github.com/lcodeca/LuSTScenario) demonstrate packaged native SUMO scenarios with reusable network/demand/config organization. They are references for structure and urban detail, not a reason to download their city datasets or reuse their performance figures. MoST's documented 3D build differs from this installation. The [official circulation tutorial](https://eclipse.dev/sumo/docs/Tutorials/Driving_in_Circles.html) is a useful route-renewal reference, while [signal modeling](https://eclipse.dev/sumo/docs/Simulation/Traffic_Lights.html) supplies functional control semantics. No linked demonstration video was watched or credited as runtime verification.

Implementation scope still awaiting approval: **N1 native visual polish and safe launcher**, followed by separately scoped N2/N3 repairs. The original request to plan first and preserve the other agent's work is why this turn ends with a reviewable proposal instead of changing those files.
