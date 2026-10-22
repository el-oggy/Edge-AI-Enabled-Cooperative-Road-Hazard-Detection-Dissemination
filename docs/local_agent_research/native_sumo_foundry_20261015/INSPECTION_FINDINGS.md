# Current workspace inspection

15 October 2026. Read-only inspection; no runtime acceptance is implied.

## Preservation and recovery

The starting checkout is branch `main`, HEAD `47b0eddf1057c4b3f0ccd4dea1faed941c25a716`. The only tracked working-tree modification is `.gitignore` with two inserted lines allowing skill Markdown. Untracked work includes `.agents/`, `simulation/smart_city/`, `src/pfpd_city_perception/`, `src/pfpd_city_protocol/`, `src/pfpd_city_runtime/`, `tests/smart_city/` and `visualization/`. The earlier research and other ignored local files are present too; Git's short status does not enumerate ignored files.

Uncommitted files already on disk can be inspected directly. No `git restore`, `reset`, `clean`, checkout, stash or regeneration is needed. Git cannot recover arbitrary untracked changes that were never saved in a commit or other snapshot. This review does not claim to have recovered missing unsaved content. The [baseline](PRESERVATION_BASELINE.json) records 170 existing source, simulation, test, model, memory and earlier research files. The in-progress geometry specialist's one planning document was explicitly excluded from that comparison.

## Existing native Foundry implementation

| Area | Inspected evidence | Meaning and remaining check |
|---|---|---|
| Native application | Installed `sumo-gui.exe --version`: 1.27.1; GUI feature present, OSG absent | Standard native overhead view is supported by the installation; native 3D and actual appearance were not tested |
| Network | `simulation/smart_city/foundry/foundry.nod.xml`, `.edg.xml`, `.net.xml` | Seven input nodes, 24 noninternal directed edges, three TLS controllers: TL_D1, TL_D2, TL_D3 |
| Road extent | Sum of reverse-paired input street polylines | **5000.006476929 m unique input centerline**; excludes lane multiplicity and internal junction links. This does not assert the compiled lane total is 5 km |
| Route connectivity | All consecutive edge pairs and last-to-first pairs for seven route definitions appear in compiled connections | Logical connectivity passes the static check; vehicle-class permissions, swept turns, demand and long-run circulation are untested |
| Traffic | Four vTypes; 12 flows; finite repeated route lists | Cars, buses, trucks and motorcycles exist in definitions. Replenished flows do not establish persistent identity or a sustained active count |
| Loaded scene | Configuration loads city and hazard additional files | City file has 36 polygons and 27 POIs. `foundry_buildings.add.xml` is present but not referenced in the active configuration |
| Current styling | `real world` scheme; zoom 260; delay 40 ms | No tailored project scheme or decals in that view file. A delay setting alone is not an FPS or real-time measurement |
| Hazard symbols | Pothole/crater polygons, barrier polygons and named POIs | POIs without `imgFile` are circles; names such as CONSTRUCTION_CONE do not create that silhouette |
| RSUs | 27 city POIs, including perimeter-gate sites | A count of 27 does not prove the proposed 200 m chainage placement, actual sensing, networking or delivery. Placement needs reconciliation |

The unique-road calculation reads each input shape, sums Euclidean leg lengths, verifies reverse pairing, and counts each physical street once. Small coordinate rounding explains the difference from exactly 5000 m. The original proposal's maximum-spacing calculation is 27/11 units at 200/500 m limits, but the current hand-placed POIs must be audited before either deployment is claimed operational.

## Launcher and inference gaps requiring later work

The existing entry point is `simulation/smart_city/foundry/run_foundry_simulation.py`. Its default mode is headless; `--gui` selects SUMO-GUI. Default 300 steps at 0.1 s is only 30 simulated seconds, even though the configuration has an end time of 3600 s.

**Do not run it as a harmless preview:** its constructor calls `FoundryHazardGenerator.generate_hazards`, which writes `foundry_hazards.add.xml`. Neither this launcher nor either generator was run in this review. Opening the `.sumocfg` directly would load static scene/traffic settings but would not execute its Python controller.

The runner loads the three ONNX models through `CityCameraPerception` and selects an archived photo according to the scheduled hazard type. The three referenced image files currently exist. If an image is missing/unreadable, the code substitutes a black frame; that is a source-health defect to remove in a later approved change. Repeated use of a still is not independent temporal evidence.

The ONNX results are printed as telemetry. Vehicle slowdown/merge behavior in `run_step` instead reads `active_hazards` and same-edge distance directly. The positive prediction is therefore **not the cause of that reaction**. The loaded hazard XML also applies closures or speed limits independently of neural predictions. Those are legitimate declared physical/traffic rules, but they cannot be counted as proof that AI warnings worked.

The adapter duplicates preprocessing/decoding, uses provisional confidence 0.25, emits a new random event ID for each box, and does not use the M1 temporal evidence path. The current M1 runner's default 0.35 is itself explicitly uncalibrated. Neither number is an accuracy guarantee; preserve declared operational settings and validate them rather than reducing thresholds to obtain detections. The generic junction runtime has direct hazard injection and radial receiver selection; those mechanisms need separation from any claim of genuine cooperative detection.

The Foundry script suppresses SUMO warnings and catches some TraCI errors silently. `vehicles_reacted` is keyed only by vehicle ID, so it cannot represent separate later hazards/revisions on recurring trips. Repair planning must include relevant-event lifecycle, restore observable errors, legal actions and observed outcomes. No fix was made during this review.

## Checks performed and limits

Successful read-only checks: Git status/diff statistics; XML parsing; input shape arithmetic; counts of nodes/edges/TLS/scene objects; membership of route edge pairs in compiled connections; launcher/perception/test source inspection; fixture existence checks; installed SUMO-GUI version/features; installed schemas/client APIs and official source/documentation research.

An early attempt to read root `AGENTS.md` found no such file. An early attempt to read the geometry specialist's document occurred before it existed; it was read successfully after delivery. An exploratory request for an official `Simulation/Obstacles.html` page did not resolve; the plan relies instead on official lane, vehicle, rerouter and speed-sign documentation. None of these failed lookups was treated as runtime evidence.

The first planning-package consistency pass found one missing local link: README referenced `VALIDATION.json` before that generated report existed. That pass created the report while recording the failure. The final pass checks the now-existing report and current document hashes. This was a document-generation ordering issue; all 170 protected existing files and HEAD matched in the initial pass.

No existing tests, SUMO simulation, ONNX forward, GUI appearance test, screenshot or performance benchmark was executed. Existing test files were inspected; a black-frame inference test is not a representative clean-road negative suite. No precision, false-positive rate, latency improvement, collision avoidance, radio performance or final visual-quality claim is established here.
