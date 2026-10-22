# Native SUMO-GUI Foundry and Market visual research

Research date: 15 October 2026. Research only. This report is the only file created by this specialist. Existing simulation, XML, models and launchers were inspected without modification. No simulation, GUI visual inspection, screenshot capture, benchmark, engine installation or Git mutation was performed. Appearance proposals below are design judgments, not demonstrated output.

## Recommendation and verified local boundary

Use the installed **standard SUMO-GUI in 2D**, opening directly from the project launcher. Aim for a richly illustrated aerial traffic simulation: dusty ground, charcoal roads, warm industrial roofs, market canopies, detailed vehicle silhouettes, and distinct hazard artwork. Preserve native pan, zoom, inspection, signals and vehicle tracking. A close tracking view can be attractive and informative, but it remains an overhead view; it is not a windshield camera or photorealistic street scene.

The executable inspected was `C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe`. Running only `--version` returned:

```text
Eclipse SUMO GUI 1.27.1
Build features: Windows-10.0.17763 AMD64 MSVC 19.29.30133.0 Release FMI Proj GUI FMT SWIG Parquet Eigen
```

**OSG is absent from that feature list**, and the inspected `bin` directory returned no `*osg*` files. The distribution contains `data/3D/*.obj` and `.mtl`, but those resources do not establish executable support. Treat native 3D as unavailable for this installed baseline. The current official downloads page separately offers Windows standard and extra distributions and identifies the extra distribution with 3D support; no replacement was downloaded. [Official distribution information](https://sumo.dlr.de/docs/Downloads.php).

The earlier `smart_city_20261014/SPECIALIST_RENDERING.md` proposed browser rendering. That recommendation does not apply to this native-only brief. No webpage, external renderer or new engine is required by this proposal.

## Current scene: inspected facts

- `simulation/smart_city/foundry/foundry.sumocfg` loads `foundry.net.xml`, `foundry.rou.xml`, `foundry_city.add.xml`, `foundry_hazards.add.xml` and `foundry_view.xml`. The separate `foundry_buildings.add.xml` is not listed in this configuration; its existence does not mean it is displayed.
- `foundry_view.xml` contains only the `real world` scheme, a viewport at approximately `(458.46,383.96)` with zoom `260`, and delay `40`. It does not define a project-specific visual scheme or decals. Its comment calling this delay realistic is not a timing measurement. With a configured 0.1 s simulation step, a 40 ms delay alone does not enforce wall-clock real time.
- `foundry_city.add.xml` uses a dark terrain polygon, flat district/building polygons and POIs for RSUs. Its irregular zone outlines warrant geometric review before being reused as final artwork; some listed points repeat.
- Hazard POIs in `foundry_hazards.add.xml` have no `imgFile`. A type such as `CONSTRUCTION_CONE` or `HAZARD_ACCIDENT_TRIANGLE` is semantic metadata, not an instruction to draw that silhouette. The loaded shapes therefore do not establish visible cones or triangles.
- The inspected routes define car, bus, truck and motorcycle classes/colors but no explicit `guiShape` or `imgFile`. Comments asserting 50+ active actors are not measured actor counts.
- `run_foundry_simulation.py` invokes hazard generation in its constructor. Do not use that script as a harmless visual inspection command: it can regenerate existing additional files. It was not executed.

## Native capabilities and precise controls

The installed 1.27.1 files are useful version-specific evidence: `data/xsd/additional_file.xsd`, `data/xsd/types/route.xsd`, `data/xsd/viewsettings_file.xsd`, and `tools/traci/_gui.py`, `_poi.py`, `_polygon.py`. They establish accepted declarations or client API signatures, not successful rendering on this GPU.

| Need | Native mechanism | Practical constraint |
|---|---|---|
| Terrain, lots, roofs, cracks | `<poly id type color fill layer shape lineWidth imgFile angle name>` | Polygon fill and outline are separate choices; use multiple polygons for layered artwork. No automatic building extrusion follows from a building-like ID. |
| Recognizable hazards and RSUs | `<poi ... imgFile width height angle>`; location by `x/y`, `lane/pos/posLat`, or geographic coordinates | Dimensions are world meters. Without an image the normal POI is circular. Set image color to white to avoid unwanted tint. |
| Textured lots/roof footprints | Polygon `imgFile` | A bitmap fill is not a material system with normals, displacement, wet reflections or dynamic lighting. Test UV appearance on irregular geometry. |
| Decorative backdrop and fixed legend | `<decal file centerX centerY width height rotation layer screenRelative>` in view settings | World coordinates position the art on the map; `screenRelative` gives screen-based placement. Prefer modest local PNGs. |
| Traffic diversity | `vType guiShape`, `imgFile`, physical dimensions and color | `imgFile`, not `imageFile`, is the XML spelling. Traffic class and visual shape are separate attributes. |

Polygon/POI declarations and units are documented in [SUMO Shapes](https://sumo.dlr.de/docs/Simulation/Shapes.html). The inspected [1.27.1 shape loader](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/src/utils/shapes/ShapeHandler.cpp) resolves non-absolute polygon and POI `imgFile` paths relative to the containing configuration file; there is no need to invent a `relativePath` XML attribute. The local schema does not list one. Use portable asset paths and validate them through the actual launcher working directory.

For vehicles, native `guiShape` options include passenger/sedan, passenger/hatchback, delivery, truck/semitrailer, bus and motorcycle. The chosen GUI drawing quality must be **simple shapes** to use these shapes, or **raster images** to use `vType imgFile`; grayscale vehicle artwork allows meaningful recoloring. `osgFile` refers to the separate OSG path and does not make the standard 2D view three-dimensional. [Vehicle type definitions and visualization](https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#visualization).

The native GUI supports saved view schemes, initial viewports, decals, snapshots, delay and automatic start. A proposed launcher would resolve the existing binary and configuration by absolute paths and invoke `sumo-gui -c <approved-config> --start`; it should expose load errors. Ship the scene's scheme with its files instead of depending on a developer's GUI registry. [SUMO-GUI configuration and decals](https://sumo.dlr.de/docs/sumo-gui.html#configuration-files).

### Settings to author through the installed GUI and export

The 1.27.1 serializer exposes `backgroundColor`, `laneShowBorders`, `showLinkDecals`, `showLinkRules`, vehicle `vehicleQuality` and `showBlinker`, junction `drawShape` and `showLane2Lane`, plus object size and text settings. Use muted road and junction color schemes, visible lane borders/arrows, real junction fill, and live signal bars. Disable diagnostic lane-to-lane lines and mass object labels for the presentation preset. Export numeric quality/mode values from the installed GUI rather than guessing enum values. [Version-pinned settings serializer](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/src/utils/gui/settings/GUIVisualizationSettings.cpp).

The corresponding [1.27.1 settings parser](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/src/utils/gui/settings/GUISettingsHandler.cpp) reads these controls. The bundled `viewsettings_file.xsd` is much less complete than the serializer: it omits many exported nested scheme settings and even the current scene's delay element. Schema-only validation is therefore not proof that a saved GUI scheme is unsupported; inspect the generated file and verify loading in the application during a later approved trial.

Traffic-signal bars carry live movement state; a decorative traffic-light image does not. Keep native bars visible and uncluttered. If realistic-looking roadside heads are later added as POIs, their colors must be synchronized to the actual TLS ID and link index by the single controller. Arbitrary red/green decoration would misrepresent control. [1.27.1 lane drawing and link-rule implementation](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/src/guisim/GUILane.cpp).

### Layer design

Use negative layers for terrain, lots, planting beds and ordinary building footprints; place hazard artwork slightly above roads and below vehicles; reserve conspicuous overlays for event status. Keep roofs and shadows clear of lanes, crossings, stop lines and signals.

The often-repeated claim that vehicles are layer 256 is stale for the reviewed release. The official **1.27.1** header defines lanes `3`, junctions `4`, vehicles `340`. Do not transplant fixed layer recipes from old developer notes. Draw routines add local offsets, so these enum values alone do not guarantee every occlusion relationship. Check lane markings, junctions, hazards and vehicles together in the later runtime trial. A starting art-layer proposal is terrain `-20`, lots `-15`, roof silhouettes `-10`, road-surface hazards `10–20`, with no large decorative polygon across live lanes. [Version-pinned drawing types](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/src/utils/gui/globjects/GUIGlObjectTypes.h).

## Proposed Foundry and Market art direction

Use a dry, sun-bleached district rather than a uniformly dark board. Suggested colors are sandy ground `#C5B391`, dusty lots `#A99A80`, charcoal asphalt `#464747`, pale curbs `#D8CDB7`, rust roofs `#875843`, dark steel `#61605A`, ochre market awnings `#C18D40`, and sparse olive planting `#72785A`. These are original palette choices requiring visual review.

The foundry should read through repeated shed bays, loading aprons, roof seams, tanks, chimneys in plan and rail/service-yard textures. The market should read through small frontage modules, cloth awnings, a shaded court and a bus stop. Use coherent, subtly offset painted shadows to give roofs depth, clearly understood as illustration. Trees should occur in intentional small clusters and planted strips, with open sightlines at junctions. Desert-industrial should not become an empty tan background: lot boundaries, worn shoulders and service entrances provide scale.

Local top-down transparent PNG assets are the strongest improvement over anonymous circles: vehicles, cone/barrier silhouettes, a fallen tree, debris, and a few roof textures. Keep shadows baked and consistent. Do not cover the entire road with a painted texture that hides SUMO lane markings or falsely shifts the apparent lane center. Build these assets only in a later approved implementation; this report supplies no completed artwork.

Reserve orange for work zones, red for active incidents, and restrained cyan for communication state. Ordinary traffic should use whites, silver, slate and occasional warm colors, with buses and freight recognizable by shape. Keep hazard identity in silhouette and text so color alone is insufficient to confuse classes.

## Hazard identity at overview and close range

The table specifies proposed visual semantics. It does not claim detector classes, model accuracy, or native collision geometry.

| Object | Close aerial artwork | Overview identification |
|---|---|---|
| Pothole | Irregular broken asphalt perimeter, granular tan rim, asymmetric dark cavity, a few fracture lines; no blue fill and no perfect circular rim | Small amber diamond beside the affected lane and short `POTHOLE` label when selected; preserve actual hole footprint |
| Puddle | Smooth uneven blue-gray translucent body, soft border and a restrained pale reflection streak; no jagged crater edge | Distinct water/drop icon if needed; do not call it pothole |
| Manhole | Regular circular metal cover with nested rim and grate/crosshatch; surface-aligned, no broken cavity | Normally silent background detail; not marked hazardous unless the event says so |
| Fallen tree | Long diagonal brown trunk crossing the lane, branching limbs and separated olive crown lobes | Tree/log silhouette with `TREE` label and a closure badge only when traffic control is actually active |
| Debris | Several disconnected angular pieces and a bounded affected patch | Cluster icon and `DEBRIS`; avoid using the same car silhouette as an accident |
| Accident / stalled vehicle | Recognizable top-down vehicle body and glazing at an incident pose, separate warning triangle, optional debris | Red incident badge; normal moving cars keep normal outlines and colors |
| Construction | Cone taper, striped barriers, excavated/rough patch and service vehicle if appropriate | Orange hatched work-zone shape and `WORKS` label; a cone type string alone is insufficient |
| Ordinary traffic | Correct car/bus/truck/motorcycle proportions and heading | Muted moving silhouettes, with the followed vehicle singled out by selection |

At a 1,000 m-wide overview rendered across 1,400 pixels, a 1 m pothole occupies only about 1.4 pixels. It cannot show meaningful crater detail. At a 120 m-wide close view it occupies about 11.7 pixels. These are geometric examples, not measurements of this display. Use separate small overview symbols beside hazards, retaining truthful world-size road artwork, and inspect hazards at close range. Do not turn a 1 m hole into a 10 m crater to satisfy visibility.

Native minimum-size, exaggeration and text settings can support scale-dependent readability. They are not a complete arbitrary per-feature LOD system. Begin with two saved schemes and test actual visibility. A future controller could read zoom/boundary to adjust annotation state, but global POI/polygon mutations affect the shared scene and may affect both open views. Do not promise independent per-view event visibility from `poi.setColor`.

## Two native views and optional animation

**Overview:** fit the whole district with a margin, restrained district labels, low-clutter vehicles and visible signal state. Display short hazard labels or select-to-inspect details; avoid IDs on every building and vehicle. A fixed small legend may be a screen-relative decal, but it must not obscure the map.

**Follow view:** use a second native 2D view covering roughly 80–150 m around one currently active vehicle. Show enough road ahead to understand stopping and lane selection. Track through the native context menu or TraCI. Provide an explicit return to overview. A fixed map orientation avoids abrupt rotation; automatic heading rotation would require controller logic. Tracking does not supply perspective, pitch, occlusion or a driver's sensor image.

Reviewed APIs include `gui.getIDList()`, `gui.addView(viewID, schemeName="", in3D=False)`, `gui.setSchema`, `gui.setBoundary`, `gui.setZoom`, `gui.setOffset`, `gui.setAngle`, and `gui.trackVehicle(viewID, vehID)`. An empty tracked ID stops following. Resolve a live vehicle after insertion and handle arrival/removal. `setBoundary` preserves the window aspect ratio by expanding one axis. `gui.screenshot(viewID, filename, width=-1, height=-1)` queues a capture for the next `simulationStep`; verify that a file was produced before reporting success. These signatures were read in the installed `_gui.py`; no API call was executed. [Official GUI change API](https://sumo.dlr.de/docs/TraCI/Change_GUI_State.html).

For event-linked emphasis, `poi.setColor`, `setPosition`, `setImageFile`, `setWidth`, `setHeight`, `setAngle` and `highlight` are supported by the inspected client. A highlight can fade over a duration. Use a short low-intensity pulse for an actual new event receipt, not continuous flashing on every asset. [Official POI API](https://sumo.dlr.de/docs/TraCI/Change_PoI_State.html).

`polygon.addDynamics(polygonID, trackedObjectID="", timeSpan=(), alphaSpan=(), looped=False, rotate=True)` can attach a polygon to a traffic object and animate alpha. A non-looping alpha timeline removes the polygon after it finishes. The documented animation is alpha, not wind-driven foliage, particle smoke or general skeletal animation. `polygon.add` has no `imgFile` argument in the installed client; preload textured polygons from XML and use POIs for dynamic image swapping. [Official polygon API](https://sumo.dlr.de/docs/TraCI/Change_Polygon_State.html).

## Decoration versus traffic causality

Drawing a cone, tree, barrier, crater or wreck does **not** make vehicles avoid it. The shape layer is visual; a speed limit is a behavior rule, not a solid obstruction. The current inspected hazard XML contains `variableSpeedSign` steps, which constrain specified lanes and do not prove geometric collision with the artwork. [Variable speed sign semantics](https://sumo.dlr.de/docs/Simulation/Variable_Speed_Signs.html).

An approved implementation must map each incident to an explicit behavior policy: slowing for a passable pothole, real lane access restrictions/closure for a blocked lane, route changes where a valid detour exists, and stopped vehicles where a traffic obstacle is intentionally represented. A lane-wide closure is different from a short obstacle patch; do not imply position-local physical avoidance from `lane.setDisallowed`. Closing permissions also needs validation for already-present vehicles and routing. [Lane state API](https://sumo.dlr.de/docs/TraCI/Change_Lane_State.html), [rerouter closure definitions](https://sumo.dlr.de/docs/Simulation/Rerouter.html).

Likewise, a pulsing RSU icon or radius circle shows configured or event-linked status; it is not evidence of radio propagation, delivered messages, camera coverage or neural recognition. The native screenshot is a visualization image, not validated hazard-model input. A future demonstration must label fixture/replay/mock/genuine inference provenance separately from rendered appearance.

## Built-in OSG: narrowly bounded option

Official SUMO documentation describes generated roads/signals, external static models via decals, vehicle `osgFile` and ego navigation. It also labels the subsystem experimental and lists missing pedestrian/cyclist models, multi-vehicle performance concerns and reload artifacts. These are published capabilities and limitations, not local results. The inspected installation lacks OSG build evidence, so exclude OSG from the current promise and do not install it as part of this research. [Official native 3D documentation](https://sumo.dlr.de/docs/sumo-gui.html#3d-visualization).

## Evidence required in a later approved visual trial

1. Launch from the exact launcher and record binary version, load log and files loaded; ensure no source generator silently replaces the authored scene.
2. Inspect and save actual native overview and close-tracking captures. Check missing textures, tint, road alignment, layer occlusion, marker readability and signal visibility.
3. Demonstrate a vehicle entering/leaving the follow view, returning to overview and arriving without controller errors. Verify screenshot files after the next simulation step.
4. Record frame responsiveness and wall-clock step rate under the intended active traffic count; neither the count nor performance is established by route-file comments.
5. Show a passable hazard and a genuinely closed lane with logged behavior state. Confirm vehicles do not visibly drive through a declared solid barrier without explanation.
6. Verify that no screenshot, marker animation or visual hazard placement is reported as fresh neural perception or measured V2X delivery.

The achievable deliverable is a composed, textured, lively native aerial SUMO scene with clear hazards and a useful close tracking view. Its quality must be judged in SUMO-GUI itself after implementation; this research does not establish that the visual target has already been met.
