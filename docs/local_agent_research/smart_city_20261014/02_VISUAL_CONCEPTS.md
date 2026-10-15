# 2 — Three original smart-city concepts

Date: 15 October 2026. **Concepts and metric design proposals, not generated SUMO networks or functioning simulations.** Open [the editable schematic gallery](VISUAL_CONCEPTS.html). The drawings communicate composition/topology and explicitly do not establish rendered realism, traffic feasibility or AI detection. Preview [all three boards](visuals/concept_comparison.png); individual editable SVG and PNG files are in [visuals/](visuals/).

## A — Garden-Ring Civic District (recommended)

A compact, landscaped civic district wraps two curved boulevards around civic-square frontage. The center itself is a signalized four-way crossroad; the green square and landscaped corners sit beside it, outside the traveled junction. Low-rise mixed-use frontage faces the inner ring; schools, a clinic, a bus stop and service courtyards sit beside the radial roads. The outer boulevard has a green belt, taller trees set back from sensor sightlines, and a maintenance lay-by. The scene uses daylight, weathered asphalt, worn lane markings, concrete curbs, modest glass, textured facades and grounded proportions.

The overhead view makes circulation and RSU cooperation readable. A bus-height chase view reveals signal arms, curb geometry, poles and changing hazard visibility. A fixed roadside camera sees a designated approach. Physical RSUs have consistent poles, antenna enclosures and small status lamps; optional overlays show event IDs and observed delivery paths. Clean sensor frames exclude those overlays.

Hazard story: a pothole on an outer service approach; a clearly modeled workzone on one inner arc; a stopped damaged vehicle on another approach. A falling-tree scene can physically appear after Vehicle A passes, but remains a recognition test with possible missed detection. Alternative ring paths allow route diversion; a one-lane radial road may require stopping rather than lane changing.

This is the best first composition for the observed laptop: mostly flat terrain, daylight, repeated building/tree assets and compact sightlines. It still requires artistic work and performance testing. The ring geometry is an engineering scaffold; frontage, planting and final curve detailing should avoid the appearance of a bare geometric diagram.

### Precise interpretation of 5 km

Recommend **5,000 m of unique physical road centerline**, counted once per two-way street, excluding internal junction connectors and lane multiplicity. This defines a district, not a 5 km trip or a 5 km radius city. Geometry may be adjusted during reviewed design refinement, but preserve an explicit centerline-length tolerance (proposed ±1%).

| Interpretation | Meaning and scale | Recommendation |
|---|---|---|
| 5 km total centerline | Sum of unique street segments; connected compact demonstrator | Adopt; controls art/geometry effort and permits circulation |
| 5 km driving route | One vehicle's path; repeated laps can satisfy it without a substantial network | Use only as a journey metric, not city scope |
| 5 km² area | Region footprint; road length depends on street density | Different, larger scope requiring a separate plan |
| 5 km diameter | Bounding extent; does not specify road quantity | Larger asset/streaming design, no automatic engineering benefit |
| 5 km radius | About 78.54 km² circular area | Defer; multiplies infrastructure/asset scope without meeting a defined road-length goal |

### Original nominal network and counts

Let `r = 5000 / (6π + 8) = 186.222819 m`. Inner ring radius `r`, outer ring radius `2r`; four cardinal radial corridors each have an inner spoke and an inner-to-outer segment. There are nine junctions: the center, four inner-ring four-way junctions, and four outer-ring T junctions. The center and inner four-way junctions are signalized (five TLS systems); outer T junctions use explicit priority geometry initially. Signals are authoritative SUMO states, with visible mapped heads in the renderer.

| Physical segment family | Count | Length each | Total |
|---|---:|---:|---:|
| Inner quarter-circle arcs | 4 | 292.518120 m | 1170.072482 m |
| Outer quarter-circle arcs | 4 | 585.036241 m | 2340.144963 m |
| Radial segments (center→inner and inner→outer) | 8 | 186.222819 m | 1489.782556 m |
| **Total** | **16** | | **5000 m** (unrounded arithmetic) |

The resulting nominal footprint is approximately 745 m across and 0.436 km² within the outer circle. Sixteen physical streets correspond to 32 directed edges before junction/internal-edge generation. Sampled curve polylines approximate analytic arcs, so actual exported length must be recomputed and refined; the formula does not establish a valid SUMO network.

Proposed lane design: inner ring and the four center spokes have two lanes per direction; outer ring and the four outer connectors have one per direction, with junction turn pockets only where justified. Nominal motor-lane length before pockets/internal edges is approximately 13.829928 km. The coordinator recommends an initial 30 km/h inner/local and 40 km/h outer speed policy, below the specialist's 40/50 km/h option, to expose inference-delay feasibility at a conservative first scope. These are design choices, not calibrated traffic speeds. Propose left-hand traffic for the Indian project context; opposing turns then need the correct right-turn conflict treatment and renderer lane offsets.

### RSU placement calculated on this network

Place one mandatory physical RSU at each of the nine junctions. For each segment of length `L`, insert `ceil(L/s) - 1` interior units at equal arclength. Shared junction units are counted once; directions and lanes do not double hardware count.

`N = 9 + Σsegments (ceil(L/s) - 1)`

| Deployment | Mandatory junctions | Inner-arc interiors | Outer-arc interiors | Radial interiors | **Total** | Actual largest nominal gap |
|---|---:|---:|---:|---:|---:|---:|
| At most 200 m spacing | 9 | 4 | 8 | 0 | **21** | 195.012080 m |
| At most 500 m spacing | 9 | 0 | 4 | 0 | **13** | 292.518120 m |

The sparse plan has smaller gaps than 500 m because required junction units constrain the topology. It is therefore a **500 m maximum-spacing design**, not a claim that every gap is exactly 500 m. Count neither `5000/200` nor `5000/500` as the final deployment; those ignore deduplication and endpoint constraints. Lane offsets, setbacks and final curve geometry require a post-export placement audit.

Keep five signal-associated units camera-capable in both deployments, with only two selected sensor streams active initially and a third after a resource gate. Other units are relay/intersection stations; camera-capable units also relay. A camera's view, radio coverage and placement spacing are different quantities. Keep sensor placement and schedules identical when comparing the 21/13 station inventories. Coverage/loss parameters are experimental assumptions until a network simulator or physical measurements support them. See [network specialist study](SPECIALIST_NETWORK_RSU.md).

Proposed fleet: 72 circulating vehicles—40 cars, 12 motorcycles, 8 custom auto-rickshaw-shaped vehicles, 6 buses and 6 trucks. Allocate 20 to each boulevard loop, 24 to ring-transfer circuits and 8 to civic wedges, with rotated/reverse variants and vehicle-class route checks. Left-hand traffic, bus turning feasibility, true insertion and a continuous minimum of 50 active actors must be validated; this population is not a capacity measurement.

## B — Canal and Campus Quarter

Two irregular circulation loops sit on opposite banks of a shallow canal, joined by two bridges. A research campus, clinic and neighborhood shopping street make the banks visually distinct. The waterside road curves with the landscape, while offset cross-streets form signalized four-way approaches and a service road supplies an alternative path. Use daylight, planted terraces, textured concrete and restrained water reflection. This is not a rectangular block grid.

The bridge-level camera is the signature view: vehicles pass planted railings and RSU masts, then enter a workzone near a bridge approach. A road-surface hazard lies at a service entrance; a disabled vehicle occupies the other bank. Visual occlusion from buses and landscaping supports negative/occluded tests. Water, railings and bridge geometry increase asset effort and may narrow rerouting choices. Static camera calibration remains tractable on mostly flat road surfaces.

Complexity: moderate to high relative to A. Optional reflective water must not consume the sensor budget. Network length, four-way junction count and deduplicated RSUs would require a separate metric segment plan after this concept is selected; A's 21/13 counts do not transfer to B. Techniques can draw from [SUMO2Unity road importing](https://github.com/SimuTraffX-Lab/SUMO2Unity) and [browser PBR references](https://www.babylonjs.com/Demos/Sponza/); those references are not this scene's implementation.

## C — Foundry and Market District

An angled freight boulevard links a warehouse/depot loop with an irregular market loop. Brick industrial frontage, service courts, bus shelters, parked delivery vehicles and older shop facades create a different visual identity. Cross-streets intersect the main spine at separated four-way signals; warehouse/service lanes supply alternate routes. Warm late-afternoon lighting, worn concrete and matte metal are the default. Wet pavement and multiple shadowed lights are optional later features.

The signature camera follows a bus or truck through a market frontage toward a visible roadwork taper. A stalled freight vehicle changes lane occupancy; construction starts after an earlier vehicle passes; debris and fallen-tree recognition remain separate model gates. Mixed traffic and street occlusion are meaningful, but parked normal vehicles must be tested as hard negatives so the accident model does not equate every stopped vehicle with a crash.

Complexity: high art/behavior effort, moderate geometry if kept flat. Rich scene detail, many vehicle categories and loading activity cost more than A. No independent rigidbody traffic AI should compete with SUMO. Four-way junctions and continuous routes are mandatory in any later metric design. RSU counts require that design, not reuse of A's count.

## Visual and scientific acceptance

For any direction, review both a network overview and a close driving camera. Require convincing road/junction scale, lane continuity, traffic-signal orientation and visibility, RSU placement, believable motion and identifiable hazard assets. The sensor camera must provide actual rendered RGB; labels, warning overlays and ground-truth markers stay outside it. Scene beauty and shader capability do not demonstrate that these models recognize the assets.

Proposed initial aesthetic target: a coherent daylight 3D district, not AAA photorealism. A realistic asset/camera slice is reviewed before full geometry construction. If that slice misses the intended experience, revise the renderer/art direction rather than calling a schematic finished. The [roadmap](05_IMPLEMENTATION_ROADMAP.md) defines these approval gates.
