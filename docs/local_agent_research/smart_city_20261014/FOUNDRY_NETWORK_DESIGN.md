# Foundry-and-Market — metric network refinement

Date: 15 October 2026, Asia/Calcutta. **Conceptual design and arithmetic only.** The user selected the Foundry-and-Market visual direction. This document refines that selection; it does not approve SC0–SC1, generate a SUMO network, establish operational signal timings, or demonstrate traffic capacity. No simulation was executed for this refinement.

## The selected street composition

A diagonal freight boulevard joins three separated four-way intersections. To its northwest, an unequal warehouse/depot lobe bends behind brick foundry frontage. To its southeast, a market lobe passes older shops, sheltered bus stops and delivery courts. A western depot return and an eastern transit return provide distinct additional paths. Cross-streets connect the middle signal to both lobes. The result is an asymmetric connected district with recurring routes, rather than the earlier circular Garden-Ring graph renamed.

The metric reference has **5,000 m of unique physical road centerline**, counted once per two-way street. It has **seven operational junctions: three four-way signalized junctions and four T junctions**. Four additional vertices describe bends, not junctions. Its centerline bounding rectangle is approximately **916.911 m × 767.913 m**, about 0.704 km²; this is a bounding rectangle, not a claim about developable land or a 5 km radius city.

Recommended first visual slice, after separate implementation approval: the middle signal D2, with the diagonal boulevard, a freight-side approach, and a market-side approach visible in the same driving composition. Concrete warehouse frontage and brick sheds should face the northwest branch; shopfronts and bus shelters should face the southeast branch. Warm afternoon daylight, weathered asphalt and restrained matte metal define the initial art direction. Decorative loading courts must remain outside circulating travel lanes unless an approved physical scenario explicitly occupies a lane.

## Reproducible coordinates and exact length convention

Let each reference vertex have the integer coordinates in the table below. Reference coordinates are an original local layout, not surveyed positions. Let `Q` be the sum of all reference polyline segment lengths in the next table, calculated using Euclidean distance without rounding. Define:

`Q = 4362.471786994569`

`k = 5000 / Q = 1.146139217428531`

`metric_position(v) = k × reference_position(v)`

The exact design definition is the distance-sum expression for Q and the scale `5000/Q`; the decimal display is a convenient approximation. All tables display rounded values. One physical segment may contain multiple shape legs; it is counted once at its summed arclength.

| Vertex | Reference x, y | Metric x m | Metric y m | Function |
|---|---|---:|---:|---|
| D1 | −210, −125 | −240.689236 | −143.267402 | West four-way TLS; depot/foundry/market split |
| D2 | 0, 0 | 0.000000 | 0.000000 | Central four-way TLS; freight/market cross street |
| D3 | 170, 130 | 194.843667 | 148.998098 | East four-way TLS; foundry/market/transit split |
| F1 | −350, 100 | −401.148726 | 114.613922 | Depot T junction |
| F2 | −210, 320 | −240.689236 | 366.764550 | Foundry T junction |
| M2 | 200, −270 | 229.227843 | −309.457589 | Market T junction |
| M3 | 320, −10 | 366.764550 | −11.461392 | Transit/frontage T junction |
| W | −400, −250 | −458.455687 | −286.534804 | Western return shape bend; no intersection |
| F3 | 50, 340 | 57.306961 | 389.687334 | Upper freight shape bend; no intersection |
| M1 | −100, −330 | −114.613922 | −378.225942 | Lower market shape bend; no intersection |
| E | 400, 260 | 458.455687 | 297.996197 | Eastern return shape bend; no intersection |

Local coordinates allow negative values. A future exported network may carry an origin offset; every mapper and renderer must retain that transform. SUMO documents metric geometry, network offsets, lane shapes and distinct directed edges; the conceptual street total below is not a claim that a sum of exported lane lengths equals 5,000 m. [SUMO network representation](https://eclipse.dev/sumo/docs/Networks/SUMO_Road_Networks.html).

### Physical streets and shape paths

Here F1/F2/M2/M3 are vertex names, while F0–F4/M0–M4 in the first column are street IDs. Implementers must resolve paths by **street ID and direction**: two different streets deliberately share endpoint pairs.

| Street ID | Name | Ordered shape vertices | Unscaled length | Scaled centerline m |
|---|---|---|---:|---:|
| S1 | West diagonal boulevard | D1 → D2 | 244.386988 | 280.101511 |
| S2 | East diagonal boulevard | D2 → D3 | 214.009346 | 245.284504 |
| F0 | Depot return | D1 → W → F1 | 580.984699 | 665.889348 |
| F1 | Foundry approach | D1 → F1 | 265.000000 | 303.726893 |
| F2 | Warehouse frontage | F1 → F2 | 260.768096 | 298.876542 |
| F3 | Upper freight bypass | F2 → F3 → D3 | 502.635829 | 576.090635 |
| F4 | Freight cross street | D2 → F2 | 382.753184 | 438.688435 |
| M0 | Lower market return | D1 → M1 → M2 | 538.588973 | 617.297944 |
| M1 | Market frontage | M2 → M3 | 286.356421 | 328.204325 |
| M2 | East market approach | M3 → D3 | 205.182845 | 235.168106 |
| M3 | Market cross street | D2 → M2 | 336.005952 | 385.109599 |
| M4 | Eastern transit return | D3 → E → M3 | 545.799453 | 625.562158 |
| **Total** | **12 physical streets** | | **4362.471786994569** | **5000 exactly by definition** |

The operational graph is a connected multigraph with V=7 and E=12, giving `E−V+1=6` independent cycles. All twelve physical streets belong to at least one cycle, so no single physical-street removal disconnects the unrestricted undirected graph. There are no dangling terminal roads. Both directions on every street give 24 base directed edges before any reviewed lane-transition or shape splits and internal links. Shape bends W/F3/M1/E can remain polyline vertices rather than extra traffic junctions.

A segment-intersection arithmetic check found no crossing between nonincident reference shape legs. This check establishes a property of the mathematical centerlines only. Road widths, junction footprints, turning pockets, bend fillets and buildings still require a geometric clearance audit.

**These sharp reference bends are construction controls, not final drivable corners.** Smooth/fillet W, F3, M1 and E before accepting truck or bus circulation. Check swept turns, sightlines and the usable distance between junction boundaries. Any curve smoothing changes arclength: recompute the canonical centerline total and rescale/revise the reviewed metric plan before reporting final 5 km compliance. Preserve the distinction between canonical physical street length, serialized edge length, lane length and internal connection length. Proposed final geometry tolerance remains 5,000 m ±1%; exact reference arithmetic does not establish exported compliance.

## Four-way signals and class-safe traffic proposal

| Junction | Spine arms | Cross-street arms | Degree and intended control |
|---|---|---|---|
| D1 | S1 and F0 | F1 and M0 | Four; operational TLS |
| D2 | S1 and S2 | F4 and M3 | Four; operational TLS |
| D3 | S2 and M4 | F3 and M2 | Four; operational TLS |
| F1 | F0/F1 and F2 | — | Three; reviewed priority/stop control initially |
| F2 | F2/F3 and F4 | — | Three; reviewed priority/stop control initially |
| M2 | M0/M1 and M3 | — | Three; reviewed priority/stop control initially |
| M3 | M1/M2 and M4 | — | Three; reviewed priority/stop control initially |

The words “spine arms” describe the intended principal flow, including the outer return at the two end signals. They do not imply geometrically straight-through connections at D1 or D3. Stop lines, internal connections, protected movements and render heads must be derived from the eventual exported network. At least these **three real signal controllers** must control permitted traffic movements; decorative light changes do not satisfy the requirement. No phase durations or conflicting green sets are approved here.

Retain the proposed left-hand driving convention. Review opposing right turns, visibility and turn pockets using the actual lane/foe/link mapping. For a reproducible first experiment, use a reviewed fixed signal baseline; future actuation is a separately declared factor. Native safety and red-light compliance remain mandatory.

Proposed lane and class policy, pending turning/connection validation:

| Street family | Nominal motor lanes | Proposed speed policy | Class/use rule |
|---|---|---|---|
| S1/S2 diagonal boulevard | Two per direction, about 3.5 m wide | 35 km/h initially | General motor fleet, with any merge subject to legal connections and safe gaps |
| F1/F2/F3/F4 freight streets | One per direction, about 3.5 m wide | 30 km/h | Cars, motorcycles, small passenger vehicles; trucks/buses only after swept-turn validation |
| F0 depot return | One per direction, about 3.5 m wide | 20–25 km/h | Optional depot circulation; keep long vehicles out until its smoothed bend passes review |
| M0 lower market return | One per direction, about 3.2 m wide | 20–25 km/h | Cars, motorcycles and custom small passenger vehicles; trucks excluded initially |
| M1/M2/M3 market/transit streets | One per direction, about 3.5 m wide | 25–30 km/h | Cars/small vehicles, with bus admission conditional on geometry; through trucks excluded initially |
| M4 eastern transit return | One per direction, about 3.5 m wide | 25 km/h | Transit/car circulation only after bend and visibility review; trucks excluded initially |

Widths and speeds are design inputs, not measured safe operating values. A custom auto-rickshaw-shaped vehicle needs an explicitly documented supported class and behavior; a mesh name is not a native driving model. The SUMO vehicle/type/route documentation provides the later configuration contract. [Vehicle types and connected routes](https://eclipse.dev/sumo/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html).

The proposed motor lane length is **11.050772 km**, before pockets/internal connections: two directional lanes on all 5,000 m, plus two additional lanes on the 525.386015 m S1+S2 spine. A one-lane street offers no adjacent same-direction merge; safe stop or class-valid diversion is the relevant alternative. Parking/loading scenery cannot create a hidden second travel lane.

### Recurring routes and useful alternatives

Each arrow below names a **physical street with direction**; a minus sign means the reverse of the shape order in the length table. Closed node strings are omitted where parallel endpoint streets would be ambiguous.

| Route family | Closed directed street sequence | Nominal lap length m | Suggested fleet allocation |
|---|---|---:|---:|
| Freight circulation | F1 → F2 → F3 → −S2 → −S1 | 1704.080085 | 20 |
| Market circulation | M0 → M1 → M2 → −S2 → −S1 | 1706.056390 | 26 |
| Cross-district transfer | F1 → F2 → −F4 → M3 → −M0 | 2043.699413 | 12 |
| Cross-town transfer | F4 → F3 → −M2 → −M1 → −M3 | 1963.261100 | 8 |
| Depot return | F0 → −F1 | 969.616241 | 4 |
| Eastern transit return | M4 → M2 | 860.730264 | 2 |
| **Total proposed initial fleet** | Reverse/rotated variants permitted after route review | | **72** |

The route arithmetic uses the nominal polyline lengths. These sequences establish logical closure, not permission to execute the turns. Review every last-to-first connection, vehicle class, turning radius, lane transition and signal mapping. Use the previously proposed 40 cars, 12 motorcycles, 8 custom auto-rickshaw-shaped vehicles, 6 buses and 6 trucks as a starting composition; place trucks on validated freight routes and buses on validated spine/freight/transit routes. Category assignment cannot override a route's permissions.

Examples of logical alternatives: F0 and F1 provide distinct D1↔F1 paths; M4 and M2 provide distinct D3↔M3 paths; F4 can be bypassed through D2↔D1↔F1↔F2; S1 can be bypassed through D1↔F1↔F2↔D2. Some alternatives are unsuitable for particular classes until reviewed. No U-turn or unsafe reverse motion is necessary to define the recurring graph. Sustained demand, blocked routes and safe route renewal still require later traffic execution.

The demonstration gate remains **at least 50 actually active vehicles at every authoritative sample after warmup**, initially over a declared 20-minute main window and later a longer robustness run. Seventy-two planned vehicles is capacity margin only. Staggered safe admission, route renewal and accounting for arrivals/removals/teleports must be validated; pending departures are not active vehicles.

## RSUs: seven mandatory anchors, then arclength subdivision

Place one unique infrastructure anchor at each operational junction: **D1, D2, D3, F1, F2, M2, M3**. Three are TLS-associated. The shape bends do not create four extra mandatory units. Poles/cabinets stand at reviewed roadside setbacks; the junction-center anchor is a registry reference, not a physical pole in the road.

For a physical street of length L and spacing limit s, use `g = ceil(L/s)` equal arclength gaps. Put `g−1` interior stations at chainages `jL/g`, for `j=1…g−1`. Count shared junction stations once; do not double for opposing lanes, direction, cameras or signal poles. Interpolate chainage along the actual polyline rather than using endpoint Euclidean distance, especially on F0/F3/M0/M4.

`N(s) = 7 + Σ12 streets (ceil(L/s) − 1)`

| Street | Length m | Interior units, s=200 m | Resulting gap m | Interior units, s=500 m | Resulting gap m |
|---|---:|---:|---:|---:|---:|
| S1 | 280.101511 | 1 | 140.050756 | 0 | 280.101511 |
| S2 | 245.284504 | 1 | 122.642252 | 0 | 245.284504 |
| F0 | 665.889348 | 3 | 166.472337 | 1 | 332.944674 |
| F1 | 303.726893 | 1 | 151.863446 | 0 | 303.726893 |
| F2 | 298.876542 | 1 | 149.438271 | 0 | 298.876542 |
| F3 | 576.090635 | 2 | 192.030212 | 1 | 288.045318 |
| F4 | 438.688435 | 2 | 146.229478 | 0 | **438.688435** |
| M0 | 617.297944 | 3 | 154.324486 | 1 | 308.648972 |
| M1 | 328.204325 | 1 | 164.102162 | 0 | 328.204325 |
| M2 | 235.168106 | 1 | 117.584053 | 0 | 235.168106 |
| M3 | 385.109599 | 1 | **192.554800** | 0 | 385.109599 |
| M4 | 625.562158 | 3 | 156.390540 | 1 | 312.781079 |
| **Interior subtotal** | **5000** | **20** | **Max 192.554800** | **4** | **Max 438.688435** |
| **Mandatory junction units** | | **7** | | **7** | |
| **Unique physical inventory** | | **27** | **At most 200 m** | **11** | **At most 500 m** |

Thus the selected Foundry design uses **27 dense / 11 sparse RSUs**, replacing Garden-Ring's **21 / 13** counts. The actual maximum gaps are **192.554800 m / 438.688435 m**, not exactly 200/500 m. This is an arclength-spacing design, not a coverage or radio-performance measurement. Recalculate inventories if curves, junction policy or street segmentation change. Artificial degree-two splits must not inflate the mandatory junction count.

Stable proposed IDs: junction `RSU_J_<vertex>`, interior `RSU_<street>_D_<j>` for dense and `RSU_<street>_S_<j>` for sparse. Camera equipment is attached to a physical station, not counted as another RSU. Keep three signal-associated sites camera-capable, with two active initially and a third only after resource/recognition review. Fix camera/model roles before scenario scheduling; keep the same sensing sites and cadence in both density experiments. Do not imply every one of the 27 sites executes three models continuously.

Camera visibility, infrastructure spacing, radio peer reach and useful warning lead distance remain separate quantities. A relay cannot infer an occluded stalled truck or fallen tree from the independent physical truth ledger. Actual clean pixels and genuine model/temporal evidence still determine whether a warning exists. Application delay/loss/backhaul assumptions retain their explicit mode label; these counts establish no RF/ETSI claim.

## Arithmetic verification and required future evidence

The document's arithmetic check recalculated all twelve Euclidean polyline lengths and the common scale, summed the scaled lengths to 5,000 m (floating-point display 4999.999999999999), checked vertex degrees and nonincident straight-leg crossings, computed all six proposed lap lengths, and computed interior station totals and maximum chainage gaps independently for both spacing limits. No SUMO/TraCI, ONNX inference, scene renderer, traffic-count trial or installation was run.

Remaining implementation gates: smooth bend geometry and reconcile physical length; generate a separately named network after approval; verify three operational TLS and all foe/link states; inspect truck/bus swept paths and class-specific alternative routes; validate continuous 50+ active traffic without hidden removals; align lane/road/render coordinates; inspect actual sensor visibility and genuine recognition; audit roadside setbacks; measure application delivery and control outcomes. The legacy corridor and M1 artifacts remain outside this document's changes.

## Required consistency updates to the planning package

The coordinator should update the selected-direction summary, gallery and package index to Foundry-and-Market; link this metric refinement rather than duplicating Garden geometry. Replace selected-scope references to nine junctions/five TLS/16 streets/32 directed edges/21–13 RSUs with **seven junctions/three TLS/12 physical streets/24 base directed edges/27–11 RSUs**. Update the selected nominal footprint to approximately **917 m × 768 m**, replace radial/ring route examples with the named street cycles, and replace five signal-associated sensing sites with three. Preserve Garden-Ring as an alternative concept with its own valid historical arithmetic. SC2's acceptance becomes three valid four-way TLS for this selected graph; SC5 compares the new 27/11 deployment inventories. SC0–SC1 implementation approval remains pending.
