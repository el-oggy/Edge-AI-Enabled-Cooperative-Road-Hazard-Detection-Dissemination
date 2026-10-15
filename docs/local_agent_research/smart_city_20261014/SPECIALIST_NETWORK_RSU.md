# Smart-city network, traffic and RSU specialist research

Research completed 15 October 2026, Asia/Calcutta. **Proposal only.** No road network was generated, SUMO was not executed, and source, models, XML, project memory and Git history were not changed by this specialist. This note supplies the city/traffic/RSU part of the new master brief; rendering and neural-perception feasibility are assessed by the other specialists.

## 1. Decision and evidence boundaries

Recommend interpreting the requested size as **approximately 5,000 m of unique physical road centerline**, counting a bidirectional street once. Build a compact planned urban district with two curved boulevards, four radial connectors and four civic-center approaches. This original **Garden-Ring District** has five genuine four-way signalized junctions, four three-way junctions, multiple closed routes and alternatives around every logical road segment. It is not a rectangular street grid. Its mathematical reference geometry is specified below so lengths, placements and experiment configurations can be reproduced.

The proposed main demonstration uses 72 circulating motor vehicles, providing margin above the required 50. Neither that population nor local rendering performance has been demonstrated on this proposed network. Admission, queueing, signal control, route validity and frame rate must pass separate approved implementation gates.

Read-only inspection covered the new master prompt, project `.agents/rules/memory.md`, the previous three research documents, the M1 implementation report, `simulation/corridor_1km/corridor.{nod,edg,net,rou,add,sumocfg}` and its cooperative controller. The current net contains two actuated TLS programs; its routes are finite journeys supported by continuous demand. Continuous demand is different from the same vehicles circulating. The controller still contains fixed hazard/confidence/latency values and an application-level warning mechanism. Its scenery and POIs do not establish obstacle physics or real radio networking. The previous research recorded a maximum of 29 active vehicles in a traffic-only corridor probe; that is historical evidence, not a fresh run in this task.

The current M1 report records real forwards for all three models and 33 passing focused tests. It also records missing labeled negatives and temporal clips. Those records supersede the old memory's inference-defect descriptions for M1, but do not validate a city-scale perception/traffic loop. No inference files were edited or benchmarked here.

## 2. Resolve the 5 km ambiguity before geometry implementation

| Interpretation | Mathematical scope | Practical implication |
|---|---|---|
| 5 km of unique road centerline | Sum of physical corridor lengths about 5,000 m | Recommended first city district; controllable scene and adequate route variety |
| 5 km of directed edges | Opposite directions counted separately | Only about 2.5 km of physical streets if all streets are bidirectional; does not match the recommendation |
| 5 km² of land | Square side 2.236 km, or disk radius 1.262 km | A larger land/asset envelope; does not define road length or number of vehicles |
| 5 km diameter | Disk area 19.635 km² | Requires a substantially broader environment if filled with streets and buildings |
| 5 km radius | Disk area π×5² = 78.540 km² | About 180 times this proposal's circular road footprint; a different asset, sensing and traffic project |

These areas are calculations, not workload or frame-rate estimates. A sparse 5 km-radius region might contain only 5 km of roads; filling it with a city would imply much more road, infrastructure and scenery. Approval should name the road-length convention and footprint independently.

The recommended mathematical network fits within a 744.891 m diameter circle, about 0.435788 km². Landscape and skyline can extend beyond this envelope, with distant low-detail geometry. This is a substantial **driving district**, not an entire metropolitan area. Lane lengths, vehicle-distance traveled and decorative paths are separate quantities.

## 3. Original topology alternatives and selection

Three topology directions were considered independently of the old corridor:

1. **Garden-Ring District — recommended reference design.** A civic quarter bounded by two curved boulevards, with four radial neighborhoods and a central cross junction. Continuous rings give circulation; quadrant routes give local journeys; center spokes provide crossing traffic. The precise circle geometry makes road-length and RSU arithmetic transparent. Landscaped setbacks, building heights, road materials and district land uses can vary without altering the approved traffic graph.
2. **River-Crescent District — optional alternative.** Two unequal crescent corridors follow a riverbank, linked by crossing streets through civic and warehouse areas. Through streets extend across the crescents to create real four-way junctions, while end connectors close circulation loops. It offers a stronger landscape narrative and more irregular geometry, but exact lengths, junction angles and station counts need a separate approved parameterization. The Garden-Ring 21/13 station counts must not be copied onto it.
3. **Offset Twin-Loop District — optional alternative.** Two unequal loops surround a residential quarter and a freight/service quarter; a linking boulevard crosses connecting streets at several signalized junctions. Bridges or an embankment can separate scenery zones without adding an uncontrolled grade crossing. It favors freight/bus route contrast and independent hazards, but lane transitions and turning feasibility need more geometry review than the reference design.

These are original concepts, not functioning simulations or claims of a reproduced real city. The circular reference is selected for experiment clarity and controlled construction effort. If an alternative is selected, its topology, lengths and RSU inventory must be recalculated before implementation. Do not deform a map artistically while retaining obsolete distances.

SUMO permits explicit node/edge descriptions, geometry shapes, lanes and connections; NETEDIT supports visual network editing. That is the appropriate way to turn an approved original concept into a traffic network, followed by connection and junction review. [SUMO plain XML](https://eclipse.dev/sumo/docs/Networks/PlainXML.html), [NETEDIT](https://eclipse.dev/sumo/docs/netedit.html).

## 4. Exact Garden-Ring reference geometry

Use meters in a local Cartesian plane, x east and y north. Let:

`r = 5000 / (6π + 8) = 186.2228192753457 m`.

The inner boulevard has radius r; the outer boulevard has radius 2r. Divide each at north/east/south/west into four quarter-circle road segments. Connect corresponding inner/outer junctions radially, then connect every inner junction to the center. Both travel directions exist on every physical segment.

| Junction ID | x m | y m | Physical approaches | Proposed control |
|---|---:|---:|---:|---|
| C | 0 | 0 | 4 | Traffic signal |
| I_N | 0 | +186.222819 | 4 | Traffic signal |
| I_E | +186.222819 | 0 | 4 | Traffic signal |
| I_S | 0 | −186.222819 | 4 | Traffic signal |
| I_W | −186.222819 | 0 | 4 | Traffic signal |
| O_N | 0 | +372.445639 | 3 | Priority/yield connector junction |
| O_E | +372.445639 | 0 | 3 | Priority/yield connector junction |
| O_S | 0 | −372.445639 | 3 | Priority/yield connector junction |
| O_W | −372.445639 | 0 | 3 | Priority/yield connector junction |

The inner-junction tangential boulevard approaches are perpendicular to their radial approaches. These are four-way junctions on a curved boulevard, **not roundabouts**; circular shape alone does not imply roundabout priority. At C the four spokes cross under signals. The three-way outer junctions give boulevard traffic priority and require connector traffic to yield, subject to a generated foe/connection matrix.

Define angles θ_N=π/2, θ_E=0, θ_S=−π/2, θ_W=π. A circle centerline is `p(θ)=(R cos θ,R sin θ)`. Each arc uses its appropriate consecutive 90° interval. A radial segment is the straight interpolation between its node coordinates. This provides unambiguous analytic reference geometry and chainage; a later renderer and SUMO importer consume the same approved geometry registry.

| Physical segment IDs | Geometry | Count | Length each m | Subtotal m |
|---|---|---:|---:|---:|
| I_N–I_E, I_E–I_S, I_S–I_W, I_W–I_N | Inner quarter arcs, R=r | 4 | πr/2 = 292.518120483103 | 1170.072481932412 |
| O_N–O_E, O_E–O_S, O_S–O_W, O_W–O_N | Outer quarter arcs, R=2r | 4 | πr = 585.036240966206 | 2340.144963864823 |
| C–I_N, C–I_E, C–I_S, C–I_W | Straight center spokes | 4 | r = 186.222819275346 | 744.891277101383 |
| I_N–O_N, I_E–O_E, I_S–O_S, I_W–O_W | Straight outer connectors | 4 | r = 186.222819275346 | 744.891277101383 |
| **Total** | **(6π+8)r** | **16** | | **5000.000000000000** |

The logical graph has 9 junctions, 16 undirected segments, 32 base directed edges and 8 independent cycles (`E−V+1`). Every physical segment belongs to a cycle, so removing a single segment does not disconnect the undirected graph. Directed vehicle-class feasibility still depends on permitted junction turns and lane connections; graph connectivity alone is insufficient.

This exactly 5,000 m quantity is **nominal center-to-center physical reference centerline**. Polyline tessellation approximates arcs; lane offsets and junction trimming change SUMO lane/edge lengths. Internal junction links and added turning-pocket splits also change serialized edge counts. Record all of those after network generation and reconcile their tolerances instead of asserting that every SUMO length sum equals 5,000 m. Degree-two geometry splits are not extra traffic intersections or extra mandatory RSUs.

Suggested land-use narrative: civic buildings and a transit stop near the center; market/residential frontage along the inner boulevard; landscape buffers and a canal/pond beyond part of the outer boulevard; a service/worksite quarter in one quadrant. The central traffic junction stays clear—put plazas and monuments beside roads, not inside an invented drivable conflict area. Roadside trees and buildings must not conceal all signal heads or camera sightlines.

## 5. Lanes, speeds and signals

| Corridor family | Nominal motor lanes | Initial design speed | Purpose |
|---|---|---|---|
| Inner boulevard | 2 per direction | 40 km/h, 11.111 m/s | Workzone/pothole lane-response experiments and visible urban traffic |
| Center spokes | 2 per direction | 30 km/h, 8.333 m/s | Signal crossing traffic, transit/civic access and safe recovery |
| Outer boulevard | 1 per direction | 50 km/h, 13.889 m/s | Scenic circulation and full-lane obstruction/rerouting experiments |
| Outer connectors | 1 per direction | 30 km/h, 8.333 m/s | District transfers and approach-route warning relevance |

This is about **13.829928 motor lane-km**, excluding turning pockets, sidewalks and bicycle facilities. One-lane outer roads deliberately make safe stopping or rerouting meaningful; they do not permit an impossible adjacent-lane merge. Inner two-lane roads support a merge request subject to actual route/gap checks. Lane count should follow the scenario, not be increased solely for prettier traffic.

Propose left-hand traffic for the Indian project context, with explicit approval and consistent road connections, lane indexing and renderer offsets. In left-hand traffic, the turn crossing oncoming traffic is generally the **right turn**; do not blindly copy right-hand-network protected-left phases. This note uses “opposing turn” to avoid that mistake.

Five signal controllers—C and all four I_*—must own actual controlled movements. Prefer reviewed fixed timing for the first reproducible baseline, then compare actuated control as a separately declared factor. A provisional 80 s cycle comprises main through/compatible turns (26 s green), cross through/compatible turns (22 s), opposing turns from main approaches (8 s), and opposing turns from cross approaches (8 s), each followed by a provisional 3 s yellow and 1 s all-red. This arithmetic totals 80 s. **These timings are a starting proposal, not a traffic-engineering clearance calculation or optimized program.** Verify stopping/clearance distances, turning pockets, vehicle lengths and every conflicting movement before adopting them. A radial approach with only one lane may require a pocket or an alternative incoming-stage plan; those splits must preserve the approved physical centerline.

SUMO signals control lane-to-lane links, and link indices are network-specific. Obtain the controlled links programmatically and drive each rendered signal head from its corresponding SUMO state; never run a separate decorative light clock. SUMO supports static and actuated controllers, with generated plans requiring review when movement priority changes. [SUMO traffic lights](https://eclipse.dev/sumo/docs/Simulation/Traffic_Lights.html), [TLS state retrieval](https://eclipse.dev/sumo/docs/TraCI/Traffic_Lights_Value_Retrieval.html).

Before AI work, prove that each signal cycles, vehicles wait on red, legal greens release them, and no protected phase simultaneously grants conflicting priority. Traffic-demand-based signal offsets are a later optimization, not an initial promise of a green wave.

## 6. Continuous traffic and the 50-vehicle requirement

Use a **fixed circulating fleet of 72**, rather than uncontrolled flows that accumulate until jammed or terminate without replacement. A proposed fleet is 40 cars, 12 motorcycles, 8 auto-rickshaw-shaped small passenger vehicles, 6 buses and 6 trucks. Auto-rickshaw behavior is an explicitly configured custom vehicle type, not an invented native `rickshaw` class. Connectedness is an independent attribute from body type. Buses/trucks initially favor the outer loops; permission and swept-turn feasibility must be checked before admitting them to each inner route.

| Route family | Example closed node sequence | Reference length m | Initial population allocation |
|---|---|---:|---:|
| Outer circulation | O_N→O_E→O_S→O_W→O_N, plus reverse | 2340.144964 | 20 |
| Inner circulation | I_N→I_E→I_S→I_W→I_N, plus reverse | 1170.072482 | 20 |
| District circuit | I_N→O_N→O_E→I_E→I_N; rotations and reverse | 1250.000000 | 24 |
| Civic wedge | C→I_N→I_E→C; rotations and reverse | 664.963759 | 8 |
| **Total** | Rotated/directional variants balance movements | | **72** |

Route sequences name logical nodes; implementation must convert them to connected directed edges and permitted turns. Give vehicles reproducible route/type/desired-speed assignments and staggered departure positions across safe available lanes. Insertions remain subject to SUMO safety. Do not use unsafe movement or teleportation to make the active count reach a target.

SUMO's `repeat` attribute repeats a route a finite number of times; it is not an unlimited-route flag. Validate the last-to-first connection of each cycle. For a bounded one-hour demonstration, 200 repeats of the shortest proposed route provides ample nominal distance even at the 50 km/h maximum, provided rerouting preserves circulation. Keep the coordinator's duration explicit; TraCI does not stop simply because the usual `--end` time was supplied. For arbitrary session duration, a reviewed controller can replace/extend the future route with a connected loop before its end, starting from the current edge and preserving route legality. [Vehicle/routes and repeated routes](https://eclipse.dev/sumo/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html), [TraCI lifecycle](https://eclipse.dev/sumo/docs/TraCI/index.html), [vehicle route changes](https://eclipse.dev/sumo/docs/TraCI/Change_Vehicle_State.html).

Start the main measurement window only after warmup and actual insertion of at least 50 vehicles. Log active vehicle count every step and require a minimum of 50 throughout the declared main window; pending departures do not count. A starting 72 fleet is a design margin, not proof of simultaneous activity. Log any disappearance, collision removal, teleport, arrived route, missed extension or vehicle-ID replacement. If 72 leads to excessive congestion, calibrate signal timings and route proportions while preserving the requirement; do not silently report attempted insertions as active traffic.

Begin with lane-based motorcycles; a sublane model can later add continuous lateral motion and side-by-side two-wheelers, with extra calibration and performance checks. Pedestrians and bicycles require their own connected facilities and signal conflict review and should be added after the core motor-traffic baseline. They must not be decorative characters claimed to have simulated interaction. [SUMO sublane model](https://sumo.dlr.de/docs/Simulation/SublaneModel.html), [bicycle modeling](https://sumo.dlr.de/docs/Simulation/Bicycles.html), [pedestrians](https://sumo.dlr.de/docs/Simulation/Pedestrians.html).

## 7. RSU placement: exact counts and no double-counting

Use one unique infrastructure station at each of the nine logical junctions. Five are traffic-signal-associated; four serve the outer connector junctions. Subdivide each physical segment of length L into `k=ceil(L/s)` equal chainage gaps for a deployment spacing limit s. Place `k−1` interior stations; share the endpoint station with every incident road.

**Count formula:** `N(s)=|V|+Σ_e(ceil(L_e/s)−1)`.

Do not count reverse edges, lanes, individual signal poles or each camera as an additional RSU. This policy fixes the junction inventory across dense and sparse experiments. Station count would differ under a different endpoint policy.

| Family | Segments | Length each m | Interior stations per segment at 200 m | At 500 m |
|---|---:|---:|---:|---:|
| Inner quarter arc | 4 | 292.518120 | 1 | 0 |
| Outer quarter arc | 4 | 585.036241 | 2 | 1 |
| Center spoke | 4 | 186.222819 | 0 | 0 |
| Outer connector | 4 | 186.222819 | 0 | 0 |
| Junction station inventory | 9 unique | Shared endpoints | 9 | 9 |
| **Total** | | | **9+4+8 = 21** | **9+4 = 13** |

At 200 m, inner-arc stations lie at L/2=146.259060 m; outer-arc stations at L/3=195.012080 m and 2L/3=390.024161 m. The largest chainage gap is 195.012080 m; radial gaps are 186.222819 m.

At 500 m, each outer arc has a midpoint station at 292.518120 m. No inner-arc station is added because the junction-to-junction length is already below 500 m. The largest chainage gap is 292.518120 m, so this is **a 500 m spacing-limit configuration**, not a claim that every pair is 500 m apart. Mandatory junction units constrain how sparse this compact graph can become. If the desired experiment requires actual 500 m gaps, it needs another topology/endpoint policy and a new count.

Explicit IDs can be stable: `RSU_C`, `RSU_I_N/E/S/W`, `RSU_O_N/E/S/W`; dense interior `RSU_I_NE_MID`, `RSU_I_ES_MID`, `RSU_I_SW_MID`, `RSU_I_WN_MID` and `RSU_O_<quadrant>_1/2`; sparse interior `RSU_O_<quadrant>_MID`. The named station anchor follows the corresponding analytic curve and chainage. A physical cabinet/pole is offset to a reviewed roadside location; this offset must be recorded for coverage calculations. An intersection anchor at its center is a registry reference, not permission to place a pole in traffic.

### Infrastructure roles

Hold the five signal-associated stations' sensing capability constant between configurations; the remaining 16 or 8 are relay/forwarding stations. A sensing-capable station may also relay. Camera capabilities and active inference jobs are separate: five stations do not imply running fifteen simultaneous model pipelines on the laptop. Select measured, scheduled sensor feeds and record view/model assignments.

One RSU with one forward-looking camera does not observe four approaches or an entire radius. Specify each camera's pose, orientation, FOV, resolution, visible-road footprint and occlusions. Initial hazards should occur inside a reviewed sensing view or be observed by a genuine mobile sensing vehicle. Hazards between blind static views remain undetected until a real observation arrives; relay units cannot infer an unseen tree merely because the physical scenario ledger knows its location.

## 8. Communication assumptions and cooperative event policy

Spacing, visual sensing range, radio radius and warning distance are four separate parameters. For a straight road with ideal symmetric radial coverage R and station gap g, `2R≥g` may cover vehicles between sites; it does **not** establish station-to-station reception, which requires `R≥g` under that same ideal model. Buildings, antenna offsets and road curvature complicate both. No RSU range has been measured in this project.

A useful initial mode is **modeled application communication**: an explicit RSU forwarding graph, delivery queue, configured latency/jitter/loss, recipient policy and receipts. Distinguish modeled infrastructure backhaul from range-gated V2I. A road-neighbor forwarding graph is not proof of wireless connectivity; if radius-gated peer links are enabled, construct those links from recorded station positions and verify connectivity for each parameter set. As one geometric illustration, sparse outer-midpoint stations can become isolated at an assumed 250 m peer radius even when other junction stations communicate through the center. Do not silently use a complete graph to hide that effect.

Proposal for the initial deterministic adapter:

- Validate event schema, model/input evidence, mapped edge/lane/chainage, observation time, eligibility time, revision and expiration before dissemination.
- Deduplicate on canonical `(event_id, revision)` rather than only hazard class. One event may have multiple model observations; nearby concurrent hazards remain separate events.
- Match observations into a located event only under reviewed spatial/time/domain gates. A bounding box alone is not geographic location.
- Keep a bounded seen-event cache; reject duplicates, expired updates, stale revisions and impossible future availability. Forward once per relevant peer/revision, with bounded hop count and a declared retry policy.
- Forward toward relevant upstream approaches and routes. Warn an equipped vehicle only if its permitted remaining path encounters the affected road/lane ahead, the alert is still valid and the configured communication relation admits delivery. Euclidean proximity alone can warn vehicles on unrelated parallel roads.
- Log send/receive/drop reasons, sender/receiver IDs, event revision, hops, bytes, modeled delay and expiry. Renderer links animate those receipts; cabinet lights do not invent receptions.
- Clearance is a new revision/cancellation with explicit evidence/policy. Scenario ground truth may physically clear an obstacle, but that does not automatically establish that a sensing system recognized clearance. A conservative timeout may expire an old alert; it is not a verified clear-road detection.

Suggested sensitivity values—**experimental assumptions, not real network measurements**—are delivery delay 0/50/200 ms, packet-loss probability 0/0.01/0.05/0.10, and range 150/250/300 m. Hold the model/input timing, sensing inventory and traffic seed fixed while comparing 21 vs 13 stations. Separate reliable infrastructure-backhaul experiments from multihop wireless-range experiments. Independent Bernoulli loss is a simplified model, not fading, congestion or MAC contention.

Route-aware relevance can use recorded routes and distance queries. TraCI subscriptions batch repeated vehicle-state retrieval; context subscriptions help retrieve neighbors, but the final recipient policy must still check route, direction, connection permissions and event lifetime. [Python subscriptions](https://eclipse.dev/sumo/docs/TraCI/Interfacing_TraCI_from_Python.html), [context subscriptions](https://eclipse.dev/sumo/docs/TraCI/Object_Context_Subscription.html), [vehicle state retrieval](https://eclipse.dev/sumo/docs/TraCI/Vehicle_Value_Retrieval.html).

## 9. Do we need a network simulator?

| Approach | What it supports | Missing evidence / cost | Recommendation |
|---|---|---|---|
| Deterministic application adapter | Warning causality, configured delay/loss, expiry, suppression, relevance and observed traffic effects | No PHY/MAC, interference, contention, standards conformance or actual radio proof | First approved integration |
| Veins + SUMO + OMNeT++ | Coupled vehicular mobility and communication simulation; lower-layer vehicular models | Requires version pinning, builds, radio parameters and a synchronized integration test | Preferred later radio-research extension |
| Artery + Vanetza + OMNeT++/SUMO | ITS-G5 facilities/network stack experiments using selected communication models | Additional stack/configuration work; not automatic DENM conformance | Add if an ITS-G5 research question warrants it |
| ns-3 with a reviewed V2X module | Detailed network studies when the chosen module and mobility coupling are appropriate | Current WAVE availability differs from old tutorials; extra mobility/event coupling and module validation | A specialist option, not the default |

Veins couples SUMO mobility with OMNeT++ over TraCI. It supplies models and an execution framework for application-specific experiments. That is a stronger foundation than a Python radius dictionary for evaluating packet-level vehicular networking, but still simulation rather than physical radio validation. [Veins documentation](https://veins.car2x.org/documentation/), [Veins modules](https://veins.car2x.org/documentation/modules/), [Veins releases](https://veins.car2x.org/download/).

Artery wraps ITS-G5 entities implemented by Vanetza and uses Veins or INET communication models rather than providing its own wireless model. Prefer it when stack behavior is an explicit objective, not merely to animate warning rings. Inspect its pinned installation dependencies before proposing a bounded build. [Artery architecture](https://artery.v2x-research.eu/architecture/), [Artery installation](https://artery.v2x-research.eu/install/).

**Version caveat:** official ns-3 release notes show that the **WAVE module was removed in release 3.39**, citing lack of maintenance and outstanding issues. Old 3.35-era `Wifi80211pHelper` examples must not be presented as turnkey current ns-3 capability. A pinned older WAVE environment or an independently reviewed maintained V2X contribution needs a specific justification and compatibility test. Ordinary Wi-Fi propagation/error models are not equivalent to validated ITS-G5 or C-V2X sidelink. [Official release notes, Release 3.39 section](https://gitlab.com/nsnam/ns-3-dev/-/raw/ns-3.44/RELEASE_NOTES.md), [current Wi-Fi model documentation](https://www2.nsnam.org/docs/models/html/wifi-user.html).

Do not let a future network framework introduce a second unsynchronized traffic authority. Its SUMO synchronization mechanism must be reconciled with the chosen coordinator. Multiple TraCI clients require declared order and all clients stepping; an engine integration using its own manager is not interchangeable with the current single-owner Python controller. [TraCI multiple clients](https://eclipse.dev/sumo/docs/TraCI/index.html).

## 10. Safe response and meaningful experiments

Retain native speed/lane-change safety. SUMO documents default speed mode 31 and lane-change mode 1621; record the actual values per vehicle, since a previous script may change them. A `changeLane` call is a request; observe the resulting lane, speed, leader/follower gap and route continuation. On a one-lane approach, stop safely or reroute through a reviewed alternative instead of requesting a nonexistent merge. Release temporary speed constraints deliberately after observed clearance and policy recovery. [TraCI vehicle control and safety modes](https://eclipse.dev/sumo/docs/TraCI/Change_Vehicle_State.html).

Use 0.1 s traffic steps as a proposed starting resolution, with driver action-step length and car-following headway selected together. Headway `tau` below decision/step length can undermine safety. A separate warning-reaction delay should be logged rather than pretending that the 0.1 s simulation step equals human reaction time. Do not disable collision checks, violate lights or shrink safety gaps to create visually dramatic hazards. [SUMO safety](https://eclipse.dev/sumo/docs/Simulation/Safety.html), [basic step/action-step definitions](https://eclipse.dev/sumo/docs/Simulation/Basic_Definition.html).

Warning range must include genuine detection availability, not just network delay. An illustrative design equation is:

`d_required = v × (t_observation_to_confirmation + t_delivery + t_driver) + v²/(2a_comfort) + margin`.

This is an engineering screening equation, not certified braking physics. Example assumptions of three sequential 2 s model frames, 0.1 s delivery, 1 s driver delay, v=13.889 m/s, a=3 m/s² and 10 m margin give about **140.8 m**. A queue or a three-specialist round-robin can make it much larger. The M1 report's roughly 2 s single-image forwards are desktop smoke observations, not a guaranteed 2 s service budget. Measure complete scheduled service/confirmation delay and route distance before selecting warning thresholds. A pretty 50 m warning animation cannot conceal seconds of processing.

Physical hazard ground truth may block a lane, place a stopped surrogate vehicle, or apply a declared surface-speed constraint in both experiment arms. Those effects are independently specified; SUMO does not model suspension damage from a decorative pothole. Global lane permissions or omniscient rerouting can leak hazard knowledge to baseline vehicles, so disclose that effect and isolate the warning policy from the physical-world mechanism. A falling tree mesh must have an approved traffic surrogate if it is claimed to obstruct traffic. Native car following already handles many stopped-vehicle encounters; warning benefit may appear as smoother braking, lower queues or earlier rerouting rather than a manufactured “100% collision avoidance.”

Proposed paired experiments use identical geometry, fleet, departure/route seeds, physical hazard schedule and signal program with warnings enabled/disabled. Store the scenario stream so rerouting-induced RNG divergence cannot silently change later hazards. Use separate deterministic RNG streams for traffic, hazards and communication. If comparison uses recorded genuine detections, label it prediction replay and keep original model/input/timing provenance; if live frames diverge, report that as a perception/traffic experiment rather than pretending the inputs were identical.

Measure: active-population minimum/mean/max; segment/lane speed and queue length; exposure and traversal of the affected lane; hard braking and actual acceleration; commanded versus observed merges; reroute completion; warning eligibility-to-delivery; receipt-to-response; duplicate/drop/expired counts; time to normal flow after clearance; collisions, emergency stops and teleports. Annotate collision semantics—SUMO's normal minGap violation registration differs from physical bumper overlap.

TripInfo normally records arrived vehicles; repeating fleets need unfinished-trip output and per-lap/controller measurements. Lane-change output records observed movements and gap information. SSM provides vehicle-conflict surrogates such as TTC, PET and DRAC with documented implementation limitations; it does not validate pothole damage physics or every lateral hazard. [TripInfo](https://eclipse.dev/sumo/docs/Simulation/Output/TripInfo.html), [lane-change output](https://eclipse.dev/sumo/docs/Simulation/Output/Lanechange.html), [SSM device](https://eclipse.dev/sumo/docs/Simulation/Output/SSM_Device.html).

## 11. Approval-dependent implementation gates

1. Approve 5 km unique centerline scope, original topology/visual direction, driving side and 21/13 station policy. If another topology is chosen, recalculate all lengths and placements.
2. Build in a new directory; preserve the existing corridor. Validate edge connections, large-vehicle turns, signal-controlled movements and source-to-renderer geometry agreement. Do not use the stale corridor generator.
3. Traffic-only gate: actual 50+ active fleet for a declared 20-minute main window, then a one-hour longevity run; zero unexplained removals/teleports, every TLS effective, every route connection valid, and declared frame/step/RAM budget met. These are future acceptance targets, not performed tests.
4. RSU-only application gate: station placement/count inventory, blind-spot documentation, duplicate/revision/expiry tests, relevance and route filtering, deterministic delay/loss and explicit delivery receipts. A mock communication probe is labeled mock and precedes any claim of genuine AI causality.
5. Genuine AI gate: approved observation sources, honest frame/model/timing identity, actual predictions and confirmation, correct scenario mapping and no scheduler-created positive detection. Existing M1 source ownership stays with its inference maintainer.
6. Paired traffic/communication experiments with fixed inputs or declared live-input variation; measure observed outcomes before safety claims.
7. Optional separately approved Veins/Artery radio study with a pinned small compatibility scenario and resource budget. No installation or billable service begins as part of this research.

## 12. Outstanding limitations

- All city geometry, traffic counts, signal timings, sensing views and RSU communication values above are proposals or mathematics. No SUMO route, capacity, 3D rendering or radio performance was measured in this task.
- Final SUMO lane lengths, internal junction links, pole offsets, turning pockets and exact rendering mesh geometry require a shared geometry audit after approval.
- The 21/13 count relies on mandatory units at all nine junctions and one physical unit serving both directions; a different installation policy changes the arithmetic.
- The sparse configuration has a 500 m maximum-spacing policy but an actual maximum gap of about 292.5 m in this topology. It should be labeled precisely in experiment results.
- Camera FOV and detector domain limitations can leave hazards unseen even within modeled communication coverage.
- No packet-level simulator was installed, and no PHY/MAC, real radio, ETSI conformance or collision-avoidance effectiveness was verified.
- Current primary documentation is version-sensitive; approved implementation must pin the SUMO executable, Python client, renderer and any network framework together and perform bounded compatibility checks.

The specialist's final recommendation is a compact, mathematically reproducible Garden-Ring traffic district with five working signal intersections, a controlled circulating fleet, explicit blind spots, and a modeled communication layer before packet-level network studies. It gives a clear implementation target while preserving the user's right to choose a different original city concept before construction.
