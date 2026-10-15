# 5 — Approval-gated roadmap

Date: 15 October 2026. New phase IDs **SC0–SC7** avoid confusing this new design with the existing M1/M2/M3 corridor plan. M1 source remains the established inference baseline. All phases below are proposed; generating prompts does not authorize them.

## Sequence and gates

| Phase | Work and owners | Dependencies | Reviewable outcome and gate |
|---|---|---|---|
| SC0 — preservation and contracts | Coordinator + integration + perception | Approval of the first slice | Read-only inventory/hashes, pinned proposed stack, clock/coordinate/frame/event contracts; no model rewrite |
| SC1 — renderer and perception vertical slice | Rendering, integration, perception | SC0; small licensed assets and labels | One realistic junction/approach, visible working signals, actual clean fixed-camera frames, immutable ONNX forwards and honest misses/false positives; visual and sensor acceptance |
| SC2 — connected metric road network | Network/traffic; renderer consumes geometry | Approved concept/scope; SC1 renderer gate | Independently named 5 km network, five valid TLS, loops/alternate paths, export length audit, native safety and route correctness |
| SC3 — sustained traffic and city art | Network/traffic + renderer | SC2 geometry/TLS mapping | 50+ simultaneous actors after warmup, continuous circulation and full scene; 20-minute initial soak, later ≥60-minute robustness run |
| SC4 — first causal warning | Perception, scenarios, RSU, integration | SC1 sensor gate, SC2 valid lanes; contracts frozen | Physical scenario → clean pixels → real prediction evidence → delayed event → delivered warning → safe action → observed outcome; no-warning/runtime-failure/stale controls |
| SC5 — multi-hazard/RSU comparison | Scenarios + RSU + evaluation | SC4 passes | 21/13 deployment manifests, two/three sensing stations, overlap, duplicates, loss/delay/TTL/clearance and recovery; per-event provenance |
| SC6 — scientific experiments | Evaluation + integration | Frozen network/models/assets and evidence ledger | Paired warnings off/on, RSU density and communication assumptions, replicated seeds, valid denominators/uncertainty; no invented benefit |
| SC7 — presentation / optional expansion | Renderer + coordinator | SC5 functional and SC6 evidence | Overview, chase, fixed sensor and causal event views; optional later GPU/radio/interactive ego work separately approved |

SC2 detailed geometry planning, source/license selection and contract drafts can proceed independently of further M1 fixture validation after approval, but warnings cannot claim validated perception until SC1/SC4 gates pass. No specialist may quietly extend M1 or the legacy corridor because it is convenient.

## First approved phase must be bounded

Recommend approving **SC0–SC1 only**: one junction/road slice with a fixed camera, functional SUMO-sourced signals and about ten actors, then a 50-actor resource probe. Reuse the current models exactly. Test representative pothole/construction/damaged-vehicle assets and negatives; tree/debris/normal breakdown recognition gets its own capability result. Include one real-footage forward as a source-identity/adapter control. No full city construction until the visual and sensor trial is reviewed.

If rendered positives fail at the existing thresholds, retain the miss and revise licensed asset/camera conditions through reviewed tests. Do not lower thresholds or duplicate stills to pass. Present real-footage mode as an explicit interim option, not a claim that the 3D model is what the detector saw. If local resource limits dominate, propose a capped remote or alternate-renderer trial with measured evidence.

## Proposed acceptance and experiments

- **Geometry:** intended connectivity, correct left-hand traffic convention if adopted, physical centerline 5,000 m ±1%, signal link mapping, no unsupported turn routes, topology permits useful alternative paths. Recalculate RSU count/positions from final geometry rather than trusting the concept formula.
- **Signals:** actual phase controls traffic, matching render heads and junction/link IDs; native right-of-way/safety retained, no bypass of red lights to manufacture traffic fluidity.
- **Circulation:** after declared warmup, every authoritative sample in a 20-minute main demonstration has at least 50 active vehicles. Record minimum/maximum/percentiles, IDs, route loops, departure/arrival/reinsertion, queues and blocked laps. A count plotted only at one moment does not pass. Initial suggested fleet is 72 with adjustments based on validation, not a measured capacity.
- **Perception:** all three genuine forwards during the online AI demonstration, model/input/output hashes, distinct source semantics, licensed labeled positives and clean/ambiguous/occluded negatives. Freeze operational thresholds and report all outcomes. Ground-truth-to-warning injection is prohibited.
- **Timing:** separate capture, readback, queue, forward, confirmation and delivery times; bounded queues and explicit drops, preserve capture-gap rules. Any simulation time dilation is visible. Warning age/lead-distance constraints are scenario-specific and may fail.
- **Response:** log commands separately from observed lane/speed/route changes; preserve native safety bits. An unsafe lane-change request rejected by SUMO is recorded, not hidden.
- **Resource/visual:** proposed 1280×720 display, 30 FPS target, minimum 20 sustained display FPS with p95 presentation interval ≤50 ms, real-time factor ≥0.9 over declared test duration; budget sensor targets separately and log rendering duration separately. Actual footage, graphics and memory need profiling, not estimates called results.
- **Communications:** seeded application delay/loss assumptions, duplicate/revision/TTL tests, relevant route recipients, reconnect/stale-frame behavior. No radio/ETSI claim without separate fidelity work.
- **Evaluation:** first five paired seeds as a pilot, then choose replication count from variability/desired precision. Report all seeds, distributions and exclusions. Compare travel time/waiting/queues, valid TTC/PET/DRAC, late warnings and observed responses; zero collisions is not a proven improvement.

## Agent work order and ownership

Coordinator freezes common contracts before parallel coding and assigns isolated worktrees only after user approval. Network agent owns new network/config/routes; renderer owns scene/assets/client; perception owns new adapters, fixtures and model scheduling; scenarios owns physical/visual hazard contracts and independent truth; RSU owns application transport/event rules; integration/evaluation owns sole TraCI stepping, logs and experiments. Each file has one writer. Changes crossing ownership wait for an interface review and named integrator.

Parallel after SC0: licensed art/scene shell, network drafting, perception fixtures/adapters, scenario manifest and RSU pure logic. Dependent: road meshes wait for approved network schema; signal render mapping waits for network link data; causal scenario waits for sensor, event and controller contracts; multi-camera/50-actor tests wait for the single-slice gate. Coordinator merges sequentially and runs integrated checks once the pieces exist.

## Main risks and mitigations

| Risk | Evidence / mitigation |
|---|---|
| Synthetic domain gap | Current model label names and smoke positives do not establish rendered recognition; SC1 gate and genuine-footage interim mode |
| CPU inference delay and temporal gap | Capture bursts, bounded queues, explicit service clocks; don't claim zero delay or 50 cameras |
| Host memory pressure | Current 7.38 GiB RAM with only about 1.30 GiB available snapshot; measured first slice before editor/scene expansion |
| Road/render misalignment | Load exported lane/junction/TLS geometry; test coordinate/reference-point conversions and sensor capture ticks |
| Claimed radio fidelity | Separate assumed application model from later Veins/Artery or physical test |
| Hidden route failures/teleports | Validate connectivity/routes and log these outcomes; disable masking policies in new experiments through reviewed configuration |
| Unsafe forced response | Preserve SUMO safety, explicit arbitration, route validity and observed command effects |
| Unclear hazard recovery | TTL is not clearance; manual/reviewed clearance initially, visibility-aware negative evidence later |
| Legacy-source interference | New directories, hashes and one writer; no stale generator or M1 edits without ownership/approval |
| Engine dependency/asset licensing | Pin selected versions, keep license/provenance ledger, avoid opaque bridge binary as scientific authority |

## Approval decisions

Review the concept, 5 km centerline interpretation, proposed browser-first renderer and SC0–SC1 pilot scope. Also approve the initial communications fidelity (assumed application model), fixed sensing stations and online mode priority. Separate later approval covers full 5 km generation, major engine/provider installation, remote resources with a cost cap, retraining/model alterations, detailed radio simulation and publishing/Git pushes. Approval for one phase does not automatically authorize the next.
