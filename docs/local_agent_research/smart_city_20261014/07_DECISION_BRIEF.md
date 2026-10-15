# 7 — Decision brief

Date: 15 October 2026. This is a proposed direction for review; the city has not been built.

## What to build and what it would look like

Build a **Garden-Ring Civic District**: about 5 km of unique connected street centerline within a compact landscaped district, two curved boulevards, four radial corridors, a civic square, five functional signalized four-way junctions and four outer T junctions. Vehicles circulate on connected loops and have alternate paths. A daylight material style supplies asphalt wear, curbs, trees, low-rise buildings, visible signals and recognizable RSU masts. Review the [three concepts and sketches](02_VISUAL_CONCEPTS.md) or [open the gallery](VISUAL_CONCEPTS.html); Canal-and-Campus and Foundry-and-Market are distinct alternatives.

Use a proper 3D browser scene as the first feasibility candidate, with headless SUMO controlling traffic and signals. Three.js/WebGL2 supports real camera render targets and instanced scene geometry; Unity URP is the strongest visual-production alternate if the reviewed slice needs it. Godot is a viable lightweight native alternative. Unreal/CARLA require substantially stronger development resources and remain later workstation/cloud options. These are researched engineering choices, not measured local capabilities. [Technology comparison](01_TECHNOLOGY_RESEARCH.md).

## How the AI genuinely participates

Actual clean camera pixels go to the three immutable specialist ONNX models, then source-specific temporal evidence and reviewed location mapping create an eligible event after service delay. RSUs disseminate it through an explicitly modeled channel; the sole TraCI controller requests safe behavior and records what vehicles actually do. The physical scenario scheduler never invents neural predictions.

Synthetic recognition is an open test gate. The current models have genuine runtime evidence on a few supplied images, but no validated rendered-camera dataset, clean-negative set or timestamped clip set. Construction produced a probable false positive and accident boxes were broad. Fallen-tree, normal-breakdown and debris recognition are not demonstrated. If rendered assets fail, genuine real-footage inference with explicit time/location mapping remains an honest interim mode; the visual scene must be labeled as a corresponding scenario rather than the model's input. Online real inference, genuine evidence replay and mock communications have visibly separate mode labels.

## Traffic, RSUs and dynamic scenarios

Target a 72-vehicle fleet initially to maintain **at least 50 active** after warmup; this is a proposal, not capacity evidence. Mix cars, motorcycles, custom auto-rickshaw-shaped vehicles, buses and trucks on class-valid closed routes, active TLS programs and native safety checks. Verify the count throughout the main session and log blocked circulation/queues.

For the nominal network, mandatory junction units plus arclength subdivision yield **21 RSUs at a 200 m maximum spacing** and **13 at a 500 m maximum spacing**. Five are traffic-signal associated. Only two or three initially sense; others relay or serve intersection roles. Unit spacing is not radio range or camera visibility. Actual placement is recalculated after final exported geometry.

Seeded physical hazards appear/clear independently of neural detections. A tree can fall after Vehicle A passes; Vehicle B is warned only if a healthy sensor sees it and the appropriate model actually recognizes it. No observation means no AI warning. Repeated/overlapping events preserve event IDs, revision, source evidence, expiry and recovery. Lane changes occur only where legal and safe; one-lane segments may require stop or diversion.

## Can it run locally; what does cloud change?

Current inspection confirms Ryzen 5 5500U, 7.38 GiB physical RAM, integrated Radeon graphics and limited free-memory headroom. Recorded M1 inference is seconds-scale: approximately 6.28 s arithmetic for a serial three-model sweep, about 18.84 s for three frames through every model before overhead. More graphics hardware does not automatically accelerate these particular INT8 graphs. First prove a limited renderer/readback/inference slice.

Cloud or another workstation can supply richer rendering, more cameras and provider trials. Tight experiments should co-locate renderer, inference and SUMO; the laptop can receive video and telemetry. Public illustrative GPU rates start around a few dollars for a ten-hour compute-only trial in the cited examples, with storage/plan/network/tax costs separate and no availability/eligibility assumed. No cloud resources were started. [Hardware and deployment details](04_HARDWARE_DEPLOYMENT.md).

## First phase after approval

Approve **SC0–SC1 only**: preserve/hashes and contracts, then one realistic signalized road/junction slice with a fixed clean camera, small licensed hazard/negative fixtures, genuine model forwards and a measured ten-to-fifty-actor resource probe. Review visuals, timing and synthetic recognition before building all 5 km. Proposed targets include 30 display FPS desired, minimum 20 sustained FPS with p95 presentation interval ≤50 ms, a declared real-time factor target, bounded queues and truthful late/missed warnings. These are prospective gates, not achieved performance.

## Decisions for approval

1. Adopt 5 km as **unique physical road centerline**, rather than a 5 km radius or repeated driving route.
2. Select Garden-Ring Civic District, Canal-and-Campus, or Foundry-and-Market visual direction. Recommended: Garden-Ring for first local scope.
3. Approve the Three.js/WebGL2 first slice, with Unity/Godot comparison only if its reviewed visual/resource gate fails.
4. Approve fixed sensing RSUs and the initial application-level communication model; detailed RF networking remains separately reviewed.
5. Approve SC0–SC1 pilot work only. Full city expansion, engine installations, remote spending, model changes and publishing require their later gates.

[Roadmap](05_IMPLEMENTATION_ROADMAP.md) and [implementation prompt collection](06_PROMPT_COLLECTION.md) make those next tasks concrete. The prompts are review deliverables and have not been executed.
