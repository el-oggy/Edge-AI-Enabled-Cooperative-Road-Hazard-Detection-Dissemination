# 6 — Implementation-ready specialist prompts

Date: 15 October 2026. **Review deliverables only. None of these prompts has been executed.** An agent receiving one must first identify the exact phase and permissions explicitly approved by the user; a prompt file alone grants no implementation authority.

| Prompt | Responsibility and owned files | Starts after |
|---|---|---|
| [Coordinator](prompts/COORDINATOR.md) | Phase scope, contracts, integration order, evidence and review | Explicit SC0–SC1 approval |
| [Network and traffic](prompts/NETWORK_TRAFFIC.md) | New network/routes/TLS/traffic types; network config/tests | Concept/scope approval; SC0 schema |
| [3D scene and sensors](prompts/RENDERER_SENSORS.md) | New Three.js client, licensed art and clean camera capture | Renderer-slice approval; snapshot contract |
| [Perception](prompts/PERCEPTION.md) | New adapters/fixtures/scheduling around immutable M1 code | Perception-slice approval; sensor/frame contract |
| [Hazards and ground truth](prompts/HAZARD_SCENARIOS.md) | Seeded physical scenarios and independent truth | Scenario-slice approval; geometry/hazard contract |
| [RSU application network](prompts/RSU_NETWORK.md) | Placement manifests, event dissemination/dedup/expiry | Event/geometry contracts and specific phase approval |
| [Integration and evaluation](prompts/INTEGRATION_EVALUATION.md) | Sole TraCI runtime, common protocol, records and experiments | SC0 contract task; live scenarios wait for upstream gates |

Common proposed code roots are assigned rather than created in this research: `simulation/smart_city/`, `visualization/pfpd_city/`, `src/pfpd_city_*`, `configs/smart_city/<owner>/`, and `tests/smart_city/<owner>/`. The coordinator may refine names before implementation, then freeze one-writer ownership. Keep the existing `simulation/corridor_1km/`, M1 source/tests, ONNX weights and user changes intact.

Parallel after contracts: network drafting, scene shell/art, licensed perception fixtures, scenario manifests and RSU pure logic. Dependent: geometry-linked road meshes/TLS heads wait for network export; physical scenarios wait for geometry; warnings wait for genuine sensor evidence; multistation experiments wait for a valid causal slice. Integration is sequential at shared interfaces. Worktrees may be created only after an approved implementation plan assigns them, and are not created by this research package.
