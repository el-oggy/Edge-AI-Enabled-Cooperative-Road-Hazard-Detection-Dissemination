# PFPD smart-city research and design package

Research began 14 October 2026; synthesis completed 15 October 2026. **Proposal for approval. No city, renderer, network, communication layer or integration was implemented.** Three specialist research agents were used; their written notes and the coordinator's read-only host/model audit are included.

Start with [the decision brief](07_DECISION_BRIEF.md) and open [the visual concept gallery](VISUAL_CONCEPTS.html), or view [all three boards together](visuals/concept_comparison.png). The gallery is a planning document containing schematic drawings; it is not a running simulation or a measured renderer demonstration. [Download the editable research archive](PFPD_SMART_CITY_RESEARCH_PACKAGE.zip); it contains this folder and read-only historical reference documents alongside it, without model weights or datasets.

| Requested deliverable | File |
|---|---|
| 1. Technology research and comparisons | [01_TECHNOLOGY_RESEARCH.md](01_TECHNOLOGY_RESEARCH.md) |
| 2. Three original visual and city proposals | [02_VISUAL_CONCEPTS.md](02_VISUAL_CONCEPTS.md), [gallery](VISUAL_CONCEPTS.html) |
| 3. Recommended system architecture | [03_RECOMMENDED_ARCHITECTURE.md](03_RECOMMENDED_ARCHITECTURE.md) |
| 4. Hardware, local/hybrid/cloud feasibility | [04_HARDWARE_DEPLOYMENT.md](04_HARDWARE_DEPLOYMENT.md) |
| 5. Staged implementation roadmap | [05_IMPLEMENTATION_ROADMAP.md](05_IMPLEMENTATION_ROADMAP.md) |
| 6. Individual specialist and coordinator prompts | [06_PROMPT_COLLECTION.md](06_PROMPT_COLLECTION.md), [prompts/](prompts/) |
| 7. Final decision brief and approval choices | [07_DECISION_BRIEF.md](07_DECISION_BRIEF.md) |

Supporting evidence: [host/project audit](HOST_PROJECT_AUDIT.json), [rendering research](SPECIALIST_RENDERING.md), [network/traffic/RSU research](SPECIALIST_NETWORK_RSU.md), [AI and integration research](SPECIALIST_AI_INTEGRATION.md). The M1 report in the parent directory remains the implementation evidence for inference; this package does not upgrade its smoke results into accuracy validation.

Evidence terminology: **inspected local** means a current file or metadata inspection; **recorded runtime** means an existing run record, not a new benchmark; **published** means a cited external source; **proposed** means an engineering choice or target awaiting approval; **unverified/blocked** marks missing tests or assets. No linked video was watched in this assignment. Public video descriptions and inspectable repositories are identified separately.

Recommended direction: a compact **Garden-Ring Civic District**, 5,000 m of unique physical road centerline, five functional signalized four-way junctions, 50+ circulating vehicles, 21/13 RSU placement alternatives, and SUMO-controlled 3D views. First approve a small renderer-and-perception vertical slice; approve full city expansion only after visual, sensor and resource gates pass.
