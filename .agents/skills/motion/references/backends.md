# Backend selection

The user clarified on October 8, 2026 that the desired workflow is direct animation code, similar to the supplied Claude example. Brag supplies useful repository context and creative direction. Hyperframes is optional. The skill itself must work without a video-generation service.

| Route | Good fit | Confirm before using |
|---|---|---|
| Motion as Code kit | Procedural 2D/3D, plotter type, WebGL effects, voice-aligned explainers | Kit location, dependencies, browser, FFmpeg, output shape, new audio data and anchors |
| Direct browser code | Product/UI ads, kinetic type, schematics, arbitrary formats | Fonts/assets, pure-time API, browser runtime, renderer and encoder |
| Brag + Hyperframes | Repository-specific short launch videos when installed | Current Brag skill, required Hyperframes domain skills, actual CLI/tools |
| Remotion | Existing React compositions, nested sequences and component reuse | Local installation, its current APIs, renderer and composition dimensions |

Discover installed skills/tools before claiming a plugin is available. For Brag's full workflow, read its own SKILL.md and the relevant inspection/planning/composition/delivery references. Its normal workflow uses Hyperframes, but borrowing its planning advice does not require a Hyperframes handoff. Respect explicit user choices. Do not install another provider simply to reproduce the name used in a reference video.

In the creation session, Brag 0.4.0 was installed; no Hyperframes domain skills or callable HyperMotion tool were found. This is a dated observation, not a permanent constraint. Recheck availability on future invocations. If a missing provider is essential to an explicit request, state the concrete missing dependency while completing independent work.

For direct renders, wait for assets/fonts, pass exact frame times `i/fps`, stream frames to the encoder with backpressure, and check the encoder's exit code. A browser preview is not an exported video. An MP4 filename or renderer log alone is not verification.
