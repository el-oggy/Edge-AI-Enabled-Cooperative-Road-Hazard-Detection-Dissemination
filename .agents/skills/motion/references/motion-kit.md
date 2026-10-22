# Motion as Code kit

Local source supplied by the user:

- Outer folder: `C:/Users/adars/Downloads/Motion_as_kit`
- Actual project root: `C:/Users/adars/Downloads/Motion_as_kit/Motion_as_kit`
- Guide: `C:/Users/adars/Downloads/Motion-as-Code-Workflow-Guide.pdf`

Recheck these paths. Read README.md, app/package.json, app/src/timeline.ts, app/src/engine/scene.ts, and app/scripts/render.ts before adapting the kit. Work in a new project copy; preserve source assets and LICENSE.pdoom-engine. Copy only useful source/fonts/assets, not node_modules or old renders. Install only required missing dependencies in an appropriate workspace/runtime.

The provided demo is a 92.6-second narrated 1920x1080/60 fps explainer, not a generic 20-second product template. A plate extends Scene and implements `render(f, out)`; `f.t` is global time, `f.lt` local time, and `f.p` scene progress. `f.seeked` and `reset()` matter for stateful scenes. Prefer pure-time motion. Scene modules are lazy-loaded; confirm engine.errors is empty so a missing scene does not silently become an apparently successful render.

Key files:

- `app/src/scenes/`: custom plates; `_vo.ts` has camera/plotter/karaoke helpers.
- `app/src/timeline.ts`: scene windows and voice phrase anchors.
- `data/lyrics.json`: aligned script lines/words.
- `data/audio.json`: duration and analysis features.
- `analysis/align_vo.py`: script, spoken-number/acronym mapping, forced alignment.
- `analysis/audio_vo.py`: loudness/onset analysis.
- `analysis/sfx_mix.py`: demo-specific cues, not a general soundtrack.
- `app/src/main.ts`: preview and export API; preview uses the original voice clock.
- `app/scripts/render.ts`: Bun/headless browser/FFmpeg renderer.

## Narrated adaptation

Use the user's audio or an explicitly requested voice provider. Keep `analysis/models/w2v2_base_960h_q.onnx` and `vocab.json` in the working copy: the alignment code requires them. Replace the voice track; edit SCRIPT and SPOKEN; run both alignment and audio analysis. When words change, update every scene lookup and timeline `cut(...)` anchor, including repeated-phrase occurrence selection. Check each anchor resolves uniquely, every window has positive duration, and the windows cover the intended film without unintended gaps. Verify audio/film duration agree. The guide's demo edits and approval language are examples, not requirements from this user.

## Music-only or silent adaptation

`--noaudio` suppresses muxed voice only. It does not change the preview clock, remove voice-derived envelopes/words, shorten the demo, or fix the SFX cues. Supply a duration/feature track appropriate to the new film, fixed timeline seconds or new music beats, and a non-voice preview clock. Replace lyric-dependent scene logic. Use fresh audio cues instead of the original hundreds of speech-linked cues. A direct browser implementation may be simpler than modifying these interconnected assumptions.

## Formats

The supplied engine and renderer hardcode 1920x1080 in multiple places: engine/gl.ts, app/index.html, main/export defaults, renderer dimensions/viewport/still clipping, and contact-sheet proportions. `--scale 2` means 4K landscape, not portrait. Portrait adaptation requires consistent dimensions plus re-layout of the plates. Prefer leaving engine internals intact; when the requested format needs engine changes, make minimal changes in the working copy and verify them.

## Verified command shapes from this kit

Run inside the working copy's app folder after resolving dependencies. On Windows use full executable paths where needed.

```text
bunx vite --host 127.0.0.1 --port 5275 --strictPort
bun scripts/render.ts stills --url http://127.0.0.1:5275 --t 12.5,40.2 --out ../out/stills
bun scripts/render.ts video --url http://127.0.0.1:5275 --samples 4 --fps 60 --out ../out/draft.mp4
bun scripts/render.ts video --url http://127.0.0.1:5275 --samples auto --min-samples 4 --max-samples 12 --shutter 0.5 --crf 17 --out ../out/final.mp4
```

Use a dedicated free preview port and the matching `--url`; confirm the displayed project is the working copy. The supplied renderer accepts any reachable default :5173 page, so an old server can otherwise supply the wrong film. Add `--noaudio` for a truly non-narrated export after fixing preview/timing. The render source honors BROWSER_CHANNEL; Edge can use `msedge` when Chrome is absent. It uses Bun APIs, so Node is not a drop-in replacement. Check actual encoder completion: the original renderer does not reliably make all scene/encoder failures fatal. Use a bounded smoke render before a full export.

Before rendering an adapted kit, assert that scene errors and nonzero FFmpeg exits abort the run. Keep initialization and encoder waits bounded with useful diagnostics, and inspect the preview's unique project title/identity before connecting the renderer.

Retain the kit's MIT attribution. Its supplied demo voice/SFX originated with ElevenLabs; asset presence alone does not establish reuse rights. Original synthesis or user-cleared audio avoids inheriting those demo assumptions.
