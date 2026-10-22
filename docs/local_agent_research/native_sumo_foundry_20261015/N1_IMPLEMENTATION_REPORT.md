# N1 — Native Foundry and Market implementation

15 October 2026, Asia/Calcutta. Branch: `codex/native-sumo-foundry-n1`.
Starting and final HEAD: `47b0eddf1057c4b3f0ccd4dea1faed941c25a716`.

## Outcome and acceptance

**N1 appearance, native launch and traffic-baseline acceptance is satisfied.**
The root launcher opens real SUMO-GUI with this map. The final native captures
show readable district signs, distinct vehicles, functioning red/green signal
bars, a comparison legend, and unobscured individual physical sample artwork.
A 20-minute simulated run retained 72 active vehicles after warm-up with no
reported collisions, teleports, emergency stops or emergency braking.

This is explicitly `native_visual_demo`, with `ai_inference: false`, no
cooperative-warning service and no physical lane obstructions. The samples
are off-road authoring fixtures. Their labels do not establish model detections.
N2 scenario behavior and N3 inference/warnings were not implemented.

## Authorization, branch and cleanup

The latest user instruction authorized creating a branch, deleting the unwanted
uncommitted Antigravity prototypes, keeping research Markdown, and beginning N1.
The branch was created before application edits. No commit, staging, push, PR,
merge, reset, clean or stash was performed.

Before deletion, each target was verified to contain no tracked files. A ZIP
was CRC-checked and every file checked by SHA-256. The following working-tree
directories were then removed:

- `simulation/smart_city`
- `src/pfpd_city_perception`
- `src/pfpd_city_protocol`
- `src/pfpd_city_runtime`
- `tests/smart_city`
- `visualization`

Recovery: `results/n1_20261015/retired_prototypes.zip`, 91 files,
17,541,174 uncompressed bytes; manifest: `cleanup_manifest.json` in that folder.
The archive is a local ignored ZIP, not a Git commit. Keep it if recovery is
needed. Installed skills, research documents and the existing `.gitignore`
modification were preserved.

The 170-file research baseline was reconciled: 144 files still match in the
workspace, 26 retired files match in the archive, and zero are missing or
mismatched. `preservation_check.json` records that result. The legacy corridor,
M1 source, model artifacts and historical project memory were preserved.

## Files created and responsibilities

| Files | Purpose |
|---|---|
| Root `Run SUMO Simulation.vbs` and `.cmd` | Native one-click entry, existing `.venv`, independent of caller directory; VBS avoids a console |
| `simulation/foundry_market/run_simulation.py` | One TraCI owner, portable paths, GUI default, pacing, capture, input hashes, traffic/signals/error evidence |
| `foundry.nod.xml`, `foundry.edg.xml`, `foundry.net.xml` | Reviewed prototype geometry copied without modification or regeneration |
| `foundry.rou.xml` | 72 actors, five visual types, connected class-valid repeated circuits |
| `foundry.sumocfg`, `foundry.view.xml`, `follow.view.xml`, `inspection.view.xml` | Explicit scene bindings and native view schemes |
| `scenery.py`, `foundry.scenery.add.xml`, `scene.manifest.json`, `assets/*.png` | Explicit authoring tool, placements, palette, local artwork and dimensions |
| `simulation/foundry_market/README.md` | Launch instructions, commands, controls and evidence limits |
| `tests/test_native_foundry.py` | Nine focused N1 regressions; existing 33 M1 tests remain intact |
| This report and research `README.md` update | Current N1 status alongside the historical research snapshot |
| `results/n1_20261015/` and `results/foundry_runs/` | Recovery archive, actual runs, failure records, screenshots and validation evidence |

The native network contains approximately **5000.0065 m of unique input
centerlines**, twelve two-way streets, seven junction nodes and three TLS
controllers. Five kilometres describes total street length; the compiled
network bounding box is approximately 917 by 768 m. The network SHA-256 is
`a495977a1a106a28533f83bb8c37b5113d22b48ff516875f9d5f20fb2139a015`.

Scenery includes 57 roofs, 19 olive trees, two storage tanks, three chimney
footprints, seven illustrative RSU masts and 18 transparent PNG assets. There
are separate pothole, puddle, manhole, asphalt-patch, fallen-tree, debris,
incident-car and construction samples. Puddle/manhole/repair are ordinary
comparisons. The legend enlarges their art for readability; physical POIs have
their stated metre dimensions. These images do not alter SUMO physics.

Vehicle inventory actually observed: 40 cars, 12 motorcycles, eight small
passenger vehicles, six buses and six trucks. The small passenger vehicle is a
custom visual with passenger behavior, not a calibrated autorickshaw model.
Routes contain finite repeated circuits; unrestricted circulation and demand
calibration belong to later work.

The runtime never invokes an authoring generator. Each run creates a new
output directory, refuses existing directories, and verifies its authored
input hashes before and after execution. Native screenshots use a single
window, with explicit camera settling and GUI event-loop yields in fast mode.
Missing or black screenshots fail capture validation. Normal window closing
is recorded separately from a failure after checking the actual SUMO exit.

## Actual delegation

- **gpt-6-luna / high**, `n1_native_scenery`: bounded artwork, scenery and
  manifest authoring, including final sign-clearance corrections.
- **gpt-6-luna / medium**, `n1_visual_verifier`: read-only review of actual
  native screenshots, independent of the authoring agent.
- Principal agent: preservation and branch, runtime, view settings, routes,
  launchers, tests, debugging and evidence reconciliation.

There were no simultaneous writers to the same paths. No new paid service,
dataset, external renderer or dependency installation was needed. Ponytail
guided reuse and bounded work. Gravitas scope/evidence principles were used;
no installed Gravitas hooks or formal product validators are claimed.

## Successful verification commands and results

Commands below ran from the repository root unless otherwise stated. Use a new
`--output` directory to repeat them; overwriting evidence is intentionally refused.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B simulation/foundry_market/run_simulation.py --headless --fast --duration 1200 --output results/n1_20261015/headless_validated
.\.venv\Scripts\python.exe -B simulation/foundry_market/run_simulation.py --fast --duration 1200 --capture --output results/n1_20261015/native_gui_validated
.\.venv\Scripts\python.exe -B simulation/foundry_market/run_simulation.py --fast --duration 180 --capture --output results/n1_20261015/native_gui_delivery
.\.venv\Scripts\python.exe -B results/n1_20261015/verify_click_launcher.py
```

| Check | Actual result |
|---|---|
| Final unit suite | 42 tests pass, including 33 existing M1 tests and nine N1 tests; exact duration in `unit-tests-final.txt` |
| Headless 1200 s | Exit 0; 25.369661 s controller wall time; all 72 inserted/running; 0 reported collisions/teleports/emergency stops/emergency braking |
| GUI 1200 s | Exit 0; 68.387476 s controller wall time; 72 active at every 1 s sample from 180 through 1200 s; at least 35 moving faster than 0.5 m/s at those samples |
| Final delivery GUI 180 s | Exit 0; 17.800612 s wall time; final scene/camera fix; 12 nonblack native PNGs; 72 active and 69 moving at 180 s |
| Signals | All three controllers changed phase; native close-up visibly shows red/green lane-state bars |
| VBS launcher | `wscript.exe` invoked from `C:\Users\adars\OneDrive`; visible, nonminimized SUMO window; normal close exit 0; `window_closed` summary; authored inputs unchanged |
| Portable CMD | Invoked from `C:\Users\adars` with absolute launcher path, `--headless --fast --duration 5 --output <absolute results/n1_20261015/launcher_external_cwd>`; exit 0 |
| Preservation | All 170 baseline hashes reconcile; all 91 retired files verified in ZIP; retired directories absent |

The 20-minute GUI run preceded the final sign relocation and camera-settling
change. Those presentation changes were checked by the final 180-second
delivery capture and unit suite; they did not change routes or traffic logic.
Use **`native_gui_delivery/`** for final visual evidence. Older capture folders
are historical debugging artifacts and some have black frames, obscured samples
or mismatched camera captions.

Environment: existing `.venv`, Python 3.11.16 64-bit Windows, Pillow 12.3.0,
installed SUMO 1.27.1 / TraCI API 22. The observed wall times include startup,
TraCI, logging and GUI scheduling as applicable. They are not detector latency,
render FPS, Raspberry Pi results, or a controlled hardware benchmark. SUMO's
completed-trip average is zero because all repeated routes were unfinished;
the per-second moving counts establish that vehicles were moving.

Unit coverage includes absent scene/artwork, missing vehicle art, no implicit
regeneration, absolute launcher paths, invalid durations, required visual
categories, image decodability/transparency, sign/court clearance, all declared
route connections including circuit closure, and the 72 departures.

## Expected failures and resolved unsuccessful checks

`failure_cases.json` preserves complete argument arrays, environment overrides,
return codes and stderr for these final checks:

```powershell
.\.venv\Scripts\python.exe -B simulation/foundry_market/run_simulation.py --headless --capture --duration 2
.\.venv\Scripts\python.exe -B simulation/foundry_market/run_simulation.py --headless --duration nan
.\.venv\Scripts\python.exe -B simulation/foundry_market/run_simulation.py --headless --fast --duration 2 --output results/n1_20261015/headless_1200
# In an isolated subprocess with SUMO_HOME pointing to a nonexistent folder
# and PATH empty (the parent environment was not changed):
.\.venv\Scripts\python.exe -B simulation/foundry_market/run_simulation.py --headless --duration 2
```

These returned **2, 2, 1 and 1**, respectively: incompatible capture flags,
nonfinite duration, existing output and missing SUMO. There was no fallback to
mock results, headless mode or regenerated assets.

Earlier unsuccessful work is retained rather than presented as a clean first run:

- Default sandbox branch creation could not write Git refs; approved escalation
  created the branch. An inaccessible temporary ZIP caused a copy failure before
  any deletion; a permanent workspace archive was rebuilt and reverified.
- The first test suite had one temporary-directory permission error. N1 tests
  now use disposable fixtures under the workspace and clean them up.
- `native_gui_first` failed because dynamically added views were not available.
  `native_gui_views` stalled and was interrupted. Subsequent experiments in
  `native_gui_revised`, `native_gui_final`, `native_gui_raster`,
  `native_gui_box_isolation`, `gui_native_canvas`, `native_gui_single_view` and
  `native_gui_single_early` stalled, timed out or missed captures. Only this
  task's identified SUMO processes were stopped. There is no proven underlying
  C++ renderer defect: failures also occurred with simple vehicle drawing.
- `native_gui_event_loop` completed after using one view and yielding for GUI
  startup/repainting. Later full inspection still found a camera lag and an
  enlarged Market sign covering physical samples. The final delivery adds
  four settling steps per camera, moves the sign outside the court, and was
  checked against the actual requested subjects.
- The first VBS check (`click_launcher_check.json`) did not establish a visible
  window and normal closure. The launcher now requests normal display with
  `pythonw`, and the controller resolves the close/exit race. The corrected
  `click_launcher_check_20261015_201323.json` passes. This test closed only its
  own uniquely identified SUMO window.

## Limits, remaining work and Git state

The sparse desert layout is intentional; appearance remains subject to user
review. Native SUMO is a top-down traffic view. No photorealism, third-person
renderer or new engine is supplied. Metre-scale samples require close-up views.
Saved PNGs are actual SUMO captures, not image-generated depictions.

Full one-hour wall-clock operation, arbitrary user interaction, other operating
systems and other SUMO/GPU versions were not tested. Native calls have a
30-second socket timeout. Ordinary traffic is not marked as hazardous merely
for stopping at signals. Hazard activation/clearance, physical blocking,
real-footage mapping, ONNX-to-TraCI causality, warning delivery, radio behavior,
RSU spacing/coverage and calibrated comparisons remain later work. No new
accuracy, neural latency or safety-improvement claims follow from N1.

The final working branch has new untracked N1 files and results. The only
tracked modification remains the pre-existing two-line `.gitignore` change;
the staged diff is empty and HEAD is unchanged. Internal Markdown remains
subject to the existing ignore policy. No commit or push was made. Work stops
at N1.
