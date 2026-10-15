# PFPD Milestone M1 Implementation Report

**Date:** 2026-10-14 (Asia/Kolkata)  
**Scope:** M1 real image inference remediation and validation only. No SUMO integration, corridor XML changes, model weight changes, or M2/M3 work.

## Outcome

The M1 inference path now executes genuine ONNX Runtime CPU inference for a single image, authenticates each immutable specialist model by SHA-256 and graph metadata, decodes its YOLO output, performs float-coordinate NMS, maps boxes back through letterbox padding to the source image, and records the raw output, source image, timings, runtime, semantic model identity, and predictions. Real inference errors return nonzero status and structured error output; they never become mock detections. Temporal association requires explicit frame/source/model identity and capture time, uses one-to-one IoU matching, and resets confirmation after misses or excessive capture gaps.

The inference module and its focused test suite were rebuilt in the current checkout after an unexplained worktree disappearance. The earlier implementation evidence and initial checkout snapshot remain under `results/archive/m1_20261014_e6bcbbd4/`. This report's current measurements and fresh raw outputs refer to the current `src/edge_rpi_hazard_inference.py` implementation and `results/m1_20261014_recovered/`.

## Files changed

- `src/edge_rpi_hazard_inference.py` — real image CLI, explicit inference and mock modes, preprocessing, output validation, NMS, coordinate restoration, provenance, timing, safe evidence output, and nonzero failure handling.
- `src/pfpd_perception/__init__.py` — perception package marker.
- `src/pfpd_perception/model_registry.py` — reviewed model paths, hashes, names, source identity, and strict input/output/metadata contracts.
- `src/pfpd_perception/temporal.py` — source/model-scoped temporal association and explicit frame/capture-time validation.
- `tests/test_m1_inference.py` — 33 focused unit tests for image handling, output decoding, model identity, failure behavior, beacon labeling, and tracking.
- `docs/local_agent_research/M1_IMPLEMENTATION_REPORT.md` — this evidence-bound milestone report.
- `.agents/rules/memory.md` — milestone status and measured outcomes appended after verification.

The existing SUMO corridor, its XML, ONNX files, and historical result assets were not edited. The original untracked result artifacts match their recorded SHA-256 hashes in `results/archive/`.

## Model smoke results

Commands below were run from the repository root with the project virtual environment. Each successful run wrote its JSON record, annotated image, and raw tensor under `results/m1_20261014_recovered/`.

| Registered model | Input | Verified SHA-256 | Output | Confidence(s) | ONNX inference time |
|---|---|---|---|---|---:|
| `pothole` | `data/test_user_pothole.png` | `289c7d60b8d920ef1da65681629c8fc854ccaf7b01474afc5997ba89b1ca077d` | `[1,5,12096]` float32; 1 box | 0.889811 | 2238.439 ms |
| `construction` | `data/test_user_construction.png` | `4144423ff28f6fab297828be7965a8104600359fbd2ed58b0da45e16fa7788f8` | `[1,5,12096]` float32; 2 boxes | 0.721434, 0.463068 | 2030.734 ms |
| `accident_obstacle` | `data/test_user_accident.png` | `dc2985e915d44ff419d205bcae0df9fc1cfdf89960e794730b3afcb2134b96bf` | `[1,5,12096]` float32; 1 box | 0.886369 | 2011.490 ms |

All three graphs reported input `images` float32 `[1,3,768,768]` and output `output0` float32 `[1,5,12096]`, with the matching specialist class-0 metadata. The runtime was Python 3.11.16, ONNX Runtime 1.31.0, NumPy 2.4.6, OpenCV 5.0.0 on Windows x64, using `CPUExecutionProvider` and four intra-op threads. These are single-image desktop timings, not Raspberry Pi performance or calibrated latency statistics.

The pothole box visually covers the supplied pothole. Construction produced a box around the roadwork sign and another over a puddle-like region, which is a visible likely false positive. The accident model returned one broad box extending across most of the damaged-vehicle image. These are qualitative observations from unannotated sample images, not accuracy measurements. The model did not receive adjusted thresholds to increase detections.

## Commands and verification

Successful commands:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m src.edge_rpi_hazard_inference --help
.\.venv\Scripts\python.exe -B src/edge_rpi_hazard_inference.py --model quantization/pothole_yolo11s_int8.onnx --image data/test_user_pothole.png --output-dir results/m1_20261014_recovered
.\.venv\Scripts\python.exe -B src/edge_rpi_hazard_inference.py --model quantization/construction_yolo11s_int8.onnx --image data/test_user_construction.png --output-dir results/m1_20261014_recovered
.\.venv\Scripts\python.exe -B src/edge_rpi_hazard_inference.py --model quantization/accident_obstacle_yolo11s_int8.onnx --image data/test_user_accident.png --output-dir results/m1_20261014_recovered
```

The final unit command passed all 33 tests in 1.299 seconds. The three real-image commands each exited 0 and produced a finite raw tensor, record, and annotated image. Final rerun records use the `_final.json` suffix under `results/m1_20261014_recovered/`. The earlier M1 evidence directory also contains six genuine forwards (a supplied image per specialist and a second unannotated road-scene smoke per specialist), two one-pass benchmark probes, and their raw arrays and logs.

Failure probes and their expected nonzero outcomes:

```powershell
.\.venv\Scripts\python.exe -B src/edge_rpi_hazard_inference.py --image data/M1_ABSENT_INPUT.png
# exit 1; structured FileNotFoundError, no inference result and no mock fallback
.\.venv\Scripts\python.exe -B src/edge_rpi_hazard_inference.py --benchmark --output-dir results/forbidden
# exit 2; rejected option combination before creating an ONNX session
```

The earlier nested cross-directory launch was:

```powershell
& 'C:\Users\adars\OneDrive\Desktop\PFPD-ll\.venv\Scripts\python.exe' -B 'C:\Users\adars\OneDrive\Desktop\PFPD-ll\src\edge_rpi_hazard_inference.py' --image data/test_user_pothole.png --conf 0.35 --iou 0.45 --output-dir 'C:\Users\adars\OneDrive\Desktop\PFPD-ll\results\m1_20261014_e6bcbbd4'
# cwd: C:\Users\adars\OneDrive\Desktop\PFPD-ll\results\m1_20261014_e6bcbbd4; exit 4294967295, empty stdout/stderr
```

That attempt is recorded in `results/archive/m1_20261014_e6bcbbd4/verification_commands.json`. I then made relative image paths resolve from the project root and reran the command from `results/m1_20261014_recovered/`; it exited 0 and performed a real forward. Module help also exited 0. The first unit-test run is preserved in `unit_tests_first.log`: two temporary-directory permission errors and one outdated integer-box expectation failed. After the test fixture was corrected and temporary folders were placed under the writable test directory, the 32-test suite passed. A separate exploratory test that changed the process cwd to the Desktop failed with `PermissionError` under the sandbox and was removed; the explicit cross-directory CLI run passed instead.

## Coverage and limitations

The tests cover genuine model hash selection and identity, static graph contract checks, safe metadata parsing, preprocessing normalization and padding, malformed output rejection, class-0 semantics, confidence filtering, float NMS, coordinate inversion, image/model/runtime failures, output overwrite prevention, mock/real mode separation, and temporal streak reset, time gaps, expiration, one-to-one association, and model/class identity consistency.

The available input images are not accompanied by trusted labels. A representative clean negative set, annotated detection ground truth, and timestamped positive/negative temporal clips are absent. Therefore precision, recall, mAP, false-alarm rate, real temporal confirmation, and accuracy claims remain unverified or blocked. Formal ONNX checker validation, Raspberry Pi performance, radio transmission, ETSI conformance, and closed-loop SUMO behavior were not tested. M1 has not integrated its outputs into SUMO.

## Git and acceptance

Branch remains `main`; there was no commit or push. The initial runner was already modified and numerous user files were untracked. All 243 missing tracked paths observed during the work were restored individually from the index while preserving the runner path. The initial archived Git status is `results/archive/m1_20261014_e6bcbbd4/incoming_git_status.txt`; it records the checkout at task start. Pre-existing result files are archived with matching hashes. The approved M1 source change and the pre-existing untracked user assets remain visible in `git status`.

**Acceptance:** the M1 implementation, unit, three-model inference, and failure-handling criteria are satisfied for the available desktop environment. Dataset-based accuracy validation and real temporal-clip validation are blocked by missing labeled/negative and temporal assets. Work stopped at M1; no M2 or M3 work was started.
