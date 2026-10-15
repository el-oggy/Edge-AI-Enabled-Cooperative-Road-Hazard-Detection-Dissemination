"""Real ONNX image inference and explicitly separate mock stream demonstration.

The reviewed artifacts are single-class YOLO11s exports. This module authenticates
model contents before inference and does not substitute mock detections on errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid
from datetime import datetime, timezone

try:
    import cv2
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None
try:
    import numpy as np
except ImportError:  # pragma: no cover - environment dependent
    np = None
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - environment dependent
    ort = None

try:
    from .pfpd_perception.model_registry import (
        MODELS, project_path, resolve_hazard_info, sha256_file,
        validate_session_contract,
    )
    from .pfpd_perception.temporal import TemporalHazardTracker, box_iou
except ImportError:  # direct script execution: python src/edge_rpi_hazard_inference.py
    from pfpd_perception.model_registry import (
        MODELS, project_path, resolve_hazard_info, sha256_file,
        validate_session_contract,
    )
    from pfpd_perception.temporal import TemporalHazardTracker, box_iou


def _require_numpy():
    if np is None:
        raise RuntimeError("NumPy is required for ONNX inference")


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV is required to decode and preprocess images")


def _input_hw(input_size):
    if isinstance(input_size, int):
        height = width = input_size
    else:
        height, width = input_size
    if (isinstance(height, bool) or isinstance(width, bool)
            or not isinstance(height, int) or not isinstance(width, int)
            or height <= 0 or width <= 0):
        raise ValueError("input_size must be a positive integer or (height, width)")
    return height, width


def preprocess_image(image_path_or_array, input_size=640):
    """Decode BGR input and apply YOLO letterbox, RGB NCHW float32 normalization."""
    _require_numpy()
    _require_cv2()
    if isinstance(image_path_or_array, (str, os.PathLike, Path)):
        path = project_path(image_path_or_array)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            if not path.is_file():
                raise FileNotFoundError(f"Image not found or unreadable: {path}")
            raise ValueError(f"OpenCV could not decode image: {path}")
    else:
        image = image_path_or_array
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] <= 0 or image.shape[1] <= 0):
        raise ValueError("image must be a nonempty uint8 BGR array with 3 channels")
    height, width = image.shape[:2]
    input_h, input_w = _input_hw(input_size)
    scale = min(input_w / width, input_h / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = input_w - resized_w, input_h - resized_h
    left, top = int(round(dw / 2 - 0.1)), int(round(dh / 2 - 0.1))
    right, bottom = dw - left, dh - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, (height, width), (scale, left, top), image


def _prediction_rows(preds):
    _require_numpy()
    values = np.asarray(preds)
    if values.size == 0:
        if values.ndim == 3 and values.shape[0] == 1 and values.shape[1] == 5:
            return np.empty((0, 5), dtype=np.float64)
        raise ValueError(f"Unsupported empty output shape: {values.shape}")
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(f"Expected batch size one, got {values.shape}")
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D/3D YOLO output, got {values.shape}")
    if values.shape[0] == 5:
        values = values.T
    elif values.shape[1] != 5:
        raise ValueError(f"Expected 5 output fields [cx,cy,w,h,score], got {values.shape}")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ONNX output contains NaN or infinity")
    if np.any(values[:, 4] < 0) or np.any(values[:, 4] > 1):
        raise ValueError("YOLO confidence scores must be in [0, 1]")
    return values


def postprocess(preds, orig_shape, ratio_pad, hazard_info=None,
                conf_thresh=0.35, iou_thresh=0.45):
    """Decode [cx,cy,w,h,score] proposals, float NMS, and restore source pixels."""
    _require_numpy()
    if not isinstance(hazard_info, dict) or not hazard_info.get("source_model_id"):
        raise ValueError("Reviewed model semantics are required for postprocessing")
    if not math.isfinite(conf_thresh) or not 0 <= conf_thresh <= 1:
        raise ValueError("conf_thresh must be in [0, 1]")
    if not math.isfinite(iou_thresh) or not 0 <= iou_thresh <= 1:
        raise ValueError("iou_thresh must be in [0, 1]")
    if (len(orig_shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in orig_shape)
            or len(ratio_pad) != 3):
        raise ValueError("Invalid source image shape or letterbox transform")
    height, width = orig_shape
    scale, pad_x, pad_y = ratio_pad
    if not all(math.isfinite(v) for v in ratio_pad) or scale <= 0:
        raise ValueError("Invalid letterbox scale or padding")
    rows = _prediction_rows(preds)
    candidates = []
    for index, (cx, cy, box_w, box_h, score) in enumerate(rows):
        if score < conf_thresh or box_w <= 0 or box_h <= 0:
            continue
        x1, y1 = (cx - box_w / 2 - pad_x) / scale, (cy - box_h / 2 - pad_y) / scale
        x2, y2 = (cx + box_w / 2 - pad_x) / scale, (cy + box_h / 2 - pad_y) / scale
        x1, x2 = min(width, max(0.0, x1)), min(width, max(0.0, x2))
        y1, y2 = min(height, max(0.0, y1)), min(height, max(0.0, y2))
        if x2 > x1 and y2 > y1:
            candidates.append({"bbox_xyxy_float": [x1, y1, x2, y2], "confidence": float(score),
                               "proposal_index": index})
    candidates.sort(key=lambda row: (-row["confidence"], row["proposal_index"]))
    kept = []
    while candidates:
        current = candidates.pop(0)
        kept.append(current)
        candidates = [candidate for candidate in candidates
                      if box_iou(current["bbox_xyxy_float"], candidate["bbox_xyxy_float"]) <= iou_thresh]
    model_id = hazard_info["source_model_id"]
    if model_id not in MODELS or hazard_info.get("name") != MODELS[model_id]["name"]:
        raise ValueError("Unrecognized or inconsistent specialist model semantics")
    detections = []
    for candidate in kept:
        x1, y1, x2, y2 = candidate["bbox_xyxy_float"]
        detections.append({
            "class_id": 0, "raw_class_id": 0, "class_name": MODELS[model_id]["name"],
            "confidence": candidate["confidence"], "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "bbox_xyxy_float": candidate["bbox_xyxy_float"], "proposal_index": candidate["proposal_index"],
            "source_model_id": model_id, "status": "SINGLE_FRAME_UNCONFIRMED", "persistence_hits": None,
        })
    return detections


def create_session(model_path, hazard_info=None, intra_op_threads=4):
    if ort is None:
        raise RuntimeError("onnxruntime is required; no mock fallback is used")
    info = hazard_info or resolve_hazard_info(model_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(info["model_path"]), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    metadata = validate_session_contract(session, info)
    return session, metadata


def generate_etsi_denm_beacon(rsu_id, hazards, speed_limit_kmh=40):
    """Create a clearly nonstandard, nontransmitted advisory-shaped record."""
    entries = []
    for hazard in hazards:
        model = MODELS.get(hazard.get("source_model_id"))
        entries.append({
            "hazardType": hazard.get("class_name"), "trackID": hazard.get("track_id"),
            "causeCode": model["cause_code"] if model else None,
            "severity": "UNASSESSED", "confidence": float(hazard["confidence"]),
            "persistenceFrames": hazard.get("persistence_hits"), "boundingBox": hazard.get("box"),
            "validationStatus": hazard.get("status", "SINGLE_FRAME_UNCONFIRMED"),
            "advisoryAction": "UNASSESSED", "advisorySpeedKmh": None,
        })
    return {"standards_compliant": False, "transmitted": False,
            "header": {"stationID": rsu_id},
            "denm": {"situation": {"hazardCount": len(entries), "hazards": entries}}}


def _annotate(image, detections):
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = MODELS[det["source_model_id"]]["color"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{det['class_name']} {det['confidence']:.3f}",
                    (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


def _reserve_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting user or historical evidence.
    with path.open("xb"):
        pass
    return path


def run_image(image_path, model_path, conf=0.35, iou=0.45, intra_op_threads=4):
    _require_numpy()
    _require_cv2()
    info = resolve_hazard_info(model_path)
    t0 = time.perf_counter()
    session, metadata = create_session(info["model_path"], info, intra_op_threads)
    t1 = time.perf_counter()
    image_path = project_path(image_path)
    tensor, shape, transform, image = preprocess_image(image_path, (768, 768))
    t2 = time.perf_counter()
    raw = session.run(["output0"], {"images": tensor})[0]
    t3 = time.perf_counter()
    raw = np.asarray(raw)
    if raw.dtype != np.float32 or raw.shape != (1, 5, 12096):
        raise ValueError(f"Runtime output violated reviewed contract: {raw.dtype} {raw.shape}")
    detections = postprocess(raw, shape, transform, info, conf, iou)
    t4 = time.perf_counter()
    record = {
        "schema": "pfpd.image_inference.v1", "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "mode": "real_onnx",
        "inference_executed": True, "mock_fallback": False,
        "model": {"source_model_id": info["source_model_id"], "path": info["model_path"],
                  "sha256": info["sha256"], "metadata": metadata},
        "input": {"path": str(Path(image_path).resolve()), "sha256": sha256_file(image_path),
                  "shape": list(shape), "tensor_shape": list(tensor.shape),
                  "tensor_sha256": __import__("hashlib").sha256(tensor.tobytes()).hexdigest(),
                  "scale_padding": list(transform)},
        "thresholds": {"confidence": conf, "iou": iou, "calibrated": False},
        "raw_output": {"shape": list(raw.shape), "dtype": str(raw.dtype),
                       "sha256": __import__("hashlib").sha256(raw.tobytes()).hexdigest(),
                       "finite_values": int(np.isfinite(raw).sum()),
                       "max_score": float(raw[0, 4].max())},
        "detection_count": len(detections), "detections": detections,
        "beacon_advisory": generate_etsi_denm_beacon("UNASSIGNED_SINGLE_IMAGE", detections),
        "timings_ms": {"model_validation_and_load": (t1-t0)*1000,
                       "decode_preprocess": (t2-t1)*1000, "inference": (t3-t2)*1000,
                       "postprocess": (t4-t3)*1000, "total": (t4-t0)*1000},
        "runtime": {"platform": platform.platform(), "python": sys.version,
                    "onnxruntime": ort.__version__, "numpy": np.__version__,
                    "opencv": cv2.__version__, "providers": session.get_providers(),
                    "intra_op_threads": intra_op_threads},
        "limitations": ["Single image cannot establish temporal confirmation or accuracy.",
                        "Confidence threshold is provisional and not calibrated."],
    }
    record["_annotation"] = _annotate(image, detections)
    record["_raw_output_array"] = raw
    return record


def _save_evidence(record, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = record["run_id"]
    raw_path = _reserve_path(output_dir / f"{stem}_raw.npy")
    with raw_path.open("wb") as stream:
        np.save(stream, record["_raw_output_array"], allow_pickle=False)
    annotation_path = _reserve_path(output_dir / f"{stem}_annotated.jpg")
    cv2.imwrite(str(annotation_path), record["_annotation"])
    json_path = _reserve_path(output_dir / f"{stem}.json")
    clean = {key: value for key, value in record.items() if not key.startswith("_")}
    json_path.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return json_path


def _mock_stream(rsu_id):
    tracker = TemporalHazardTracker(min_hits=3)
    frames = []
    for frame_id, detections in enumerate([
        [{"class_id": 0, "raw_class_id": 0, "class_name": "road_accident_obstacle", "confidence": .94,
          "source_model_id": "accident_obstacle", "box": [200,250,450,420]}],
        [{"class_id": 0, "raw_class_id": 0, "class_name": "road_accident_obstacle", "confidence": .95,
          "source_model_id": "accident_obstacle", "box": [202,248,452,422]}],
        [{"class_id": 0, "raw_class_id": 0, "class_name": "road_accident_obstacle", "confidence": .96,
          "source_model_id": "accident_obstacle", "box": [201,249,451,421]}],
        [], []], 0):
        confirmed = tracker.update(detections, frame_id=frame_id, capture_time_s=frame_id * .1,
                                   source_id="synthetic-demo", source_model_id="accident_obstacle")
        frames.append({"frame_id": frame_id, "detections": detections, "confirmed": confirmed})
    return {"mode": "mock", "inference_executed": False, "frames": frames,
            "beacon": generate_etsi_denm_beacon(rsu_id, frames[2]["confirmed"])}


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Edge road-hazard ONNX image inference")
    parser.add_argument("--model", default=MODELS["pothole"]["path"])
    parser.add_argument("--image", help="Run genuine single-image ONNX inference")
    parser.add_argument("--conf", type=float, default=.35)
    parser.add_argument("--iou", type=float, default=.45)
    parser.add_argument("--rsu-id", default="RSU_ZONE_04")
    parser.add_argument("--json-out", help="Save a JSON record to a new path")
    parser.add_argument("--output-dir", help="Save unique raw output, annotation, and JSON evidence")
    parser.add_argument("--simulate-stream", action="store_true", help="Run an explicitly synthetic tracker demonstration")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark ONNX inference on an image or synthetic zero tensor")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1 or not math.isfinite(args.conf):
        parser.error("--conf must be in [0,1]")
    if not 0 <= args.iou <= 1 or not math.isfinite(args.iou):
        parser.error("--iou must be in [0,1]")
    if args.intra_op_threads < 1:
        parser.error("--intra-op-threads must be positive")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if args.simulate_stream and (args.image or args.benchmark or args.json_out or args.output_dir):
        parser.error("--simulate-stream cannot be combined with image, benchmark, or output options")
    if args.benchmark and (args.json_out or args.output_dir):
        parser.error("benchmark output artifacts are not supported")
    if args.output_dir and args.json_out:
        parser.error("choose either --output-dir or --json-out")
    if not args.image and not args.simulate_stream and not args.benchmark:
        parser.error("provide --image, --benchmark, or explicit --simulate-stream")
    try:
        if args.simulate_stream:
            print(json.dumps(_mock_stream(args.rsu_id), indent=2, allow_nan=False))
            return 0
        info = resolve_hazard_info(args.model)
        if args.json_out:
            target = Path(args.json_out)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite: {target}")
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.benchmark:
            _require_numpy()
            session, _ = create_session(info["model_path"], info, args.intra_op_threads)
            if args.image:
                tensor = preprocess_image(args.image, (768,768))[0]
                source_kind = "real_image_tensor"
            else:
                tensor = np.zeros((1,3,768,768), dtype=np.float32)
                source_kind = "synthetic_zero_tensor"
            for _ in range(args.warmup):
                session.run(["output0"], {"images": tensor})
            times = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                session.run(["output0"], {"images": tensor})
                times.append((time.perf_counter()-start)*1000)
            print(json.dumps({"mode":"real_onnx_benchmark", "model": info,
                              "input_kind":source_kind, "iterations":args.iterations,
                              "warmup":args.warmup, "latency_ms_p50":float(np.percentile(times,50)),
                              "latency_ms_p95":float(np.percentile(times,95)),
                              "inference_only_per_second":1000/float(np.mean(times)),
                              "providers":session.get_providers()}, indent=2, allow_nan=False))
            return 0
        record = run_image(args.image, info["model_path"], args.conf, args.iou, args.intra_op_threads)
        if args.output_dir:
            record["evidence_json"] = str(_save_evidence(record, args.output_dir))
        if args.json_out:
            target = _reserve_path(args.json_out)
            target.write_text(json.dumps({k:v for k,v in record.items() if not k.startswith("_")},
                                         indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({k:v for k,v in record.items() if not k.startswith("_")}, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc),
                 "inference_result": None, "mock_fallback": False}
        print(json.dumps(error, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
