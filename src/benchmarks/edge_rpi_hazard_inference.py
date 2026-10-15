"""Synchronous, authenticated YOLO11s ONNX image inference and temporal test utility.

Desktop/Pi performance and detection quality require separate validation. The
explicit mock stream exercises the tracker only; it never substitutes for ONNX.
The compatibility beacon is DENM-inspired application JSON, not a radio message.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import onnxruntime as ort
except ImportError:
    ort = None

if __package__:
    from .pfpd_perception.model_registry import (
        OUTPUT_SHAPE, MODELS, project_path, resolve_hazard_info,
        sha256_file, validate_session_contract,
    )
    from .pfpd_perception.temporal import TemporalHazardTracker, box_iou
else:
    from pfpd_perception.model_registry import (
        OUTPUT_SHAPE, MODELS, project_path, resolve_hazard_info,
        sha256_file, validate_session_contract,
    )
    from pfpd_perception.temporal import TemporalHazardTracker, box_iou

# Display-domain indices are retained for legacy consumers; raw ONNX class is 0.
CLASS_NAMES = {0: "pothole", 1: "construction_hazard", 2: "road_accident_obstacle"}
CLASS_COLORS = {index: MODELS[model]["color"] for index, model in
                enumerate(("pothole", "construction", "accident_obstacle"))}
MODEL_HAZARD_MAP = MODELS


def require_image_dependencies():
    if np is None or cv2 is None:
        raise RuntimeError("Real image inference requires NumPy and OpenCV in the project .venv")


def preprocess_image(image_path_or_array, input_size=768):
    """Letterbox BGR uint8 to contiguous normalized RGB NCHW; record integer pads.

    Returns the existing four-item API: tensor, original (H,W), (scale,left,top),
    and original BGR image. input_size may be an integer or (height,width).
    """
    require_image_dependencies()
    if isinstance(image_path_or_array, (str, Path)):
        path = project_path(image_path_or_array)
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode input image: {path}")
    else:
        image = image_path_or_array
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
        raise ValueError("Expected a nonempty uint8 BGR image with three channels")
    size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
    if len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
        raise ValueError("Input size must be positive (height,width)")
    h0, w0 = image.shape[:2]
    height, width = size
    scale = min(height / h0, width / w0)
    resized_width, resized_height = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    dh, dw = height - resized_height, width - resized_width
    top, left = dh // 2, dw // 2
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= np.float32(255.0)
    return tensor, (h0, w0), (scale, left, top), image


def _prediction_rows(preds):
    array = np.asarray(preds)
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[1] == 5:
        rows = array[0].T
    elif array.ndim == 2 and array.shape[0] == 5:
        rows = array.T
    elif array.ndim == 2 and array.shape[1] == 5:
        rows = array
    else:
        raise ValueError(f"Unsupported prediction shape {array.shape}; expected [1,5,N]")
    if not np.issubdtype(rows.dtype, np.number) or not np.isfinite(rows).all():
        raise ValueError("ONNX prediction contains nonfinite or nonnumeric values")
    if ((rows[:, 4] < 0) | (rows[:, 4] > 1)).any():
        raise ValueError("Class scores outside [0,1] violate the reviewed output contract")
    return rows.astype(np.float64, copy=False)


def postprocess(preds, orig_shape, ratio_pad, default_hazard_info=None,
                conf_thresh=0.35, iou_thresh=0.45):
    """Decode pixel xywh + one class score, source-scoped float NMS, then unpad.

    Confidence is inclusive. Degenerate/off-image boxes are rejected. Keeping
    floats through NMS avoids early truncation of small objects. `box` remains
    final rounded xyxy for existing consumers; bbox_xyxy_float preserves values.
    """
    if np is None:
        raise RuntimeError("Postprocessing requires NumPy")
    if not 0 <= conf_thresh <= 1 or not 0 <= iou_thresh <= 1:
        raise ValueError("Confidence/NMS thresholds must be finite values in [0,1]")
    if default_hazard_info is None:
        raise ValueError("Postprocessing requires authenticated source-model semantics")
    h0, w0 = orig_shape
    scale, left, top = ratio_pad
    if (min(h0, w0) <= 0 or not np.isfinite([scale, left, top]).all() or scale <= 0):
        raise ValueError("Invalid image dimensions or letterbox transform")
    rows = _prediction_rows(preds)
    indices = np.flatnonzero((rows[:, 4] >= conf_thresh) & (rows[:, 2] > 0) & (rows[:, 3] > 0))
    selected = rows[indices]
    if not len(selected):
        return []
    boxes = np.column_stack((selected[:, 0] - selected[:, 2] / 2,
                             selected[:, 1] - selected[:, 3] / 2,
                             selected[:, 0] + selected[:, 2] / 2,
                             selected[:, 1] + selected[:, 3] / 2))
    # Stable confidence order breaks ties by original proposal index.
    order = np.argsort(-selected[:, 4], kind="stable").tolist()
    kept = []
    while order:
        index = order.pop(0)
        kept.append(index)
        order = [other for other in order if box_iou(boxes[index], boxes[other]) <= iou_thresh]
    detections = []
    for index in kept:
        box = (boxes[index] - np.array([left, top, left, top])) / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w0)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h0)
        rounded = np.rint(box).astype(int).tolist()
        if box[2] <= box[0] or box[3] <= box[1] or rounded[2] <= rounded[0] or rounded[3] <= rounded[1]:
            continue
        detections.append({"class_id": 0, "raw_class_id": 0,
                           "class_name": default_hazard_info["name"],
                           "source_model_id": default_hazard_info["source_model_id"],
                           "confidence": float(selected[index, 4]), "box": rounded,
                           "bbox_xyxy_float": box.tolist(),
                           "proposal_index": int(indices[index]),
                           "status": "SINGLE_FRAME_UNCONFIRMED", "persistence_hits": None})
    return detections


def generate_etsi_denm_beacon(rsu_id, confirmed_hazards, speed_limit_kmh=40):
    """Preserved compatibility formatter; application JSON with explicit limits.

    Single-image predictions remain unconfirmed. Codes are coarse CDD V1.3.1
    mappings, and the broad accident/obstacle cause remains unknown. No transport
    or geographic localization is performed by this function.
    """
    now_ms = int(time.time() * 1000)
    beacon = {"format": "DENM-inspired application JSON", "standards_compliant": False,
              "cdd_version": "1.3.1", "transmitted": False,
              "header": {"protocolVersion": 2, "messageID": 1, "stationID": rsu_id},
              "denm": {"management": {"actionID": f"{rsu_id}_{uuid.uuid4().hex}",
                        "detectionTime": now_ms, "referenceTime": now_ms,
                        "validityDurationSec": 300, "transmissionIntervalMs": 100},
                       "situation": {"hazardCount": len(confirmed_hazards), "hazards": []}}}
    by_name = {record["name"]: record for record in MODELS.values()}
    for hazard in confirmed_hazards:
        record = by_name.get(hazard["class_name"])
        if record is None:
            raise ValueError(f"Unknown hazard semantics: {hazard['class_name']}")
        beacon["denm"]["situation"]["hazards"].append({
            "hazardType": hazard["class_name"].upper(), "trackID": hazard.get("track_id"),
            "rawClassID": hazard.get("raw_class_id", hazard["class_id"]),
            "sourceModelID": hazard.get("source_model_id"), "causeCode": record["cause_code"],
            "subCauseCode": 0 if record["cause_code"] is not None else None,
            "severity": "UNASSESSED", "confidence": float(hazard["confidence"]),
            "persistenceFrames": hazard.get("persistence_hits"),
            "validationStatus": hazard.get("status", "UNCONFIRMED"), "boundingBox": hazard["box"],
            "advisoryAction": "UNASSESSED", "advisorySpeedKmh": None})
    return beacon


def create_session(hazard_info, threads):
    if ort is None:
        raise RuntimeError("ONNX Runtime is unavailable; use the project's existing .venv")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(hazard_info["model_path"], options, providers=["CPUExecutionProvider"])
    metadata = validate_session_contract(session, hazard_info)
    return session, metadata


def runtime_manifest(session, threads):
    result = {"python": sys.version, "executable": sys.executable,
              "os": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "logical_cpus": __import__("os").cpu_count(),
              "numpy": np.__version__, "opencv": cv2.__version__, "onnxruntime": ort.__version__,
              "providers": session.get_providers(), "intra_op_threads": threads,
              "inter_op_threads": 1, "execution_mode": "ORT_SEQUENTIAL",
              "graph_optimization": "ORT_ENABLE_ALL", "hardware_scope": "actual_execution_host"}
    try:
        import psutil
        result.update({"physical_cpus": psutil.cpu_count(logical=False),
                       "total_ram_bytes": psutil.virtual_memory().total,
                       "process_rss_bytes": psutil.Process().memory_info().rss})
    except ImportError:
        pass
    return result


def run_image(args):
    started = time.perf_counter()
    require_image_dependencies()
    info = resolve_hazard_info(args.model)
    image_path = project_path(args.image)
    input_hash = sha256_file(image_path)
    asset_verified = time.perf_counter()
    session, metadata = create_session(info, args.threads)
    loaded = time.perf_counter()
    input_shape = tuple(session.get_inputs()[0].shape)
    blob, original_shape, transform, image = preprocess_image(image_path, input_shape[2:])
    preprocessed = time.perf_counter()
    preds = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: blob})[0]
    inferred = time.perf_counter()
    if tuple(preds.shape) != OUTPUT_SHAPE or preds.dtype != np.float32:
        raise ValueError(f"Unexpected runtime prediction contract: {preds.shape}, {preds.dtype}")
    detections = postprocess(preds, original_shape, transform, info, args.conf, args.iou)
    postprocessed = time.perf_counter()
    record = {"schema_version": "pfpd.image_inference.v1", "mode": "real_onnx",
              "run_id": uuid.uuid4().hex, "created_utc": datetime.now(timezone.utc).isoformat(),
              "validation_status": "SINGLE_FRAME_UNCONFIRMED", "source_id": args.rsu_id,
              "model": {"source_model_id": info["source_model_id"], "path": info["model_path"],
                        "sha256": info["sha256"], "raw_class_id": 0, "semantic_class": info["name"],
                        "input_name": "images", "input_shape": list(input_shape),
                        "input_type": "float32", "output_name": "output0", "metadata": metadata},
              "input": {"path": str(image_path), "sha256": input_hash,
                        "height": original_shape[0], "width": original_shape[1],
                        "tensor_sha256": hashlib.sha256(blob.tobytes()).hexdigest(),
                        "color_order": "RGB", "normalization": "uint8 / 255", "pad_value": 114,
                        "scale": transform[0], "left_pad": transform[1], "top_pad": transform[2],
                        "right_pad": input_shape[3] - round(original_shape[1] * transform[0]) - transform[1],
                        "bottom_pad": input_shape[2] - round(original_shape[0] * transform[0]) - transform[2]},
              "thresholds": {"confidence": args.conf, "nms_iou": args.iou, "calibrated": False},
              "raw_prediction": {"shape": list(preds.shape), "dtype": str(preds.dtype),
                       "sha256": hashlib.sha256(preds.tobytes()).hexdigest(),
                       "finite_values": int(np.isfinite(preds).sum()), "value_count": int(preds.size),
                       "max_class_score": float(preds[:, 4, :].max()),
                       "proposals_at_threshold": int((preds[:, 4, :] >= args.conf).sum())},
              "detections": detections, "detection_count": len(detections),
              "timings_ms": {"asset_validation": (asset_verified - started) * 1000,
                          "session_load": (loaded - asset_verified) * 1000,
                          "decode_preprocess": (preprocessed - loaded) * 1000,
                          "inference": (inferred - preprocessed) * 1000,
                          "postprocess": (postprocessed - inferred) * 1000,
                          "service": (postprocessed - loaded) * 1000,
                          "total_through_postprocess": (postprocessed - started) * 1000},
              "runtime": runtime_manifest(session, args.threads),
              "beacon": generate_etsi_denm_beacon(args.rsu_id, detections),
              "limitations": ["Single image; no temporal confirmation", "Scores are not accuracy or severity",
                              "No location mapping or transport", "Performance is one cold forward on this host"]}
    if args.benchmark:
        record["benchmark"] = benchmark_session(session, blob, args.iterations, args.warmup)
    if args.output_dir:
        save_evidence(record, preds, image, info, project_path(args.output_dir))
    if args.json_out:
        with project_path(args.json_out).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
    return record


def benchmark_session(session, blob, iterations, warmup):
    for _ in range(warmup):
        session.run(None, {"images": blob})
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        outputs = session.run(None, {"images": blob})
        if tuple(outputs[0].shape) != OUTPUT_SHAPE or not np.isfinite(outputs[0]).all():
            raise ValueError("Invalid predictions during benchmark")
        durations.append((time.perf_counter() - start) * 1000)
    mean = float(np.mean(durations))
    return {"scope": "single_model_inference_only", "iterations": iterations, "warmup": warmup,
            "durations_ms": durations, "mean_ms": mean, "p50_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)), "inference_passes_per_s": 1000 / mean}


def save_evidence(record, preds, image, info, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{info['source_model_id']}_{record['run_id']}"
    paths = {key: directory / f"{stem}{suffix}" for key, suffix in
             (("raw_prediction", ".npy"), ("annotation", ".jpg"), ("record", ".json"))}
    # Exclusive creation prevents replacing any input or historical evidence.
    with paths["raw_prediction"].open("xb") as stream:
        np.save(stream, preds, allow_pickle=False)
    annotation = image.copy()
    for det in record["detections"]:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(annotation, (x1, y1), (x2, y2), info["color"], 2)
        cv2.putText(annotation, f"{det['class_name']} {det['confidence']:.3f}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, info["color"], 1, cv2.LINE_AA)
    encoded_ok, encoded = cv2.imencode(".jpg", annotation)
    if not encoded_ok:
        raise OSError("OpenCV could not encode annotation")
    with paths["annotation"].open("xb") as stream:
        stream.write(encoded.tobytes())
    record["artifacts"] = {key: {"path": str(path), "sha256": sha256_file(path)}
                           for key, path in paths.items() if key != "record"}
    record["artifacts"]["record"] = {"path": str(paths["record"])}
    with paths["record"].open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def mock_stream(rsu_id):
    tracker = TemporalHazardTracker()
    obstacle = {"class_id": 0, "class_name": "road_accident_obstacle", "confidence": 0.94,
                "source_model_id": "accident_obstacle", "box": [200, 250, 450, 420]}
    frames = [[obstacle], [{**obstacle, "confidence": 0.95}], [{**obstacle, "confidence": 0.96}], [], []]
    results = []
    for frame_id, raw in enumerate(frames):
        confirmed = tracker.update(raw, frame_id=frame_id, capture_time_s=frame_id * 0.1,
                                   source_id=rsu_id, source_model_id="accident_obstacle")
        results.append({"frame_id": frame_id, "raw_count": len(raw), "confirmed": confirmed})
    return {"mode": "mock", "inference_executed": False, "synthetic_tracker_fixture": True,
            "frames": results, "limitations": ["Synthetic detections; not real model or video evidence"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODELS["pothole"]["path"], help="Reviewed ONNX artifact; relative to project")
    parser.add_argument("--image", default="", help="Input image; relative to project")
    parser.add_argument("--conf", type=float, default=0.35, help="Provisional inclusive score threshold, not calibrated precision")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--rsu-id", default="RSU_ZONE_04")
    parser.add_argument("--simulate-stream", action="store_true", help="Explicit MOCK tracker-only fixture")
    parser.add_argument("--benchmark", action="store_true", help="Real ONNX inference-only timing on actual host")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", help="Optional unique prediction/annotation/JSON evidence outputs")
    parser.add_argument("--json-out", help="Optional exact JSON path; must not already exist")
    args = parser.parse_args(argv)
    if (not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1
            or args.threads < 1 or args.iterations < 1 or args.warmup < 0):
        parser.error("Invalid thresholds/thread count/benchmark iterations")
    if args.simulate_stream and (args.image or args.benchmark or args.output_dir or args.json_out):
        parser.error("Mock stream cannot be combined with real inference or evidence outputs")
    if not args.simulate_stream and not args.image and not args.benchmark:
        parser.error("Select --image, --benchmark, or explicit --simulate-stream mock")
    if args.benchmark and not args.image and (args.output_dir or args.json_out):
        parser.error("Benchmark-only evidence uses stdout; use --image for asset-linked output files")
    try:
        if args.json_out and project_path(args.json_out).exists():
            raise FileExistsError(f"JSON output already exists: {project_path(args.json_out)}")
        if args.simulate_stream:
            record = mock_stream(args.rsu_id)
        elif args.image:
            record = run_image(args)
        else:
            require_image_dependencies()
            info = resolve_hazard_info(args.model)
            session, metadata = create_session(info, args.threads)
            blob = np.zeros(tuple(session.get_inputs()[0].shape), dtype=np.float32)
            record = {"mode": "real_onnx", "input_kind": "synthetic_zero_tensor_benchmark",
                      "model": info, "metadata": metadata, "runtime": runtime_manifest(session, args.threads),
                      "benchmark": benchmark_session(session, blob, args.iterations, args.warmup)}
        print(json.dumps(record, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "mode": "real_onnx" if not args.simulate_stream else "mock",
                          "error_type": type(exc).__name__, "message": str(exc),
                          "inference_result": None, "mock_fallback": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
