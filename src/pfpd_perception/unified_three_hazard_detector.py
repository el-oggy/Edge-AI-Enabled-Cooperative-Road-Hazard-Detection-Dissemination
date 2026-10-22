"""
Unified Three-Hazard Perception Interface (PFPD Step 2 Deliverable)
===================================================================
Target Hardware: Raspberry Pi 5 / Edge RSU (BCM2712 Cortex-A76)

Combines the outputs of the three authentic specialist YOLO11s models:
  1. quantization/pothole_yolo11s_int8.onnx            -> Class 0: pothole
  2. quantization/construction_yolo11s_int8.onnx       -> Class 1: construction_hazard
  3. quantization/accident_obstacle_yolo11s_int8.onnx  -> Class 2: road_accident_obstacle

Provides a single unified callable interface that evaluates all three hazard domains
for every frame and returns a consolidated multi-class detection payload with unified NMS.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
    import numpy as np
    import onnxruntime as ort
except ImportError:
    cv2 = None
    np = None
    ort = None

try:
    from src.pfpd_perception.model_registry import (
        MODELS, project_path, resolve_hazard_info, sha256_file
    )
    from src.pfpd_perception.temporal import box_iou
    from src import edge_rpi_hazard_inference as edge
except ImportError:
    from pfpd_perception.model_registry import (
        MODELS, project_path, resolve_hazard_info, sha256_file
    )
    from pfpd_perception.temporal import box_iou
    import edge_rpi_hazard_inference as edge

# Unified 3-Class Taxonomy
UNIFIED_CLASSES = {
    0: "pothole",
    1: "construction_hazard",
    2: "road_accident_obstacle"
}

DOMAIN_TO_UNIFIED_ID = {
    "pothole": 0,
    "construction": 1,
    "accident_obstacle": 2
}

DEFAULT_CONF_THRESHOLDS = {
    "pothole": 0.25,
    "construction": 0.25,
    "accident_obstacle": 0.35
}


class UnifiedThreeHazardDetector:
    """
    Unified multi-hazard detector wrapping the 3 authentic YOLO11s INT8 models.
    Evaluates potholes, construction, and accident/debris sequentially per frame
    and merges them into a single coherent multi-class detection set.
    """

    def __init__(
        self,
        conf_thresholds: Optional[Dict[str, float]] = None,
        iou_thresh: float = 0.45,
        cross_class_iou_thresh: float = 0.65,
        allow_cross_class_suppression: bool = False,
        threads: int = 2,
        input_size: Tuple[int, int] = (768, 768)
    ):
        if cv2 is None or np is None or ort is None:
            raise RuntimeError("cv2, numpy, and onnxruntime are required for UnifiedThreeHazardDetector")

        self.iou_thresh = iou_thresh
        self.cross_class_iou_thresh = cross_class_iou_thresh
        self.allow_cross_class_suppression = allow_cross_class_suppression
        self.threads = max(1, threads)
        self.input_size = input_size
        self.conf_thresholds = dict(DEFAULT_CONF_THRESHOLDS)
        if conf_thresholds:
            self.conf_thresholds.update(conf_thresholds)

        self.sessions = {}
        self.metadata = {}
        self._init_models()

    def _init_models(self):
        """Initialize all three authenticated specialist ONNX sessions with thread tuning."""
        for domain, record in MODELS.items():
            info = resolve_hazard_info(record["path"])
            session_opts = ort.SessionOptions()
            session_opts.intra_op_num_threads = self.threads
            session_opts.inter_op_num_threads = 1
            session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            sess = ort.InferenceSession(
                str(info["model_path"]),
                sess_options=session_opts,
                providers=["CPUExecutionProvider"]
            )
            meta = edge.validate_session_contract(sess, info)
            self.sessions[domain] = {
                "session": sess,
                "info": info,
                "input_name": sess.get_inputs()[0].name,
                "output_name": sess.get_outputs()[0].name
            }
            self.metadata[domain] = meta

    def preprocess(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float, float], np.ndarray]:
        """Letterbox and normalize input image to [1, 3, 768, 768] float32 tensor."""
        if isinstance(image_input, (str, Path)):
            img_path = project_path(image_input)
            img0 = cv2.imread(str(img_path))
            if img0 is None:
                raise FileNotFoundError(f"Cannot load image from {img_path}")
        elif isinstance(image_input, np.ndarray):
            img0 = image_input
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        orig_shape = img0.shape[:2]
        target_h, target_w = self.input_size
        h0, w0 = orig_shape
        scale = min(target_w / w0, target_h / h0)
        nw, nh = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
        resized = cv2.resize(img0, (nw, nh), interpolation=cv2.INTER_LINEAR)

        dw, dh = target_w - nw, target_h - nh
        pad_x, pad_y = dw / 2.0, dh / 2.0
        left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
        right, bottom = dw - left, dh - top

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        blob = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        ) / 255.0
        return blob, orig_shape, (scale, pad_x, pad_y), img0

    def _class_aware_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies class-aware Non-Maximum Suppression (Intra-Class NMS).

        Crucial C-ITS Safety Invariant:
        Distinct hazard classes (e.g., a pothole occurring inside a construction work zone,
        or an obstacle adjacent to a vehicle collision) must NOT suppress each other.
        Suppression is performed strictly within each individual class domain.

        If allow_cross_class_suppression is explicitly enabled, only near-identical duplicate boxes
        (IoU > cross_class_iou_thresh) across different classes are resolved by confidence.
        """
        if len(detections) <= 1:
            return detections

        # Group detections by unified_class_id for strict intra-class NMS
        per_class: Dict[int, List[Dict[str, Any]]] = {}
        for det in detections:
            cid = det.get("unified_class_id", 0)
            per_class.setdefault(cid, []).append(det)

        kept: List[Dict[str, Any]] = []
        for cid, class_dets in per_class.items():
            class_dets.sort(key=lambda d: d["confidence"], reverse=True)
            class_kept: List[Dict[str, Any]] = []
            for det in class_dets:
                overlap = False
                for k in class_kept:
                    iou = box_iou(det["bbox_xyxy_float"], k["bbox_xyxy_float"])
                    if iou > self.iou_thresh:
                        overlap = True
                        break
                if not overlap:
                    class_kept.append(det)
            kept.extend(class_kept)

        # Optional cross-class deduplication only if explicitly configured
        if getattr(self, "allow_cross_class_suppression", False) and len(kept) > 1:
            kept.sort(key=lambda d: d["confidence"], reverse=True)
            cross_kept: List[Dict[str, Any]] = []
            for det in kept:
                overlap = False
                for k in cross_kept:
                    if det["unified_class_id"] != k["unified_class_id"]:
                        iou = box_iou(det["bbox_xyxy_float"], k["bbox_xyxy_float"])
                        if iou > self.cross_class_iou_thresh:
                            overlap = True
                            break
                if not overlap:
                    cross_kept.append(det)
            kept = cross_kept

        return kept

    def detect(self, image_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Execute full genuine three-class hazard perception on a single frame.
        Measures decode/preprocess, per-model inference, and consolidated postprocess.
        """
        t_start = time.perf_counter()

        # 1. Preprocess
        t_prep_0 = time.perf_counter()
        blob, orig_shape, ratio_pad, orig_img = self.preprocess(image_input)
        prep_ms = (time.perf_counter() - t_prep_0) * 1000.0

        all_detections: List[Dict[str, Any]] = []
        model_timings: Dict[str, float] = {}

        # 2. Sequential Inference Across All 3 Specialists
        t_infer_total_0 = time.perf_counter()
        for domain in ["pothole", "construction", "accident_obstacle"]:
            item = self.sessions[domain]
            t0 = time.perf_counter()
            raw_out = item["session"].run([item["output_name"]], {item["input_name"]: blob})[0]
            lat = (time.perf_counter() - t0) * 1000.0
            model_timings[domain] = round(lat, 2)

            conf_th = self.conf_thresholds[domain]
            domain_dets = edge.postprocess(
                raw_out, orig_shape, ratio_pad, item["info"],
                conf_thresh=conf_th, iou_thresh=self.iou_thresh
            )

            # Assign Unified Class IDs
            unified_class_id = DOMAIN_TO_UNIFIED_ID[domain]
            unified_class_name = UNIFIED_CLASSES[unified_class_id]

            for d in domain_dets:
                d["unified_class_id"] = unified_class_id
                d["unified_class_name"] = unified_class_name
                all_detections.append(d)

        total_infer_ms = (time.perf_counter() - t_infer_total_0) * 1000.0

        # 3. Postprocess & Cross-Class NMS
        t_post_0 = time.perf_counter()
        final_detections = self._class_aware_nms(all_detections)
        post_ms = (time.perf_counter() - t_post_0) * 1000.0

        total_e2e_ms = (time.perf_counter() - t_start) * 1000.0

        per_domain_counts = {
            "pothole": sum(1 for d in final_detections if d["unified_class_id"] == 0),
            "construction_hazard": sum(1 for d in final_detections if d["unified_class_id"] == 1),
            "road_accident_obstacle": sum(1 for d in final_detections if d["unified_class_id"] == 2)
        }

        return {
            "timestamp": time.time(),
            "total_detections": len(final_detections),
            "detections": final_detections,
            "per_domain_counts": per_domain_counts,
            "timings_ms": {
                "preprocess": round(prep_ms, 2),
                "model_inferences": model_timings,
                "total_inference": round(total_infer_ms, 2),
                "postprocess_nms": round(post_ms, 2),
                "total_e2e": round(total_e2e_ms, 2)
            },
            "e2e_fps": round(1000.0 / total_e2e_ms, 2) if total_e2e_ms > 0 else 0.0,
            "image_shape": orig_shape
        }
