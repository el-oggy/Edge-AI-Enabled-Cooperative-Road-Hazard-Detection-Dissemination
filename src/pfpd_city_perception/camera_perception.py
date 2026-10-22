"""
PFPD Smart City Camera Perception Adapter (Phase SC1)
Connects physical/synthetic camera sensors to verified immutable INT8 ONNX models
and produces structured HazardDetection events for cooperative RSU dissemination.
"""

import time
import uuid
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import onnxruntime as ort

from pfpd_city_protocol.contracts import (
    HazardDetection,
    EXPECTED_MODEL_HASHES,
    verify_model_integrity
)

CATEGORY_ADVISORY_MAP = {
    "POTHOLE": "REDUCE_SPEED",
    "CONSTRUCTION_HAZARD": "MERGE_LANE",
    "ROAD_ACCIDENT_OBSTACLE": "MANDATORY_LANE_CHANGE",
}

@dataclass
class CameraObservation:
    camera_id: str
    rsu_id: str
    timestamp_s: float
    frame: np.ndarray
    field_of_view_deg: float = 90.0
    mounting_height_m: float = 6.5
    orientation_deg: float = 0.0

class CityCameraPerception:
    """
    Manages ONNX runtime inference across the 3 immutable hazard detection models:
    - pothole_yolo11s_int8.onnx
    - construction_yolo11s_int8.onnx
    - accident_obstacle_yolo11s_int8.onnx
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        num_threads: int = 4
    ):
        if model_dir is None:
            # Default to repo quantization directory
            repo_root = Path(__file__).resolve().parents[2]
            model_dir = repo_root / "quantization"

        self.model_dir = Path(model_dir)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_threads = num_threads
        
        # Verify integrity and initialize sessions
        self.models: Dict[str, Dict[str, Any]] = {}
        self._init_models()

    def _init_models(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        model_specs = [
            ("POTHOLE", "pothole_yolo11s_int8.onnx", "pothole"),
            ("CONSTRUCTION_HAZARD", "construction_yolo11s_int8.onnx", "construction"),
            ("ROAD_ACCIDENT_OBSTACLE", "accident_obstacle_yolo11s_int8.onnx", "accident_obstacle")
        ]

        for category, filename, hash_key in model_specs:
            model_path = self.model_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"Required model weight missing: {model_path}")
            
            # Verify SHA-256
            actual_hash = verify_model_integrity(model_path)
            expected_hash = EXPECTED_MODEL_HASHES.get(hash_key)
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(
                    f"Integrity violation for {filename}! "
                    f"Expected {expected_hash}, got {actual_hash}"
                )

            session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape

            self.models[category] = {
                "session": session,
                "input_name": input_name,
                "input_shape": input_shape,
                "hash": actual_hash,
                "path": model_path
            }

    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: int = 768) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, int, int]]:
        """Preprocesses BGR image with standard letterbox resize and NCHW float32 normalization."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel uint8 BGR numpy array")

        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = target_size - new_w, target_size - new_h
        left, top = int(round(pad_w / 2 - 0.1)), int(round(pad_h / 2 - 0.1))
        right, bottom = pad_w - left, pad_h - top
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # BGR -> RGB, HWC -> CHW, float32 [0, 1]
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return tensor, (h, w), (scale, left, top)

    def _decode_yolo_outputs(
        self,
        output: np.ndarray,
        orig_shape: Tuple[int, int],
        ratio_pad: Tuple[float, int, int]
    ) -> List[Dict[str, Any]]:
        """Decodes raw YOLO output tensor and applies Non-Maximum Suppression (NMS)."""
        scale, pad_x, pad_y = ratio_pad
        h, w = orig_shape
        
        preds = np.asarray(output)
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        if preds.shape[0] == 5:
            preds = preds.T
            
        candidates = []
        for cx, cy, box_w, box_h, score in preds:
            if score < self.conf_thresh or box_w <= 0 or box_h <= 0:
                continue
            x1 = (cx - box_w / 2 - pad_x) / scale
            y1 = (cy - box_h / 2 - pad_y) / scale
            x2 = (cx + box_w / 2 - pad_x) / scale
            y2 = (cy + box_h / 2 - pad_y) / scale
            
            x1 = max(0.0, min(float(w), float(x1)))
            y1 = max(0.0, min(float(h), float(y1)))
            x2 = max(0.0, min(float(w), float(x2)))
            y2 = max(0.0, min(float(h), float(y2)))
            
            if x2 > x1 and y2 > y1:
                candidates.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(score)
                })

        # NMS
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        kept = []
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            b1 = best["box"]
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            survivors = []
            for other in candidates:
                b2 = other["box"]
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                xx1 = max(b1[0], b2[0])
                yy1 = max(b1[1], b2[1])
                xx2 = min(b1[2], b2[2])
                yy2 = min(b1[3], b2[3])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                if iou < self.iou_thresh:
                    survivors.append(other)
            candidates = survivors
        return kept

    def detect_hazards(
        self,
        image: np.ndarray,
        rsu_id: str,
        location_m: Tuple[float, float] = (0.0, 0.0),
        target_categories: Optional[List[str]] = None
    ) -> List[HazardDetection]:
        """
        Runs neural inference across specified or all model categories on the camera image.
        Returns a list of structured HazardDetection objects.
        """
        categories = target_categories or list(self.models.keys())
        tensor, orig_shape, ratio_pad = self.preprocess_image(image)
        
        detections: List[HazardDetection] = []
        now = time.time()

        for cat in categories:
            if cat not in self.models:
                continue
            model_info = self.models[cat]
            session = model_info["session"]
            input_name = model_info["input_name"]
            
            outputs = session.run(None, {input_name: tensor})
            raw_output = outputs[0]
            
            decoded = self._decode_yolo_outputs(raw_output, orig_shape, ratio_pad)
            for item in decoded:
                detection = HazardDetection(
                    hazard_id=f"HAZ-{cat[:4]}-{uuid.uuid4().hex[:6]}",
                    category=cat,
                    confidence=item["confidence"],
                    box=item["box"],
                    advisory=CATEGORY_ADVISORY_MAP.get(cat, "REDUCE_SPEED"),
                    detected_timestamp=now,
                    rsu_id=rsu_id,
                    location_m=location_m
                )
                detections.append(detection)
                
        return detections
