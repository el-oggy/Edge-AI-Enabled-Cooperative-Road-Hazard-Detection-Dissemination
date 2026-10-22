"""Persistent execution of the three authenticated specialist ONNX models."""
from __future__ import annotations

import math
from pathlib import Path
import platform
import sys
import time
import uuid

try:
    from src import edge_rpi_hazard_inference as edge
except ImportError:
    import edge_rpi_hazard_inference as edge
from .model_registry import MODELS, resolve_hazard_info, sha256_file


class ThreeModelInference:
    """One session per model; every admitted photograph is evaluated by all three."""

    def __init__(self, conf=.35, iou=.45, threads=4):
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (conf, iou)):
            raise ValueError("Confidence and IoU must be finite values in [0,1]")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        edge._require_numpy()
        edge._require_cv2()
        self.conf, self.iou, self.threads = conf, iou, threads
        self.models = {}
        for model_id, model in MODELS.items():
            info = resolve_hazard_info(model["path"])
            start = time.perf_counter()
            session, metadata = edge.create_session(info["model_path"], info, threads)
            self.models[model_id] = {"info": info, "session": session,
                "metadata": metadata, "load_ms": (time.perf_counter() - start) * 1000}

    def identities(self):
        return {key: {"path": item["info"]["model_path"], "sha256": item["info"]["sha256"],
                       "metadata": item["metadata"], "load_ms": item["load_ms"]}
                for key, item in self.models.items()}

    def infer_image(self, image_path, encounter=None):
        start = time.perf_counter()
        path = edge.project_path(image_path)
        tensor, shape, transform, image = edge.preprocess_image(path, (768, 768))
        preprocess_ms = (time.perf_counter() - start) * 1000
        input_info = {"path": str(path), "sha256": sha256_file(path), "shape": list(shape),
                      "tensor_shape": list(tensor.shape),
                      "tensor_sha256": __import__("hashlib").sha256(tensor.tobytes()).hexdigest()}
        for model_id, item in self.models.items():
            infer_start = time.perf_counter()
            raw = item["session"].run(["output0"], {"images": tensor})[0]
            infer_done = time.perf_counter()
            if raw.dtype != edge.np.float32 or raw.shape != (1, 5, 12096):
                raise ValueError(f"Invalid output from {model_id}: {raw.dtype} {raw.shape}")
            detections = edge.postprocess(raw, shape, transform, item["info"], self.conf, self.iou)
            completed = time.perf_counter()
            yield {"schema": "pfpd.live_image.v1", "run_id": str(uuid.uuid4()),
                "mode": "real_onnx", "inference_executed": True, "mock_fallback": False,
                "model": {"source_model_id": model_id, "path": item["info"]["model_path"],
                          "sha256": item["info"]["sha256"], "metadata": item["metadata"]},
                "input": dict(input_info), "encounter": dict(encounter or {}),
                "completed_wall_s": completed,
                "thresholds": {"confidence": self.conf, "iou": self.iou, "calibrated": False},
                "detections": detections, "detection_count": len(detections),
                "raw_output": {"shape": list(raw.shape), "dtype": str(raw.dtype),
                               "sha256": __import__("hashlib").sha256(raw.tobytes()).hexdigest(),
                               "max_score": float(raw[0, 4].max())},
                "timings_ms": {"model_load": item["load_ms"], "decode_preprocess": preprocess_ms,
                    "inference": (infer_done - infer_start) * 1000,
                    "postprocess": (completed - infer_done) * 1000},
                "runtime": {"python": sys.version, "platform": platform.platform(),
                    "onnxruntime": edge.ort.__version__, "numpy": edge.np.__version__,
                    "opencv": edge.cv2.__version__, "providers": item["session"].get_providers(),
                    "intra_op_threads": self.threads},
                "limitations": ["Mapped road photograph, not a camera rendered by SUMO.",
                    "Single image; no temporal confirmation or accuracy claim."],
                "_annotation": edge._annotate(image, detections), "_raw_output_array": raw}
