"""Identity and contracts for the three reviewed, immutable specialist artifacts."""

import ast
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_SHAPE = (1, 3, 768, 768)
OUTPUT_SHAPE = (1, 5, 12096)
MODELS = {
    "pothole": {
        "path": "quantization/pothole_yolo11s_int8.onnx",
        "sha256": "289c7d60b8d920ef1da65681629c8fc854ccaf7b01474afc5997ba89b1ca077d",
        "id": 0, "name": "pothole", "cause_code": 9,
        "color": (0, 140, 255),
    },
    "construction": {
        "path": "quantization/construction_yolo11s_int8.onnx",
        "sha256": "4144423ff28f6fab297828be7965a8104600359fbd2ed58b0da45e16fa7788f8",
        "id": 0, "name": "construction_hazard", "cause_code": 3,
        "color": (0, 0, 255),
    },
    "accident_obstacle": {
        "path": "quantization/accident_obstacle_yolo11s_int8.onnx",
        "sha256": "dc2985e915d44ff419d205bcae0df9fc1cfdf89960e794730b3afcb2134b96bf",
        "id": 0, "name": "road_accident_obstacle", "cause_code": None,
        "color": (0, 255, 255),
    },
}


def project_path(path):
    """Relative assets refer to the project, independent of shell working directory."""
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT_DIR / value).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_hazard_info(model_path):
    """Authenticate contents, never infer the domain from a filename substring."""
    path = project_path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    digest = sha256_file(path)
    for model_id, record in MODELS.items():
        if digest == record["sha256"]:
            return {**record, "source_model_id": model_id,
                    "model_path": str(path), "raw_class_id": 0}
    raise ValueError(f"Unreviewed model SHA-256 {digest}: {path}; use a registered artifact")


def validate_session_contract(session, hazard_info):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "images"
            or inputs[0].type != "tensor(float)"
            or tuple(inputs[0].shape) != INPUT_SHAPE):
        raise ValueError("Unsupported ONNX input contract; expected images float32 [1,3,768,768]")
    if (len(outputs) != 1 or outputs[0].name != "output0"
            or outputs[0].type != "tensor(float)"
            or tuple(outputs[0].shape) != OUTPUT_SHAPE):
        raise ValueError("Unsupported ONNX output contract; expected output0 float32 [1,5,12096]")
    metadata = session.get_modelmeta().custom_metadata_map
    try:
        names = ast.literal_eval(metadata.get("names", ""))
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Missing or invalid ONNX class-name metadata") from exc
    if (names != {0: hazard_info["name"]} or metadata.get("task") != "detect"
            or metadata.get("end2end") != "False"):
        raise ValueError("ONNX metadata contradicts reviewed specialist semantics or detect contract")
    return metadata
