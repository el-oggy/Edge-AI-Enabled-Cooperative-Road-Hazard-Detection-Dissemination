"""
PFPD Smart City Protocol & Data Contracts (Phase SC0)
Defines immutable data schemas, coordinate conversions, and model validation hashes
for the Edge-AI Cooperative Road Hazard Perception & Dissemination Architecture.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import hashlib
from pathlib import Path
import json
import time

# Official SHA-256 Hash Manifest for Immutable Models
EXPECTED_MODEL_HASHES = {
    "pothole": "289c7d60b8d920ef1da65681629c8fc854ccaf7b01474afc5997ba89b1ca077d",
    "construction": "4144423ff28f6fab297828be7965a8104600359fbd2ed58b0da45e16fa7788f8",
    "accident_obstacle": "dc2985e915d44ff419d205bcae0df9fc1cfdf89960e794730b3afcb2134b96bf"
}

def verify_model_integrity(model_path: Path) -> str:
    """Computes SHA-256 hash of an ONNX weight file to ensure zero tampering."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    hasher = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class HazardDetection:
    hazard_id: str
    category: str  # 'POTHOLE', 'CONSTRUCTION_HAZARD', 'ROAD_ACCIDENT_OBSTACLE'
    confidence: float
    box: List[int]
    advisory: str  # 'REDUCE_SPEED', 'MERGE_LANE', 'MANDATORY_LANE_CHANGE'
    detected_timestamp: float
    rsu_id: str
    location_m: Tuple[float, float] = (0.0, 0.0)

@dataclass
class VehicleState:
    vehicle_id: str
    x: float
    y: float
    speed_mps: float
    heading_deg: float
    lane_id: str
    vehicle_type: str
    active_advisories: List[str] = field(default_factory=list)

@dataclass
class TrafficLightState:
    tls_id: str
    phase_index: int
    state_string: str
    duration_s: float

@dataclass
class RSUBeacon:
    rsu_id: str
    position_xy: Tuple[float, float]
    range_m: float
    active_hazards: List[HazardDetection] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class SimulationSnapshot:
    tick: int
    sim_time_s: float
    vehicles: List[VehicleState] = field(default_factory=list)
    traffic_lights: List[TrafficLightState] = field(default_factory=list)
    rsus: List[RSUBeacon] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))
