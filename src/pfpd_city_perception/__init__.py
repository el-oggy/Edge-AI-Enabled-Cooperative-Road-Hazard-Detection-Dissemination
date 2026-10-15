"""
PFPD Smart City Perception Module (Phase SC1)
Camera perception adapter integrating verified immutable INT8 ONNX models
with RSU-mounted camera sensors for junction hazard detection.
"""

from .camera_perception import CityCameraPerception, CameraObservation

__all__ = ["CityCameraPerception", "CameraObservation"]
