import sys
from pathlib import Path
import numpy as np


# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pfpd_city_protocol.contracts import EXPECTED_MODEL_HASHES
from pfpd_city_perception.camera_perception import CityCameraPerception

def test_perception_initialization_and_hashes():
    perception = CityCameraPerception()
    assert len(perception.models) == 3
    assert "POTHOLE" in perception.models
    assert "CONSTRUCTION_HAZARD" in perception.models
    assert "ROAD_ACCIDENT_OBSTACLE" in perception.models

    assert perception.models["POTHOLE"]["hash"] == EXPECTED_MODEL_HASHES["pothole"]
    assert perception.models["CONSTRUCTION_HAZARD"]["hash"] == EXPECTED_MODEL_HASHES["construction"]
    assert perception.models["ROAD_ACCIDENT_OBSTACLE"]["hash"] == EXPECTED_MODEL_HASHES["accident_obstacle"]

def test_clean_frame_negative():
    perception = CityCameraPerception()
    clean_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = perception.detect_hazards(clean_frame, rsu_id="RSU_J0")
    # Clean baseline frame should produce 0 false positive detections
    assert len(detections) == 0


def test_inference_with_real_image():
    # Test on an existing dataset image if available
    repo_root = Path(__file__).resolve().parents[2]
    sample_imgs = list(repo_root.glob("data/**/*.jpg")) + list(repo_root.glob("data/**/*.png"))
    if sample_imgs:
        img_path = sample_imgs[0]
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            perception = CityCameraPerception()
            dets = perception.detect_hazards(img, rsu_id="RSU_J0")
            assert isinstance(dets, list)

if __name__ == "__main__":
    test_perception_initialization_and_hashes()
    test_clean_frame_negative()
    print("Perception tests passed successfully!")
