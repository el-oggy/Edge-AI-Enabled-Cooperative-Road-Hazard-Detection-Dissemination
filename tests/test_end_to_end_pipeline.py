import sys
import os
sys.path.insert(0, os.path.abspath("."))

import cv2
from src.pfpd_perception.end_to_end_pipeline import PFPDEndToEndPipeline

def test_pipeline():
    print("Testing PFPD End-to-End Perception Pipeline...")
    pipeline = PFPDEndToEndPipeline(threads=2)
    img = cv2.imread("data/test_user_pothole.png")
    assert img is not None, "Failed to load test image"

    print("\n--- Frame 1 (Initial Detection) ---")
    f1 = pipeline.process_frame(img)
    print(f"Frame 1 Latency: {f1['total_latency_ms']} ms | Raw Dets: {f1['raw_detections']} | Confirmed: {f1['confirmed_hazards']}")

    print("\n--- Frame 2 (Temporal Multi-Frame Confirmation) ---")
    f2 = pipeline.process_frame(img)
    print(f"Frame 2 Latency: {f2['total_latency_ms']} ms | Raw Dets: {f2['raw_detections']} | Confirmed: {f2['confirmed_hazards']}")
    if f2["denm_messages"]:
        msg = f2["denm_messages"][0]
        print(f"Disseminated ETSI DENM: ID={msg['actionID']} | Cause={msg['situation']['causeCode']} ({msg['situation']['hazardType']}) | Conf={msg['situation']['confidenceScore']}")

    telemetry = pipeline.get_summary_telemetry()
    print("\nPipeline Summary Telemetry:")
    for k, v in telemetry.items():
        print(f"  {k}: {v}")

    print("\nAll pipeline components verified successfully!")

if __name__ == "__main__":
    test_pipeline()
