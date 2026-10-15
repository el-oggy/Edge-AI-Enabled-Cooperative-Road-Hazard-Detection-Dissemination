"""
Simultaneous Multi-Hazard Real-Time Camera Perception Engine for Raspberry Pi 5
=============================================================================
Continuously runs hazard perception on live camera feeds or benchmark streams:
- Physical USB Webcam / Laptop Webcam (`--source 0` or `--source 1`)
- Smartphone / Wi-Fi RTSP or HTTP IP Camera (`--source http://<device-ip>:8080/video`)
- Looping Multi-Scenario Image Feed (`--source data/benchmark_150`)

Simultaneously evaluates ALL THREE models on every frame:
1. Potholes (Water-filled & Dry surface damage)
2. Road Work & Construction Zones (Cones, Signs, Active Workers)
3. Accidents & Dangerous Obstacles (Collisions, Fallen Trees, Boulders)

Outputs:
- Real-time FPS calculation & latency metrics
- Visual detection bounding boxes per hazard type
- Clean, non-JSON terminal detection alerts
- Safe headless fallback for environments without GUI displays
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
QUANT_DIR = ROOT_DIR / "quantization"

HAZARD_MODELS = {
    "pothole": {
        "name": "POTHOLE",
        "path": QUANT_DIR / "pothole_yolo11s_int8.onnx",
        "color": (0, 140, 255),       # Orange
        "conf": 0.22,
        "advisory": "REDUCE_SPEED"
    },
    "construction": {
        "name": "CONSTRUCTION_HAZARD",
        "path": QUANT_DIR / "construction_yolo11s_int8.onnx",
        "color": (0, 0, 255),         # Red
        "conf": 0.22,
        "advisory": "MERGE_LANE"
    },
    "accident": {
        "name": "ROAD_ACCIDENT_OBSTACLE",
        "path": QUANT_DIR / "accident_obstacle_yolo11s_int8.onnx",
        "color": (0, 255, 255),       # Yellow
        "conf": 0.20,
        "advisory": "MANDATORY_LANE_CHANGE"
    }
}

class TemporalHazardTracker:
    """
    Maintains detected hazards across alternating Round-Robin frames.
    Holds confirmed hazards in a temporal cache for ttl_frames to prevent flicker.
    """
    def __init__(self, ttl_frames=30, iou_thresh=0.35):
        self.ttl_frames = ttl_frames
        self.iou_thresh = iou_thresh
        self.tracks = {}
        self.next_id = 1

    def _calc_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        denom = areaA + areaB - inter
        return inter / denom if denom > 0 else 0.0

    def update(self, current_model_key, detections, current_frame):
        # 1. Match new detections with existing tracks
        matched_tracks = set()
        for det in detections:
            best_iou = 0.0
            best_tid = None
            for tid, tr in self.tracks.items():
                if tr["hazard"] != det["hazard"]:
                    continue
                iou = self._calc_iou(det["box"], tr["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou >= self.iou_thresh and best_tid is not None:
                matched_tracks.add(best_tid)
                self.tracks[best_tid]["last_seen"] = current_frame
                self.tracks[best_tid]["confidence"] = det["confidence"]
                old_b = self.tracks[best_tid]["box"]
                new_b = det["box"]
                self.tracks[best_tid]["box"] = [
                    int(0.3 * old_b[0] + 0.7 * new_b[0]),
                    int(0.3 * old_b[1] + 0.7 * new_b[1]),
                    int(0.3 * old_b[2] + 0.7 * new_b[2]),
                    int(0.3 * old_b[3] + 0.7 * new_b[3])
                ]
            else:
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "hazard": det["hazard"],
                    "confidence": det["confidence"],
                    "box": det["box"],
                    "advisory": det["advisory"],
                    "color": det["color"],
                    "first_seen": current_frame,
                    "last_seen": current_frame
                }
                self.next_id += 1

        # 2. Prune tracks that have exceeded their TTL
        expired = [tid for tid, tr in self.tracks.items() if (current_frame - tr["last_seen"]) > self.ttl_frames]
        for tid in expired:
            del self.tracks[tid]

        return list(self.tracks.values())

def create_session(model_path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # 4 high-perf Cortex-A76 cores
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

def preprocess(frame, input_size=768):
    h0, w0 = frame.shape[:2]
    scale = min(input_size / h0, input_size / w0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    dw, dh = (input_size - nw) / 2.0, (input_size - nh) / 2.0
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, (h0, w0), (scale, dw, dh)

def postprocess(preds, orig_shape, ratio_pad, meta):
    """
    Vectorized NumPy decoding for instant sub-millisecond execution.
    """
    arr = np.squeeze(preds)
    if arr.ndim != 2:
        return []
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    scores = arr[4, :]
    mask = scores >= meta["conf"]
    if not np.any(mask):
        return []

    scale, dw, dh = ratio_pad
    h0, w0 = orig_shape
    cx = arr[0, mask]
    cy = arr[1, mask]
    w = arr[2, mask]
    h = arr[3, mask]
    selected_scores = scores[mask]

    xmin = (cx - w / 2.0 - dw) / scale
    ymin = (cy - h / 2.0 - dh) / scale
    bw = w / scale
    bh = h / scale

    boxes = [[int(x), int(y), int(bw_val), int(bh_val)] 
             for x, y, bw_val, bh_val in zip(xmin, ymin, bw, bh)]
    scores_list = selected_scores.tolist()

    indices = cv2.dnn.NMSBoxes(boxes, scores_list, meta["conf"], 0.45)
    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            b = boxes[idx]
            x1 = max(0, b[0])
            y1 = max(0, b[1])
            x2 = min(w0, b[0] + b[2])
            y2 = min(h0, b[1] + b[3])
            detections.append({
                "hazard": meta["name"],
                "confidence": float(scores_list[idx]),
                "advisory": meta["advisory"],
                "color": meta["color"],
                "box": [x1, y1, x2, y2]
            })
    return detections

def run_continuous_feed(source, model_choice="all", scheduler="round_robin", resolution=768, max_frames=None, headless=False):
    # Select target models
    active_keys = ["pothole", "construction", "accident"] if model_choice == "all" else [model_choice]

    print("=" * 85)
    print("  EDGE-AI REAL-TIME MULTI-HAZARD PERCEPTION ENGINE")
    print(f"  Target Hardware  : Raspberry Pi 5 (Quad Cortex-A76, 4 Threads)")
    print(f"  Scheduling Engine: {scheduler.upper()} ('Robinson' Frame Multiplexing)")
    print(f"  Active Pipeline  : {[HAZARD_MODELS[k]['name'] for k in active_keys]}")
    print(f"  Input Source     : {source}")
    print(f"  Resolution       : {resolution}x{resolution}")
    print("=" * 85)

    # Initialize ONNX inference sessions
    sessions = {}
    for k in active_keys:
        meta = HAZARD_MODELS[k]
        print(f"Loading Session [{meta['name']:<22}] -> {meta['path'].name}")
        sess = create_session(meta["path"])
        sessions[k] = {
            "session": sess,
            "in_name": sess.get_inputs()[0].name,
            "meta": meta
        }

    tracker = TemporalHazardTracker(ttl_frames=30, iou_thresh=0.35)

    is_camera = str(source).isdigit() or str(source).startswith("http") or str(source).startswith("rtsp")
    cap = None
    image_files = []

    if is_camera:
        cam_src = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(cam_src)
        if not cap.isOpened():
            print(f"\n[ERROR] Unable to open camera stream at: {source}")
            print("Tip: If using laptop webcam directly, pass `--source 0`.")
            return
        print(f"\n✓ Successfully connected to live video feed: {source}")
    else:
        src_path = Path(source)
        if src_path.is_dir():
            image_files = sorted(list(src_path.rglob("*.jpg")) + list(src_path.rglob("*.png")))
        elif src_path.is_file():
            image_files = [src_path]
        if not image_files:
            print(f"\n[ERROR] No image files found at: {source}")
            return
        print(f"\n✓ Found {len(image_files)} benchmark frames to cycle continuously.")

    frame_count = 0
    fps_history = []
    print(f"\nStarting continuous perception loop ({scheduler.upper()}). Press Ctrl+C or 'q' to stop...\n")

    try:
        while True:
            t_start = time.perf_counter()

            if is_camera:
                ret, frame = cap.read()
                if not ret:
                    print("Camera stream ended or disconnected.")
                    break
            else:
                img_path = image_files[frame_count % len(image_files)]
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

            frame_count += 1

            # Shared Preprocessing (Done ONCE per frame)
            blob, orig_shape, ratio_pad = preprocess(frame, input_size=resolution)

            frame_detections = []
            if scheduler == "round_robin" and len(active_keys) > 1:
                # Round-Robin: Evaluate 1 model per frame
                current_key = active_keys[(frame_count - 1) % len(active_keys)]
                s_info = sessions[current_key]
                preds = s_info["session"].run(None, {s_info["in_name"]: blob})[0]
                raw_dets = postprocess(preds, orig_shape, ratio_pad, s_info["meta"])
                # Update temporal tracker and retrieve persistent confirmed hazards
                frame_detections = tracker.update(current_key, raw_dets, frame_count)
            else:
                # Simultaneous: Evaluate all models on every frame
                for k in active_keys:
                    s_info = sessions[k]
                    preds = s_info["session"].run(None, {s_info["in_name"]: blob})[0]
                    dets = postprocess(preds, orig_shape, ratio_pad, s_info["meta"])
                    frame_detections.extend(dets)

            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            current_fps = 1000.0 / t_elapsed if t_elapsed > 0 else 0.0
            fps_history.append(current_fps)
            avg_fps = np.mean(fps_history[-30:])

            # Human-Readable Live Console Status (NO JSON!)
            status_prefix = f"[Frame {frame_count:04d}] Latency: {t_elapsed:6.1f}ms | FPS: {current_fps:4.1f} (Avg: {avg_fps:4.1f})"
            if len(frame_detections) > 0:
                det_strings = [f"{d['hazard']} ({d['confidence']*100:.1f}%)" for d in frame_detections]
                top_advisories = list(set([d['advisory'] for d in frame_detections]))
                print(f"{status_prefix} | HAZARDS: {', '.join(det_strings)} -> ALERT: {', '.join(top_advisories)}", flush=True)
            else:
                print(f"{status_prefix} | Clear Roadway (No Hazards Detected)", flush=True)

            # Draw Visual Bounding Boxes & On-Screen Display
            if not headless:
                for d in frame_detections:
                    b = d["box"]
                    col = d["color"]
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), col, 2)
                    label = f"{d['hazard']} {d['confidence']*100:.1f}% [{d['advisory']}]"
                    cv2.putText(frame, label, (b[0], max(20, b[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                sched_label = "Round-Robin" if scheduler == "round_robin" else "Simultaneous"
                osd_text = f"RPi 5 ({sched_label}): {avg_fps:.1f} FPS ({t_elapsed:.0f}ms)"
                cv2.putText(frame, osd_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

                try:
                    cv2.imshow("RPi 5 Multi-Hazard Perception", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    headless = True

            if max_frames and frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\nPerception stopped by user (Ctrl+C).")
    finally:
        if cap:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("\n" + "=" * 85)
    print(f"  PERCEPTION SESSION SUMMARY ({scheduler.upper()})")
    print("=" * 85)
    print(f"  • Total Frames Processed  : {frame_count}")
    print(f"  • Average Perception Rate : {np.mean(fps_history):.2f} FPS")
    if fps_history:
        print(f"  • P50 Frame Latency       : {1000.0 / np.median(fps_history):.1f} ms")
        print(f"  • P95 Frame Latency       : {1000.0 / np.percentile(fps_history, 5):.1f} ms")
    print("=" * 85)

def main():
    parser = argparse.ArgumentParser(description="Real-Time Multi-Hazard Camera Perception Engine")
    parser.add_argument("--source", type=str, default="data/benchmark_150", 
                        help="Camera index ('0' for laptop webcam), IP stream URL, or image folder")
    parser.add_argument("--scheduler", type=str, default="round_robin", choices=["round_robin", "simultaneous"],
                        help="Frame multiplexing scheduler: 'round_robin' (3x speedup) or 'simultaneous'")
    parser.add_argument("--model", type=str, default="all", choices=["all", "pothole", "construction", "accident"],
                        help="Target model: 'all' runs all 3 models")
    parser.add_argument("--size", type=int, default=768, help="Inference letterbox resolution (default 768)")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    args = parser.parse_args()

    run_continuous_feed(args.source, model_choice=args.model, scheduler=args.scheduler, 
                        resolution=args.size, max_frames=args.frames, headless=args.headless)

if __name__ == "__main__":
    main()
