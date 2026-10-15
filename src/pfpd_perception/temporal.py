"""IoU association of distinct processed frames, with explicit source/capture clocks."""

import math

from .model_registry import MODELS


def box_iou(box_a, box_b):
    intersection = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(
        0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TemporalHazardTracker:
    """Consecutive processed hits; synthetic tests are not real temporal validation.

    Callers must supply increasing integer frame IDs and capture times (seconds)
    per source/model. An update is one processed frame, including empty detections.
    A skipped frame ID is permitted; an excessive capture-time gap resets streaks.
    """

    def __init__(self, iou_thresh=0.40, min_hits=3, max_misses=5,
                 max_capture_gap_s=1.0, expiry_s=2.0):
        if (not 0 < iou_thresh <= 1
                or isinstance(min_hits, bool) or not isinstance(min_hits, int) or min_hits < 1
                or isinstance(max_misses, bool) or not isinstance(max_misses, int) or max_misses < 1
                or not math.isfinite(max_capture_gap_s) or max_capture_gap_s <= 0
                or not math.isfinite(expiry_s) or expiry_s <= 0):
            raise ValueError("Invalid temporal tracker thresholds")
        self.iou_thresh, self.min_hits, self.max_misses = iou_thresh, min_hits, max_misses
        self.max_capture_gap_s, self.expiry_s = max_capture_gap_s, expiry_s
        self.tracks, self.next_id, self.last_frames = [], 1, {}

    _calc_iou = staticmethod(box_iou)

    def update(self, raw_detections, *, frame_id, capture_time_s,
               source_id, source_model_id):
        key = (source_id, source_model_id)
        if (not isinstance(source_id, str) or not source_id.strip()
                or not isinstance(source_model_id, str) or not source_model_id.strip()
                or source_model_id not in MODELS
                or isinstance(frame_id, bool)
                or not isinstance(frame_id, int) or frame_id < 0
                or not math.isfinite(capture_time_s) or capture_time_s < 0):
            raise ValueError("Tracking requires source/model, integer frame ID and finite capture seconds")
        previous = self.last_frames.get(key)
        if previous and (frame_id <= previous[0] or capture_time_s <= previous[1]):
            raise ValueError("Repeated or out-of-order frame ID/capture time")
        detections = [dict(det) for det in raw_detections]
        for det in detections:
            if (isinstance(det.get("class_id"), bool) or not isinstance(det.get("class_id"), int)
                    or det["class_id"] != 0 or not isinstance(det.get("class_name"), str)
                    or det["class_name"] != MODELS[source_model_id]["name"]
                    or det.get("raw_class_id", 0) != 0):
                raise ValueError("Detection requires raw class 0 and a nonempty semantic class name")
            box, confidence = det["box"], det["confidence"]
            if (len(box) != 4 or not all(math.isfinite(value) for value in box)
                    or box[2] <= box[0] or box[3] <= box[1]
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1
                    or det.get("source_id", source_id) != source_id
                    or det.get("source_model_id", source_model_id) != source_model_id):
                raise ValueError("Invalid detection geometry, score or source identity")
        self.last_frames[key] = (frame_id, capture_time_s)
        self.tracks = [track for track in self.tracks
                       if track["key"] != key or capture_time_s - track["last_seen_s"] <= self.expiry_s]
        active = [track for track in self.tracks if track["key"] == key]
        if previous and capture_time_s - previous[1] > self.max_capture_gap_s:
            for track in active:
                track["hits"], track["evidence"] = 0, []
        pairs = []
        for d_index, det in enumerate(detections):
            for t_index, track in enumerate(active):
                if track["class_name"] == det["class_name"]:
                    overlap = box_iou(det["box"], track["box"])
                    if overlap >= self.iou_thresh:
                        pairs.append((-overlap, track["id"], d_index, t_index))
        matched_tracks, matched_detections = set(), set()
        for _, _, d_index, t_index in sorted(pairs):
            if t_index in matched_tracks or d_index in matched_detections:
                continue
            matched_tracks.add(t_index)
            matched_detections.add(d_index)
            track, det = active[t_index], detections[d_index]
            track["hits"] += 1
            track["misses"] = 0
            track["box"] = [0.3 * old + 0.7 * new for old, new in zip(track["box"], det["box"])]
            self._record_hit(track, det, frame_id, capture_time_s)
        for t_index, track in enumerate(active):
            if t_index not in matched_tracks:
                track["misses"] += 1
                track["hits"], track["evidence"] = 0, []
        self.tracks = [track for track in self.tracks if track["misses"] < self.max_misses]
        for d_index, det in enumerate(detections):
            if d_index not in matched_detections:
                track = {"id": self.next_id, "key": key, "box": list(det["box"]),
                         "class_id": det["class_id"], "class_name": det["class_name"],
                         "hits": 1, "misses": 0, "evidence": []}
                self.next_id += 1
                self._record_hit(track, det, frame_id, capture_time_s)
                self.tracks.append(track)
        return [self._confirmation(track) for track in self.tracks
                if track["key"] == key and track["last_frame_id"] == frame_id
                and track["misses"] == 0 and track["hits"] >= self.min_hits]

    def _record_hit(self, track, det, frame_id, capture_time_s):
        track["last_frame_id"], track["last_seen_s"] = frame_id, capture_time_s
        track["evidence"].append({"frame_id": frame_id, "capture_time_s": capture_time_s,
                                  "confidence": float(det["confidence"])})
        track["evidence"] = track["evidence"][-self.min_hits:]

    def _confirmation(self, track):
        return {"track_id": track["id"], "class_id": track["class_id"],
                "raw_class_id": track["class_id"], "class_name": track["class_name"],
                "source_id": track["key"][0], "source_model_id": track["key"][1],
                "confidence": min(hit["confidence"] for hit in track["evidence"]),
                "box": list(track["box"]), "status": "CONFIRMED_HAZARD",
                "persistence_hits": track["hits"],
                "evidence": [dict(hit) for hit in track["evidence"]]}
