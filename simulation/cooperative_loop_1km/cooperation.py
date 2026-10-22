"""Application-level RSU memory; all clocks and delivery assumptions are explicit."""
import math


class RouteDistances:
    """Reuse route-distance queries while odometer updates account for travel."""
    def __init__(self):
        self.values = {}

    def get(self, connection, vehicle_id, hazard, road, lane_pos_m, odometer_m):
        key = vehicle_id, hazard["id"]
        if road == hazard["edge"] and lane_pos_m < hazard["pos_m"]:
            distance = hazard["pos_m"] - lane_pos_m
        else:
            previous = self.values.get(key)
            if previous:
                if previous[0] < 0:
                    return previous[0]
                distance = previous[0] - (odometer_m - previous[1])
                if distance > 0:
                    return distance
            distance = connection.vehicle.getDrivingDistance(vehicle_id, hazard["edge"], hazard["pos_m"])
        self.values[key] = distance, odometer_m
        return distance


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Clock values must be finite nonnegative seconds")
    return value


def request_is_current(request, warning, now, odometer_m):
    """An old command must not be credited to a later lap or expired report."""
    return bool(request and warning and request["warning_id"] == warning["id"]
                and now < warning["expires_at_s"]
                and 0 <= now - request["requested_at_s"] <= 30
                and 0 <= odometer_m - request["odometer_m"] <= 160)


def observed_lane_response(request, warning, now, odometer_m, lane):
    return bool(request_is_current(request, warning, now, odometer_m)
                and now > request["requested_at_s"]
                and lane == request["requested_lane"] and lane != request["from_lane"])


class WarningBroker:
    def __init__(self, rsu_ids, delay_s=.2, ttl_s=300):
        self.rsu_ids = list(rsu_ids)
        if not self.rsu_ids or len(set(self.rsu_ids)) != len(self.rsu_ids):
            raise ValueError("At least one unique RSU ID is required")
        self.delay_s, self.ttl_s = _seconds(delay_s), _seconds(ttl_s)
        if not self.ttl_s:
            raise ValueError("TTL must be positive")
        self.events, self.seen, self.announced = {}, set(), set()

    def report(self, record, now):
        now = _seconds(now)
        if record.get("mode") != "real_onnx" or record.get("inference_executed") is not True or record.get("mock_fallback", False):
            raise ValueError("Only genuine ONNX prediction records can create reports")
        context = record["encounter"]
        captured = _seconds(context["capture_sim_s"])
        captured_wall = _seconds(context["capture_wall_s"])
        completed_wall = _seconds(record["completed_wall_s"])
        if captured > now or completed_wall < captured_wall:
            raise ValueError("Invalid encounter/completion order")
        model = record["model"]["source_model_id"]
        if model not in ("pothole", "construction", "accident_obstacle"):
            raise ValueError("Unknown specialist model")
        if not record["detections"]:
            return None
        key = context["id"] + ":" + model
        if key in self.seen:
            return None
        self.seen.add(key)
        available = max(now, captured + completed_wall - captured_wall) + self.delay_s
        event = {"id": key, "hazard_id": context["hazard_id"], "edge": context["edge"],
                 "hazard_kind": context.get("hazard_kind", context["hazard_id"]),
                 "lane_index": context["lane_index"], "pos_m": context["pos_m"],
                 "reporter_vehicle_id": context["vehicle_id"], "source_model_id": model,
                 "prediction_run_id": record["run_id"], "reported_at_s": now,
                 "capture_sim_s": captured, "available_at_s": available,
                 "expires_at_s": available + self.ttl_s, "rsu_ids": list(self.rsu_ids),
                 "confidence": max(d["confidence"] for d in record["detections"]),
                 "evidence_status": "SINGLE_IMAGE", "transport": "modeled_full_loop_coverage"}
        self.events[key] = event
        return event

    def known(self, now):
        now = _seconds(now)
        self.events = {key: event for key, event in self.events.items() if event["expires_at_s"] > now}
        return [dict(event) for event in self.events.values() if event["available_at_s"] <= now]

    def tick(self, now):
        ready = [event for event in self.known(now) if event["id"] not in self.announced]
        self.announced.update(event["id"] for event in ready)
        return ready
