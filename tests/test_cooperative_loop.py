"""Deterministic unit tests for verified real-model hazard brokering."""

from copy import deepcopy
from collections import Counter
import json
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.cooperative_loop_1km.cooperation import WarningBroker, request_is_current, observed_lane_response


def record(*, mode="real_onnx", confidence=0.8, run_id="run-1",
           encounter_id="enc-1", model_id="pothole", capture_sim_s=10.0,
           capture_wall_s=100.0, completed_wall_s=105.0):
    return {
        "mode": mode,
        "inference_executed": True,
        "mock_fallback": False,
        "model": {"source_model_id": model_id},
        "detections": [{"confidence": confidence, "class_id": 0,
                         "source_model_id": model_id}],
        "run_id": run_id,
        "encounter": {"id": encounter_id, "vehicle_id": "veh-7",
                      "hazard_id": "hz-7", "edge": "edge_2",
                      "lane_index": 1, "pos_m": 125.5,
                      "capture_sim_s": capture_sim_s,
                      "capture_wall_s": capture_wall_s},
        "completed_wall_s": completed_wall_s,
    }


class WarningBrokerTests(unittest.TestCase):
    def setUp(self):
        self.broker = WarningBroker(["rsu-west", "rsu-east"], delay_s=0.2, ttl_s=300)

    def test_empty_negative_report_does_not_block_later_positive(self):
        negative = record(confidence=0.0)
        negative["detections"] = []
        self.assertIsNone(self.broker.report(negative, now=10.0))
        event = self.broker.report(record(), now=10.0)
        self.assertIsNotNone(event)
        self.assertEqual(event["source_model_id"], "pothole")

    def test_fabricated_or_mock_mode_is_rejected(self):
        for mode in ("mock", "fixture", "real_onnx_mock_fallback"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.broker.report(record(mode=mode), now=10.0)

    def test_two_rsus_receive_one_event_after_inference_elapsed_plus_delay(self):
        event = self.broker.report(record(), now=10.0)
        self.assertEqual(event["available_at_s"], 15.2)
        self.assertEqual(event["expires_at_s"], 315.2)
        self.assertEqual(event["rsu_ids"], ["rsu-west", "rsu-east"])
        self.assertEqual(event["hazard_id"], "hz-7")
        self.assertEqual(event["edge"], "edge_2")
        self.assertEqual(event["lane_index"], 1)
        self.assertEqual(event["pos_m"], 125.5)
        self.assertEqual(event["reporter_vehicle_id"], "veh-7")
        self.assertEqual(event["reported_at_s"], 10.0)
        self.assertEqual(self.broker.tick(15.199), [])
        available = self.broker.tick(15.2)
        self.assertEqual([item["id"] for item in available], [event["id"]])
        self.assertEqual(self.broker.tick(15.2), [])
        self.assertEqual([item["id"] for item in self.broker.known(15.2)], [event["id"]])

    def test_late_completion_is_never_delivered_retroactively(self):
        event = self.broker.report(record(capture_sim_s=10.0, capture_wall_s=100.0,
                                         completed_wall_s=105.0), now=20.0)
        self.assertEqual(event["available_at_s"], 20.2)
        self.assertEqual(self.broker.tick(20.199), [])
        self.assertEqual([x["id"] for x in self.broker.tick(20.2)], [event["id"]])

    def test_duplicate_encounter_and_model_does_not_create_second_event(self):
        first = self.broker.report(record(), now=10.0)
        duplicate = self.broker.report(record(run_id="run-2"), now=11.0)
        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)
        self.assertEqual(len(self.broker.known(11.0)), 0)
        other_model = self.broker.report(record(model_id="construction"), now=11.0)
        self.assertIsNotNone(other_model)
        self.assertEqual({item["source_model_id"] for item in self.broker.known(16.2)},
                         {"pothole", "construction"})

    def test_expiry_starts_at_availability_and_is_exclusive(self):
        broker = WarningBroker(["rsu-1"], delay_s=0.2, ttl_s=5.0)
        event = broker.report(record(), now=10.0)
        self.assertEqual(event["expires_at_s"], 20.2)
        self.assertEqual([x["id"] for x in broker.known(20.199)], [event["id"]])
        self.assertEqual(broker.known(20.2), [])
        self.assertEqual(broker.tick(20.2), [])

    def test_invalid_clock_values_are_rejected_without_state_change(self):
        bad_now = [-1.0, math.nan, math.inf, -math.inf, True, "10"]
        for now in bad_now:
            with self.subTest(now=now):
                before = deepcopy(self.broker.__dict__)
                with self.assertRaises(ValueError):
                    self.broker.report(record(), now=now)
                self.assertEqual(self.broker.__dict__, before)
        for method in (self.broker.tick, self.broker.known):
            for now in (math.nan, math.inf, -1.0, True):
                with self.subTest(method=method.__name__, now=now), self.assertRaises(ValueError):
                    method(now)

    def test_invalid_capture_or_completion_clock_is_rejected(self):
        invalid = [
            record(capture_sim_s=math.nan),
            record(capture_wall_s=math.inf),
            record(completed_wall_s=math.nan),
            record(capture_wall_s=105.0, completed_wall_s=104.0),
        ]
        for item in invalid:
            with self.subTest(item=item["encounter"]), self.assertRaises(ValueError):
                self.broker.report(item, now=10.0)
        self.assertEqual(self.broker.known(10.0), [])


class ResponseAttributionTests(unittest.TestCase):
    def setUp(self):
        self.warning = {"id": "warning-1", "expires_at_s": 100}
        self.request = {"warning_id": "warning-1", "requested_at_s": 10,
                        "odometer_m": 200, "from_lane": 0, "requested_lane": 1}

    def test_command_alone_is_not_an_observed_response(self):
        self.assertFalse(observed_lane_response(self.request, self.warning, 10, 200, 1))
        self.assertFalse(observed_lane_response(self.request, self.warning, 11, 205, 0))
        self.assertTrue(observed_lane_response(self.request, self.warning, 11, 205, 1))

    def test_expired_or_replaced_warning_cannot_credit_a_pass(self):
        self.assertFalse(request_is_current(self.request, {"id": "warning-1", "expires_at_s": 11}, 11, 205))
        self.assertFalse(request_is_current(self.request, {"id": "warning-2", "expires_at_s": 100}, 11, 205))
        self.assertFalse(request_is_current(self.request, None, 11, 205))

    def test_later_lap_or_expired_lane_request_is_not_current(self):
        self.assertFalse(request_is_current(self.request, self.warning, 12, 1200))
        self.assertFalse(request_is_current(self.request, self.warning, 41, 205))
        self.assertTrue(request_is_current(self.request, self.warning, 11, 205))


SCENE_DIR = ROOT / "simulation" / "cooperative_loop_1km"


def _xy(text):
    x, y = text.split(",")[:2]
    return float(x), float(y)


def _project_polyline(point, shape):
    """Return nearest distance and cumulative along-shape offset in metres."""
    px, py = point
    walked = 0.0
    best_distance, best_offset = float("inf"), None
    for start, end in zip(shape, shape[1:]):
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        length = math.sqrt(length_sq)
        fraction = 0.0 if not length_sq else max(0.0, min(1.0,
            ((px - ax) * dx + (py - ay) * dy) / length_sq))
        qx, qy = ax + fraction * dx, ay + fraction * dy
        distance = math.hypot(px - qx, py - qy)
        if distance < best_distance:
            best_distance = distance
            best_offset = walked + fraction * length
        walked += length
    return best_distance, best_offset


class CompiledSceneContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((SCENE_DIR / "scene.json").read_text(encoding="utf-8"))
        cls.net = ET.parse(SCENE_DIR / "loop.net.xml").getroot()
        cls.routes = ET.parse(SCENE_DIR / "loop.rou.xml").getroot()
        cls.scenery = ET.parse(SCENE_DIR / "loop.scenery.add.xml").getroot()
        cls.edges = {edge.get("id"): edge for edge in cls.net.findall("edge")}
        cls.lanes = {lane.get("id"): lane for edge in cls.net.findall("edge")
                     for lane in edge.findall("lane")}
        cls.connections = {(item.get("from"), item.get("to"))
                           for item in cls.net.findall("connection")}

    def test_hazards_resolve_to_compiled_lanes_and_project_to_manifest_positions(self):
        pois = {item.get("id"): item for item in self.scenery.findall("poi")}
        for hazard in self.scene["hazards"]:
            with self.subTest(hazard=hazard["id"]):
                edge = self.edges[hazard["edge"]]
                lane_id = hazard["lane_id"]
                lane = self.lanes[lane_id]
                self.assertIn(lane, edge.findall("lane"))
                self.assertEqual(int(lane.get("index")), hazard["lane_index"])
                lane_length = float(lane.get("length"))
                self.assertGreaterEqual(hazard["pos_m"], 0)
                self.assertLessEqual(hazard["pos_m"], lane_length)

                shape = [_xy(pair) for pair in lane.get("shape").split()]
                distance, projected_offset = _project_polyline(hazard["position_m"], shape)
                self.assertLessEqual(distance, 0.05)
                self.assertAlmostEqual(projected_offset, hazard["pos_m"], delta=0.05)

                poi = pois["hazard_" + hazard["id"]]
                poi_distance, _ = _project_polyline((float(poi.get("x")), float(poi.get("y"))), shape)
                self.assertLessEqual(poi_distance, 0.05)
                self.assertAlmostEqual(float(poi.get("x")), hazard["position_m"][0], delta=0.05)
                self.assertAlmostEqual(float(poi.get("y")), hazard["position_m"][1], delta=0.05)

    def test_all_vehicle_counts_departures_and_route_connection_pairs_match(self):
        vehicles = self.routes.findall("vehicle")
        counts = Counter(vehicle.get("type") for vehicle in vehicles)
        self.assertEqual(len(vehicles), self.scene["counts"]["vehicles_total"])
        self.assertEqual(counts, Counter(self.scene["counts"]["vehicles_by_type"]))
        departures = sorted(float(vehicle.get("depart")) for vehicle in vehicles)
        expected = [i * self.scene["counts"]["depart_interval_s"] for i in range(len(vehicles))]
        self.assertEqual(departures, expected)
        self.assertEqual(departures[-1], self.scene["counts"]["depart_last_s"])

        route_map = {route.get("id"): route.get("edges").split()
                     for route in self.routes.findall("route")}
        self.assertEqual(len(route_map), len(self.routes.findall("route")))
        for vehicle in vehicles:
            self.assertIn(vehicle.get("route"), route_map)
        for route_id, edges in route_map.items():
            with self.subTest(route=route_id):
                self.assertGreaterEqual(len(edges), 2)
                for edge_id in edges:
                    self.assertIn(edge_id, self.edges)
                for pair in zip(edges, edges[1:]):
                    self.assertIn(pair, self.connections, f"missing compiled route connection {pair}")

    def test_artwork_tls_and_evidence_panel_hazards_are_valid(self):
        for filename in ("loop.rou.xml", "loop.scenery.add.xml"):
            root = ET.parse(SCENE_DIR / filename).getroot()
            for node in root.iter():
                image = node.get("imgFile")
                if image:
                    with self.subTest(file=filename, image=image):
                        self.assertTrue((SCENE_DIR / image).is_file())

        tls_ids = {item.get("id") for item in self.net.findall("tlLogic")}
        self.assertGreaterEqual(len(tls_ids), 2)
        rsu_pois = {item.get("id") for item in self.scenery.findall("poi")
                    if item.get("type") == "roadside_unit_fixture"}
        scene_rsus = {item["id"] for item in self.scene["rsus"]}
        self.assertGreaterEqual(self.scene["counts"]["rsus"], 8)
        self.assertEqual(self.scene["counts"]["rsus"], len(scene_rsus))
        self.assertEqual(scene_rsus, rsu_pois)
        panel = self.scene["evidence_panel"]["bounds_m"]
        x0, y0, x1, y1 = panel
        for hazard in self.scene["hazards"]:
            x, y = hazard["position_m"]
            with self.subTest(hazard=hazard["id"]):
                self.assertFalse(x0 <= x <= x1 and y0 <= y <= y1,
                                 f"evidence panel overlaps {hazard['id']} lane hazard")


if __name__ == "__main__":
    unittest.main()
