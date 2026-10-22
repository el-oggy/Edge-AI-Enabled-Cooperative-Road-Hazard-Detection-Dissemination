"""Structural contract for the expanded native SUMO demonstration map."""

from collections import Counter
import json
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "simulation" / "cooperative_loop_1km"
KINDS = ("pothole", "construction", "accident_obstacle")

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCENE_DIR))
from cooperation import RouteDistances


def _xyz(text):
    parts = [float(value) for value in text.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid SUMO coordinate: {text!r}")
    return parts[0], parts[1], parts[2] if len(parts) > 2 else 0.0


def _project_polyline(point, shape):
    """Return nearest distance and cumulative along-shape offset."""
    px, py = point
    walked = 0.0
    best_distance, best_offset = float("inf"), None
    for (ax, ay, _), (bx, by, _) in zip(shape, shape[1:]):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        length = math.sqrt(length_sq)
        fraction = 0.0 if not length_sq else max(0.0, min(1.0,
            ((px - ax) * dx + (py - ay) * dy) / length_sq))
        qx, qy = ax + fraction * dx, ay + fraction * dy
        distance = math.hypot(px - qx, py - qy)
        if distance < best_distance:
            best_distance, best_offset = distance, walked + fraction * length
        walked += length
    return best_distance, best_offset


class ExpandedSceneContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((SCENE_DIR / "scene.json").read_text(encoding="utf-8"))
        cls.sources = json.loads((SCENE_DIR / "encounters.json").read_text(encoding="utf-8"))
        cls.net = ET.parse(SCENE_DIR / "loop.net.xml").getroot()
        cls.routes = ET.parse(SCENE_DIR / "loop.rou.xml").getroot()
        cls.scenery = ET.parse(SCENE_DIR / "loop.scenery.add.xml").getroot()
        cls.nodes = {node.get("id"): node for node in cls.net.findall("junction")}
        cls.edges = {edge.get("id"): edge for edge in cls.net.findall("edge")}
        cls.lanes = {lane.get("id"): lane for edge in cls.net.findall("edge")
                     for lane in edge.findall("lane")}
        cls.connections = {(item.get("from"), item.get("to"))
                           for item in cls.net.findall("connection")}

    def test_nominal_and_compiled_loop_lengths_are_about_five_kilometres(self):
        self.assertEqual(self.scene["nominal_loop_length_m"], 5000)
        self.assertGreaterEqual(self.scene["main_route_length_m"], 4500)
        self.assertLessEqual(self.scene["main_route_length_m"], 5500)

    def test_eight_rsus_and_nine_hazards_match_scene_and_scenery(self):
        rsu_ids = {rsu["id"] for rsu in self.scene["rsus"]}
        rsu_pois = {poi.get("id") for poi in self.scenery.findall("poi")
                    if poi.get("type") == "roadside_unit_fixture"}
        self.assertEqual(self.scene["counts"]["rsus"], 8)
        self.assertEqual(len(rsu_ids), 8)
        self.assertEqual(rsu_ids, rsu_pois)

        hazards = self.scene["hazards"]
        self.assertEqual(self.scene["counts"]["hazards"], 9)
        self.assertEqual(len(hazards), 9)
        self.assertEqual(len({hazard["id"] for hazard in hazards}), 9)
        self.assertEqual(Counter(hazard["kind"] for hazard in hazards),
                         Counter({kind: 3 for kind in KINDS}))
        for kind in KINDS:
            ids = {hazard["id"] for hazard in hazards if hazard["kind"] == kind}
            self.assertIn(kind, ids)
            self.assertIn(kind + "_02", ids)
            self.assertIn(kind + "_03", ids)

        hazard_pois = {poi.get("id") for poi in self.scenery.findall("poi")
                       if (poi.get("id") or "").startswith("hazard_")}
        self.assertEqual(hazard_pois, {"hazard_" + hazard["id"] for hazard in hazards})

    def test_all_hazards_project_to_their_compiled_lanes_and_positions(self):
        for hazard in self.scene["hazards"]:
            with self.subTest(hazard=hazard["id"]):
                edge = self.edges[hazard["edge"]]
                lane = self.lanes[hazard["lane_id"]]
                self.assertIn(lane, edge.findall("lane"))
                self.assertEqual(int(lane.get("index")), hazard["lane_index"])
                self.assertGreaterEqual(hazard["pos_m"], 0)
                self.assertLessEqual(hazard["pos_m"], float(lane.get("length")))
                shape = [_xyz(pair) for pair in lane.get("shape").split()]
                distance, projected = _project_polyline(hazard["position_m"], shape)
                self.assertLessEqual(distance, 0.05)
                self.assertAlmostEqual(projected, hazard["pos_m"], delta=0.05)

    def test_every_hazard_kind_has_a_mapped_image_source(self):
        sources = self.sources["sources"]
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertTrue(sources[kind])
                for image_path in sources[kind]:
                    self.assertTrue((ROOT / image_path).is_file(), image_path)

    def test_actor_count_and_all_compiled_route_pairs_match_scene(self):
        actors = self.routes.findall("vehicle")
        self.assertEqual(len(actors), 144)
        self.assertEqual(self.scene["counts"]["vehicles_total"], len(actors))
        counts = Counter(actor.get("type") for actor in actors)
        self.assertEqual(counts, Counter(self.scene["counts"]["vehicles_by_type"]))
        route_map = {route.get("id"): route.get("edges", "").split()
                     for route in self.routes.findall("route")}
        self.assertGreaterEqual(len(route_map), 4)
        self.assertEqual(len(route_map), len(self.routes.findall("route")))
        for actor in actors:
            self.assertIn(actor.get("route"), route_map)
        for route_id, edges in route_map.items():
            with self.subTest(route=route_id):
                self.assertGreaterEqual(len(edges), 2)
                for edge_id in edges:
                    self.assertIn(edge_id, self.edges)
                for pair in zip(edges, edges[1:]):
                    self.assertIn(pair, self.connections,
                                  f"missing compiled route connection {pair}")

    def test_viaduct_is_elevated_and_crosses_above_a_separate_lower_route(self):
        bridge = self.scene["bridge"]
        self.assertEqual(bridge["id"], "foundry_viaduct")
        self.assertEqual(bridge["upper_edges"], ["bridge_W_E", "bridge_E_W"])
        self.assertEqual(bridge["lower_edges"], ["underpass_J1_C2", "underpass_C2_J2",
                                                 "underpass_J2_C2", "underpass_C2_J1"])
        self.assertEqual(bridge["deck_height_m"], 8)
        self.assertEqual(bridge["crossing_position_m"], [800, 450])
        self.assertEqual(bridge["bounds_m"], [630, 420, 970, 480])

        upper = [self.edges[edge_id] for edge_id in bridge["upper_edges"]]
        lower = [self.edges[edge_id] for edge_id in bridge["lower_edges"]]
        upper_nodes = {node for edge in upper for node in (edge.get("from"), edge.get("to"))}
        lower_nodes = {node for edge in lower for node in (edge.get("from"), edge.get("to"))}
        self.assertNotIn("C2", upper_nodes)
        self.assertIn("C2", lower_nodes)
        self.assertGreaterEqual(float(self.nodes["B_W"].get("z", 0)), 8)
        self.assertGreaterEqual(float(self.nodes["B_E"].get("z", 0)), 8)
        self.assertAlmostEqual(float(self.nodes["C2"].get("z", 0)), 0, delta=0.1)
        self.assertAlmostEqual(float(self.nodes["C2"].get("x")), 800, delta=0.1)
        self.assertAlmostEqual(float(self.nodes["C2"].get("y")), 450, delta=0.1)

        for edge in upper:
            shapes = [lane.get("shape", "").split() for lane in edge.findall("lane")]
            self.assertTrue(shapes)
            for shape in shapes:
                self.assertTrue(shape)
                elevations = [_xyz(point)[2] for point in shape]
                self.assertGreaterEqual(min(elevations), 7.9)
        for edge in lower:
            shapes = [lane.get("shape", "").split() for lane in edge.findall("lane")]
            self.assertTrue(shapes)
            self.assertTrue(any(abs(_xyz(point)[2]) <= 0.1
                                for shape in shapes for point in shape))

        upper_ids = set(bridge["upper_edges"])
        lower_ids = set(bridge["lower_edges"])
        self.assertFalse(any((source in upper_ids and target in lower_ids)
                             or (source in lower_ids and target in upper_ids)
                             for source, target in self.connections))


class FakeVehicleAPI:
    def __init__(self, distances):
        self.distances = iter(distances)
        self.calls = []

    def getDrivingDistance(self, vehicle_id, edge, position):
        self.calls.append((vehicle_id, edge, position))
        return next(self.distances)


class RouteDistanceTests(unittest.TestCase):
    def setUp(self):
        self.vehicle_api = FakeVehicleAPI([100.0, 240.0, 75.0])
        self.connection = type("Connection", (), {"vehicle": self.vehicle_api})()
        self.cache = RouteDistances()
        self.hazard = {"id": "pothole", "edge": "edge_target", "pos_m": 50.0}

    def test_cached_distance_counts_down_without_repeated_traci_queries(self):
        first = self.cache.get(self.connection, "veh-1", self.hazard,
                               "edge_approach", 10.0, 1000.0)
        second = self.cache.get(self.connection, "veh-1", self.hazard,
                                "edge_approach", 10.0, 1025.0)
        self.assertEqual(first, 100.0)
        self.assertEqual(second, 75.0)
        self.assertEqual(len(self.vehicle_api.calls), 1)

    def test_target_edge_uses_exact_lane_geometry_instead_of_estimate(self):
        self.cache.get(self.connection, "veh-1", self.hazard,
                       "edge_approach", 10.0, 1000.0)
        exact = self.cache.get(self.connection, "veh-1", self.hazard,
                               "edge_target", 32.0, 1010.0)
        self.assertEqual(exact, 18.0)
        self.assertEqual(len(self.vehicle_api.calls), 1)

    def test_passed_or_expired_countdown_refreshes_for_next_visit(self):
        self.cache.get(self.connection, "veh-1", self.hazard,
                       "edge_approach", 10.0, 1000.0)
        refreshed = self.cache.get(self.connection, "veh-1", self.hazard,
                                   "edge_approach", 10.0, 1105.0)
        self.assertEqual(refreshed, 240.0)
        self.assertEqual(len(self.vehicle_api.calls), 2)
        next_lap = self.cache.get(self.connection, "veh-1", self.hazard,
                                  "edge_approach", 10.0, 1110.0)
        self.assertEqual(next_lap, 235.0)
        self.assertEqual(len(self.vehicle_api.calls), 2)


if __name__ == "__main__":
    unittest.main()
