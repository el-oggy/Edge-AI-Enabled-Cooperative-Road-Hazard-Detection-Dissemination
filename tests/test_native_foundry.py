"""N1 regressions: portable launch, immutable assets and valid traffic geometry."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "simulation" / "foundry_market"
spec = importlib.util.spec_from_file_location("native_foundry", SCENE / "run_simulation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeFoundryTests(unittest.TestCase):
    def test_scene_complete_and_art_resolved(self):
        runner.validate_scene()

    def test_missing_art_fails_instead_of_rebuilding(self):
        with tempfile.TemporaryDirectory(prefix="tmp_native_", dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("foundry.net.xml", "foundry.rou.xml", "foundry.view.xml", "follow.view.xml", "inspection.view.xml", "foundry.sumocfg", "scene.manifest.json"):
                (folder / name).write_text("<routes/>" if name == "foundry.rou.xml" else "unchanged", encoding="utf-8")
            (folder / "foundry.scenery.add.xml").write_text('<additional><poi imgFile="missing.png"/></additional>', encoding="utf-8")
            before = runner.input_hashes(folder)
            with self.assertRaisesRegex(FileNotFoundError, "artwork"):
                runner.validate_scene(folder)
            self.assertEqual(before, runner.input_hashes(folder))

    def test_missing_vehicle_art_is_also_rejected(self):
        with tempfile.TemporaryDirectory(prefix="tmp_native_", dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("foundry.net.xml", "foundry.view.xml", "follow.view.xml", "inspection.view.xml", "foundry.sumocfg", "scene.manifest.json"):
                (folder / name).write_text("unchanged", encoding="utf-8")
            (folder / "foundry.scenery.add.xml").write_text("<additional/>", encoding="utf-8")
            (folder / "foundry.rou.xml").write_text('<routes><vType imgFile="missing.png"/></routes>', encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "artwork"):
                runner.validate_scene(folder)

    def test_command_uses_absolute_scene_and_run_paths(self):
        output = ROOT / "results" / "example"
        command = runner.command_for(Path("C:/SUMO/bin/sumo-gui.exe"), output, 120)
        self.assertEqual(Path(command[command.index("-c") + 1]), SCENE / "foundry.sumocfg")
        self.assertTrue(Path(command[command.index("--error-log") + 1]).is_absolute())
        self.assertNotIn("--no-warnings", command)

    def test_duration_rejects_nonfinite_or_nonpositive(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                runner.positive_seconds(value)

    def test_required_sample_categories_remain_visual_only(self):
        manifest = json.loads((SCENE / "scene.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({s["category"] for s in manifest["samples"]}, {
            "pothole", "puddle", "manhole", "asphalt_patch", "fallen_tree",
            "debris", "incident_car", "construction"})
        additional = ET.parse(SCENE / "foundry.scenery.add.xml").getroot()
        self.assertFalse(additional.findall("rerouter"))
        self.assertFalse(additional.findall("variableSpeedSign"))
        self.assertEqual(len(additional.findall("poi[@id='sample-pothole']")), 1)

    def test_images_decode_and_preserve_transparency(self):
        from PIL import Image
        for path in (SCENE / "assets").glob("*.png"):
            with Image.open(path) as im:
                self.assertEqual(im.mode, "RGBA", path.name)
                im.verify()

    def test_district_signs_do_not_cover_inspection_samples(self):
        manifest = json.loads((SCENE / "scene.manifest.json").read_text(encoding="utf-8"))
        x0, y0, x1, y1 = manifest["inspection_court"]["boundary_m"]
        for poi in ET.parse(SCENE / "foundry.scenery.add.xml").getroot().findall("poi[@type='district-sign']"):
            x, y, w, h = (float(poi.get(key)) for key in ("x", "y", "width", "height"))
            overlaps = x - w / 2 < x1 and x + w / 2 > x0 and y - h / 2 < y1 and y + h / 2 > y0
            self.assertFalse(overlaps, poi.get("id"))

    def test_every_route_connection_including_loop_is_declared(self):
        net = ET.parse(SCENE / "foundry.net.xml").getroot()
        links = {(node.get("from"), node.get("to")) for node in net.findall("connection")}
        routes = ET.parse(SCENE / "foundry.rou.xml").getroot()
        for route in routes.findall("route"):
            edges = route.get("edges").split()
            for pair in list(zip(edges, edges[1:])) + [(edges[-1], edges[0])]:
                self.assertIn(pair, links, (route.get("id"), pair))
        vehicles = routes.findall("vehicle")
        self.assertEqual(len(vehicles), 72)
        self.assertEqual(sorted(float(v.get("depart")) for v in vehicles), list(range(0, 144, 2)))


if __name__ == "__main__":
    unittest.main()
