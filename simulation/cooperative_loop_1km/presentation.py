"""Native SUMO evidence artwork and deliberately settled viewport captures."""
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    path = Path(r"C:\Windows\Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def frame_bounds(connection, view, bounds):
    """Fit the rectangle and explicitly center it after native window resizing."""
    x0, y0, x1, y1 = bounds
    connection.gui.setBoundary(view, x0, y0, x1, y1)
    connection.gui.setOffset(view, (x0 + x1) / 2, (y0 + y1) / 2)
    (vx0, vy0), (vx1, vy1) = connection.gui.getBoundary(view)
    scale = min((vx1-vx0)/(x1-x0), (vy1-vy0)/(y1-y0))
    connection.gui.setZoom(view, connection.gui.getZoom(view) * scale * .98)


class EvidencePanel:
    def __init__(self, output, placement, rsu_count):
        self.output, self.placement = output, placement
        self.records, self.encounter_id, self.sequence = {}, None, 0
        self.source = None
        self.history = {}
        self.rsu_count = rsu_count

    def receive(self, record):
        hazard = record["encounter"]["hazard_id"]
        group = self.history.get(hazard)
        if not group or group["source"]["encounter"]["id"] != record["encounter"]["id"]:
            group = {"source": record, "records": {}}
            self.history[hazard] = group
        group["source"] = record
        group["records"][record["model"]["source_model_id"]] = record
        self.encounter_id = record["encounter"]["id"]
        self.source = record
        self.records = group["records"]

    def render(self, connection, now, counters):
        self.sequence += 1
        completed = [group for group in self.history.values() if len(group["records"]) == 3]
        if completed:
            chosen = completed[int(now // 10) % len(completed)]
            self.source, self.records = chosen["source"], chosen["records"]
        image = Image.new("RGBA", (1440, 1100), "#302F2A")
        draw = ImageDraw.Draw(image)
        cream, ochre = "#F0E2C5", "#D1A556"
        draw.rounded_rectangle((3, 3, 1436, 1096), radius=28, outline=ochre, width=6)
        draw.text((38, 28), "REAL ONNX  /  COOPERATIVE ROAD WARNINGS", font=font(41, True), fill=cream)
        draw.text((40, 88), "Mapped photographs  |  Actual results from this run  |  Single images", font=font(27), fill=ochre)
        if self.source:
            photo = Image.open(self.source["input"]["path"]).convert("RGB")
            overlay = ImageDraw.Draw(photo)
            colors = {"pothole": "#FFAF4C", "construction": "#F4DB55", "accident_obstacle": "#79D9C0"}
            for model, record in self.records.items():
                for det in record["detections"]:
                    overlay.rectangle(det["box"], outline=colors[model], width=3)
            photo.thumbnail((820, 470), Image.Resampling.LANCZOS)
            image.paste(photo, (40 + (820 - photo.width) // 2, 160 + (470 - photo.height) // 2))
            context = self.source["encounter"]
            lines = ["VEHICLE ENCOUNTER", context["vehicle_id"],
                     "Mapped location:", context["hazard_id"],
                     f"Capture: {context['capture_sim_s']:.1f} s", "Input: " + Path(self.source["input"]["path"]).name]
            for index, line in enumerate(lines):
                draw.text((900, 176 + index * 62), line, font=font(25, index == 0), fill=cream)
        else:
            draw.text((70, 320), "Waiting for a vehicle to reach a sensing zone...", font=font(34), fill=cream)
        for index, model in enumerate(("pothole", "construction", "accident_obstacle")):
            x, y = 38 + index * 460, 660
            draw.rounded_rectangle((x, y, x + 440, y + 150), radius=12, fill="#514B3E")
            draw.text((x + 18, y + 14), model.upper(), font=font(27, True), fill=ochre)
            record = self.records.get(model)
            if record:
                count = record["detection_count"]
                score = max((d["confidence"] for d in record["detections"]), default=0)
                draw.text((x + 18, y + 61), f"{count} boxes  |  max {score:.3f}", font=font(28), fill=cream)
                draw.text((x + 18, y + 105), f"{record['timings_ms']['inference']:.1f} ms actual inference", font=font(24), fill=cream)
            else:
                draw.text((x + 18, y + 73), "Waiting / processing", font=font(28), fill=cream)
        draw.text((40, 842), f"t={now:.1f}s   Reports {counters['reports']}   Warned {counters['warnings']}   Warned bypasses {counters['avoided']}", font=font(30, True), fill=cream)
        draw.text((40, 904), "Orange: reporter   Yellow: warning   Green: observed warned bypass", font=font(27), fill=ochre)
        draw.text((40, 959), f"{self.rsu_count} RSUs  |  Full-map coverage assumed  |  +200 ms modeled delivery", font=font(27), fill=cream)
        draw.text((40, 1014), "No physical LoRa transmission; map symbols are not CNN inputs.", font=font(25), fill=cream)
        path = self.output / f"panel_{self.sequence:04d}.png"
        image.save(path)
        p = self.placement
        if self.sequence == 1:
            connection.poi.add("inference_evidence", *p["position_m"], (255, 255, 255, 255),
                poiType="real_onnx_evidence", layer=40, imgFile=str(path),
                width=p["width_m"], height=p["height_m"])
        else:
            connection.poi.setImageFile("inference_evidence", str(path))
        return str(path)


class NativeCaptures:
    def __init__(self, connection, scene, output):
        self.connection, self.scene, self.output = connection, scene, output
        self.view = connection.gui.getIDList()[0]
        self.jobs, self.stage, self.steps, self.current = [], "idle", 0, None
        self.files, self.contexts = [], []

    def start(self, vehicle_id=None):
        self.jobs = [{"name": "overview", "bounds": self.scene["bounds"]}]
        if "bridge" in self.scene:
            self.jobs.append({"name": "bridge", "bounds": self.scene["bridge"]["bounds_m"]})
        if vehicle_id:
            self.jobs.append({"name": "follow", "vehicle": vehicle_id})
        for hazard in self.scene["hazards"]:
            x, y = hazard["position_m"]
            self.jobs.append({"name": hazard["id"], "bounds": [x-24, y-16, x+24, y+16]})
        x, y = self.scene["evidence_panel"]["position_m"]
        w, h = self.scene["evidence_panel"]["width_m"], self.scene["evidence_panel"]["height_m"]
        self.jobs.append({"name": "inference_panel", "bounds": [x-w/2-5, y-h/2-5, x+w/2+5, y+h/2+5]})
        self.stage = "prepare"

    def step(self, now):
        c, view = self.connection, self.view
        if self.stage == "prepare":
            if not self.jobs:
                c.gui.trackVehicle(view, "")
                frame_bounds(c, view, self.scene["bounds"])
                self.stage = "done"
                return
            self.current = self.jobs.pop(0)
            c.gui.trackVehicle(view, "")
            if "vehicle" in self.current and self.current["vehicle"] in c.vehicle.getIDList():
                x, y = c.vehicle.getPosition(self.current["vehicle"])
                frame_bounds(c, view, [x-28, y-14, x+28, y+14])
                c.gui.trackVehicle(view, self.current["vehicle"])
            else:
                frame_bounds(c, view, self.current.get("bounds", self.scene["bounds"]))
            self.stage, self.steps = "settle", 0
            time.sleep(.15)
        elif self.stage == "settle":
            self.steps += 1
            time.sleep(.05)
            if self.steps >= 4:
                self.stage = "request"
        elif self.stage == "request":
            if "bounds" in self.current:
                frame_bounds(c, view, self.current["bounds"])
            path = self.output / (self.current["name"] + ".png")
            c.gui.screenshot(view, str(path))
            self.files.append(str(path))
            self.contexts.append({"name": self.current["name"], "request_sim_s": now,
                                  "boundary": c.gui.getBoundary(view)})
            self.stage = "advance"
            time.sleep(.15)
        elif self.stage == "advance":
            self.contexts[-1]["completed_after_sim_step_s"] = now
            self.stage = "prepare"

    def validate(self):
        if self.stage != "done":
            raise RuntimeError("Run ended before all native captures completed")
        for filename in self.files:
            with Image.open(filename) as frame:
                if frame.convert("RGB").getbbox() is None:
                    raise RuntimeError(f"Black native capture: {filename}")
