"""Native Foundry district driven by genuine ONNX predictions and modeled RSUs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET

SCENE = Path(__file__).resolve().parent
ROOT = SCENE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCENE))
from simulation.foundry_market.run_simulation import positive_seconds, sumo_installation, input_hashes
from src.pfpd_perception.live_inference import ThreeModelInference, edge
from cooperation import WarningBroker, RouteDistances, request_is_current, observed_lane_response
from presentation import EvidencePanel, NativeCaptures, frame_bounds


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_scene():
    for filename in ("loop.net.xml", "loop.rou.xml", "loop.scenery.add.xml", "loop.view.xml", "loop.sumocfg", "scene.json", "encounters.json"):
        if not (SCENE / filename).is_file():
            raise FileNotFoundError(f"Missing authored scene file: {SCENE / filename}")
    for filename in ("loop.rou.xml", "loop.scenery.add.xml"):
        for node in ET.parse(SCENE / filename).iter():
            image = node.get("imgFile")
            if image and not (SCENE / image).is_file():
                raise FileNotFoundError(f"Missing artwork: {SCENE / image}")
    scene = json.loads((SCENE / "scene.json").read_text())
    sources = json.loads((SCENE / "encounters.json").read_text())
    ids = [h["id"] for h in scene["hazards"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Hazard location IDs must be unique")
    for hazard in scene["hazards"]:
        if hazard.get("kind", hazard["id"]) not in sources["sources"]:
            raise ValueError(f"Unmapped hazard kind: {hazard['id']}")
    for paths in sources["sources"].values():
        for path in paths:
            if not (ROOT / path).is_file():
                raise FileNotFoundError(f"Missing road photograph: {ROOT / path}")
    return scene, sources


class InferenceWorker:
    def __init__(self, engine, output):
        self.engine, self.output = engine, output
        self.pending, self.results = queue.Queue(maxsize=1), queue.SimpleQueue()
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self._work, name="genuine_onnx", daemon=True)
        self.thread.start()

    def submit(self, context, image):
        try:
            self.pending.put_nowait((context, image))
            return True
        except queue.Full:
            return False

    def _work(self):
        while not self.stop_requested.is_set():
            task = self.pending.get()
            if task is None:
                return
            context, image = task
            try:
                if time.perf_counter() - context["capture_wall_s"] > 30:
                    self.results.put(("stale", context))
                else:
                    for record in self.engine.infer_image(image, context):
                        path = edge._save_evidence(record, self.output / "inference")
                        record["evidence_path"] = str(path)
                        self.results.put(("prediction", record))
            except Exception as exc:
                self.results.put(("error", {"encounter": context, "error": str(exc)}))
            finally:
                self.results.put(("finished", context))

    def close(self):
        self.stop_requested.set()
        while not self.pending.empty():
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
        self.pending.put_nowait(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Inference worker did not stop within 30 seconds")


def run_smoke(engine, scene, sources, output):
    results = []
    for hazard in scene["hazards"]:
        context = {"id": "smoke_" + hazard["id"], "vehicle_id": "smoke_only_no_vehicle",
                   "hazard_id": hazard["id"], "edge": hazard["edge"], "lane_index": hazard["lane_index"],
                   "pos_m": hazard["pos_m"], "capture_sim_s": 0, "capture_wall_s": time.perf_counter()}
        context["hazard_kind"] = hazard.get("kind", hazard["id"])
        for record in engine.infer_image(ROOT / sources["sources"][context["hazard_kind"]][0], context):
            path = edge._save_evidence(record, output / "inference")
            results.append({"location": hazard["id"], "model": record["model"]["source_model_id"],
                            "detections": record["detections"], "timings_ms": record["timings_ms"],
                            "evidence_path": str(path)})
    summary = {"status": "completed", "mode": "real_onnx_smoke", "inference_executed": True,
               "forward_count": len(results), "results": results}
    write_json(output / "summary.json", summary)
    return summary


def run(args):
    scene, sources = validate_scene()
    if args.scenario != "all":
        scene["hazards"] = [h for h in scene["hazards"] if h.get("kind", h["id"]) == args.scenario]
    output = args.output.resolve() if args.output else ROOT / "results" / "cooperative_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    before = input_hashes(SCENE)
    started = time.perf_counter()
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "traffic_only" if args.traffic_only else "real_onnx_integrated",
                "source_mode": sources["source_mode"], "single_image_evidence": True,
                "physical_radio": False, "modeled_radio_delay_s": .2,
                "radio_coverage": "entire demonstration loop, assumed",
                "scene": scene["scene"], "rsu_ids": [r["id"] for r in scene["rsus"]],
                "main_route_length_m": scene.get("main_route_length_m"),
                "distance_method": "TraCI query with odometer countdown; exact position on target edge",
                "input_sha256": before, "python": sys.version,
                "argv": sys.argv, "scenario": args.scenario}
    write_json(output / "inputs.json", metadata)
    engine = None
    if not args.traffic_only:
        engine = ThreeModelInference()
        metadata["models"] = engine.identities()
        write_json(output / "inputs.json", metadata)
    if args.smoke:
        return run_smoke(engine, scene, sources, output)
    binary, tools = sumo_installation(args.headless)
    sys.path.insert(0, str(tools))
    import traci
    from traci import constants as tc
    broker = WarningBroker([rsu["id"] for rsu in scene["rsus"]])
    distances = RouteDistances()
    command = [str(binary), "-c", str(SCENE / "loop.sumocfg"), "--end", str(args.duration),
               "--start", "--quit-on-end", "true", "--window-size", "1440,960",
               "--log", str(output / "sumo.log"), "--error-log", str(output / "sumo-errors.log"),
               "--statistic-output", str(output / "statistics.xml"),
               "--tripinfo-output", str(output / "trips.xml"), "--ignore-route-errors", "false"]
    metadata["command"] = command
    write_json(output / "inputs.json", metadata)
    connection, worker, captures, panel = None, None, None, None
    status, error, now = "completed", None, 0.0
    counts = Counter(reports=0, warnings=0, avoided=0, traversals=0, drops=0)
    model_counts, observed_types, signals = Counter(), {}, {}
    sampled, in_flight, attempted, source_index = [], set(), set(), Counter()
    received, acted, last_states, seen = {}, {}, {}, set()
    processed_hazards, last_panel_s = set(), -100.0
    bridge_seen = {"upper":set(), "lower":set()}
    bridge_elevations = {}
    collision_count = teleport_count = 0
    log = (output / "events.jsonl").open("w", encoding="utf-8")
    def event(kind, **fields):
        log.write(json.dumps({"kind": kind, "sim_time_s": now, **fields}, allow_nan=False) + "\n")
        log.flush()
    try:
        with (output / "process.log").open("w", encoding="utf-8") as process_log:
            label = "cooperative_" + uuid.uuid4().hex
            traci.start(command, label=label, stdout=process_log)
            connection = traci.getConnection(label)
            connection._socket.settimeout(30)
            metadata["sumo_version"] = list(connection.getVersion())
            write_json(output / "inputs.json", metadata)
            active_ids = {h["id"] for h in scene["hazards"]}
            for poi in connection.poi.getIDList():
                if poi.startswith("hazard_") and poi.removeprefix("hazard_") not in active_ids:
                    connection.poi.remove(poi)
            for polygon in connection.polygon.getIDList():
                if polygon.startswith("hazard_marker_") and polygon.removeprefix("hazard_marker_") not in active_ids:
                    connection.polygon.remove(polygon)
            for hazard in scene["hazards"]:
                if hazard["blocking"]:
                    vid = "obstruction_" + hazard["id"]
                    route = vid + "_route"
                    connection.route.add(route, [hazard["edge"]])
                    front = hazard["pos_m"] + hazard["length_m"] / 2
                    connection.vehicle.add(vid, route, typeID="car", depart="now",
                        departLane=str(hazard["lane_index"]), departPos=str(front), departSpeed="0")
                    connection.vehicle.setLength(vid, hazard["length_m"])
                    connection.vehicle.setWidth(vid, hazard["width_m"])
                    connection.vehicle.setColor(vid, (255, 255, 255, 0))
                    connection.vehicle.setStop(vid, hazard["edge"], pos=front,
                        laneIndex=hazard["lane_index"], duration=args.duration + 60)
                    event("physical_obstruction", vehicle_id=vid, hazard=hazard)
            if engine:
                worker = InferenceWorker(engine, output)
            if not args.headless:
                time.sleep(1)
                view = connection.gui.getIDList()[0]
                connection.gui.setSchema(view, "Foundry Daylight")
                frame_bounds(connection, view, scene["bounds"])
                if engine:
                    panel = EvidencePanel(output, scene["evidence_panel"], len(scene["rsus"]))
                    panel.render(connection, now, counts)
                if args.capture:
                    captures = NativeCaptures(connection, scene, output)
            wall_origin = time.perf_counter()
            tick = 0
            while now < args.duration - 1e-6:
                connection.simulationStep()
                now, tick = connection.simulation.getTime(), tick + 1
                if not args.headless and tick == 5:
                    frame_bounds(connection, view, scene["bounds"])
                collision_count += connection.simulation.getCollidingVehiclesNumber()
                teleport_count += connection.simulation.getStartingTeleportNumber()
                vehicles = [v for v in connection.vehicle.getIDList() if not v.startswith("obstruction_")]
                for vid in vehicles:
                    if vid not in seen:
                        seen.add(vid)
                        connection.vehicle.subscribe(vid, (tc.VAR_ROAD_ID, tc.VAR_LANE_INDEX,
                            tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_DISTANCE, tc.VAR_TYPE))
                        observed_types[vid] = connection.vehicle.getTypeID(vid)
                dirty_panel = False
                if worker:
                    while not worker.results.empty():
                        kind, value = worker.results.get_nowait()
                        if kind == "finished":
                            in_flight.discard(value["hazard_id"])
                            processed_hazards.add(value["hazard_id"])
                        elif kind == "error":
                            event("inference_error", **value)
                            raise RuntimeError("Real inference failed: " + value["error"])
                        elif kind == "stale":
                            counts["drops"] += 1
                            event("stale_input", encounter=value)
                        else:
                            model_counts[value["model"]["source_model_id"]] += 1
                            event("prediction", evidence_path=value["evidence_path"],
                                encounter=value["encounter"], model=value["model"]["source_model_id"],
                                detection_count=value["detection_count"], completed_wall_s=value["completed_wall_s"])
                            if now - value["encounter"]["capture_sim_s"] > 30:
                                counts["drops"] += 1
                                event("stale_result", prediction_run_id=value["run_id"])
                            else:
                                report = broker.report(value, now)
                                if report:
                                    counts["reports"] += 1
                                    event("vehicle_report", **report)
                                    reporter = report["reporter_vehicle_id"]
                                    if reporter in vehicles:
                                        connection.vehicle.setColor(reporter, (220, 137, 50, 255))
                            if panel:
                                panel.receive(value)
                                dirty_panel = True
                for broadcast in broker.tick(now):
                    event("rsu_broadcast", **broadcast)
                    dirty_panel = True
                known = broker.known(now)
                known_ids = {record["id"] for record in known}
                received = {key: warning for key, warning in received.items()
                            if warning["expires_at_s"] > now and warning["id"] in known_ids}
                by_hazard = {}
                for record in known:
                    by_hazard.setdefault(record["hazard_id"], record)
                if tick % 5 == 0:
                    subscriptions = connection.vehicle.getAllSubscriptionResults()
                    for vid in vehicles:
                        state = subscriptions.get(vid)
                        if not state:
                            continue
                        road, lane, pos = state[tc.VAR_ROAD_ID], state[tc.VAR_LANE_INDEX], state[tc.VAR_LANEPOSITION]
                        for level in ("upper","lower"):
                            if road in scene.get("bridge",{}).get(level+"_edges",[]):
                                bridge_seen[level].add(vid)
                                if level not in bridge_elevations:
                                    bridge_elevations[level] = connection.vehicle.getPosition3D(vid)[2]
                        prior = last_states.get(vid)
                        for hazard in scene["hazards"]:
                            hid = hazard["id"]
                            request = acted.get((vid, hid))
                            warning_received = received.get((vid, hid))
                            if request and not request_is_current(request, warning_received, now, state[tc.VAR_DISTANCE]):
                                acted.pop((vid, hid), None)
                                request = None
                            if request and not request["observed_response"] and observed_lane_response(
                                    request, warning_received, now, state[tc.VAR_DISTANCE], lane):
                                request["observed_response"] = True
                                event("lane_response_observed", vehicle_id=vid, hazard_id=hid,
                                      warning_id=request["warning_id"], lane_index=lane,
                                      requested_at_s=request["requested_at_s"])
                            previous = prior and prior["road"] == hazard["edge"]
                            if previous and road == hazard["edge"] and prior["pos"] < hazard["pos_m"] <= pos:
                                warning = received.get((vid, hid))
                                bypass = lane != hazard["lane_index"]
                                warned_response = bool(bypass and warning and request and request["observed_response"])
                                counts["avoided"] += int(warned_response)
                                counts["traversals"] += int(not bypass)
                                event("observed_pass", vehicle_id=vid, hazard_id=hid,
                                    lane_index=lane, bypass=bypass, warned_request_and_response_preceded=warned_response,
                                    warning_id=warning["id"] if warning else None,
                                    measurement="front crossed mapped hazard center; no damage physics",
                                    attribution="warning/request/response sequence; no counterfactual causality claim")
                                if warned_response:
                                    connection.vehicle.setColor(vid, (94, 176, 126, 255))
                                    connection.vehicle.setSpeed(vid, -1)
                                acted.pop((vid, hid), None)
                            if not engine:
                                continue
                            distance = distances.get(connection, vid, hazard, road, pos, state[tc.VAR_DISTANCE])
                            key = (vid, hid)
                            if distance <= 0 or distance > 40:
                                attempted.discard(key)
                            if not distance > 0:
                                continue
                            warning = by_hazard.get(hid)
                            if warning:
                                old = received.get((vid, hid))
                                if old is None or old["id"] != warning["id"]:
                                    received[(vid, hid)] = warning
                                    counts["warnings"] += 1
                                    event("vehicle_warning", vehicle_id=vid, hazard_id=hid,
                                          warning_id=warning["id"], distance_m=distance,
                                          available_at_s=warning["available_at_s"])
                                    connection.vehicle.setColor(vid, (226, 191, 80, 255))
                                if distance <= 80 and lane == hazard["lane_index"] and (vid, hid) not in acted:
                                    if not road.startswith(":") and connection.edge.getLaneNumber(road) > 1:
                                        connection.vehicle.changeLane(vid, 1, 30)
                                        connection.vehicle.slowDown(vid, 5.56, 2)
                                        acted[(vid, hid)] = {"warning_id": warning["id"],
                                            "requested_at_s": now, "odometer_m": state[tc.VAR_DISTANCE],
                                            "from_lane": lane, "requested_lane": 1, "observed_response": False}
                                        event("avoidance_request", vehicle_id=vid, hazard_id=hid,
                                              warning_id=warning["id"], distance_m=distance,
                                              requested_lane=1, speed_mps=5.56)
                            # One pending encounter preserves brief sensing opportunities;
                            # completion age still decides whether a result can issue a warning.
                            if 0 < distance <= 40 and not warning and hid not in in_flight and not worker.pending.full():
                                if key not in attempted:
                                    paths = sources["sources"][hazard.get("kind", hid)]
                                    context = {"id": uuid.uuid4().hex, "vehicle_id": vid, "hazard_id": hid,
                                        "edge": hazard["edge"], "lane_index": hazard["lane_index"],
                                        "pos_m": hazard["pos_m"], "capture_sim_s": now,
                                        "capture_wall_s": time.perf_counter(), "distance_m": distance}
                                    context.update({"actual_capture_edge": road, "actual_capture_lane": lane,
                                                    "actual_capture_lane_pos_m": pos,
                                                    "hazard_kind": hazard.get("kind", hid),
                                                    "sensing": "mapped photo; hazard may be visible from adjacent lane"})
                                    image_path = ROOT / paths[source_index[hid] % len(paths)]
                                    if worker.submit(context, image_path):
                                        attempted.add(key)
                                        source_index[hid] += 1
                                        in_flight.add(hid)
                                        event("sensor_capture", encounter=context, image=str(image_path))
                                    else:
                                        counts["drops"] += 1
                                        event("queue_full_drop", encounter=context)
                        last_states[vid] = {"road": road, "lane": lane, "pos": pos}
                    tls_states = {tls_id: connection.trafficlight.getRedYellowGreenState(tls_id)
                                  for tls_id in connection.trafficlight.getIDList()}
                    for tls_id, phase in tls_states.items():
                        signals.setdefault(tls_id, set()).add(phase)
                    speeds = [subscriptions[v][tc.VAR_SPEED] for v in vehicles if v in subscriptions]
                    sampled.append({"sim_time_s": now, "active_vehicles": len(vehicles),
                                    "moving_vehicles": sum(s > .5 for s in speeds), "tls": tls_states})
                    if tick % 50 == 0:
                        event("traffic_sample", **sampled[-1])
                    if panel and (dirty_panel or (now-last_panel_s >= 10 and known)):
                        panel.render(connection, now, counts)
                        last_panel_s = now
                elif panel and dirty_panel:
                    panel.render(connection, now, counts)
                    last_panel_s = now
                if captures:
                    if captures.stage == "idle" and now >= min(150, args.duration*.65) and (not engine or processed_hazards >= active_ids):
                        follower = next((v for v, hid in reversed(list(acted)) if v in vehicles), vehicles[0] if vehicles else None)
                        captures.start(follower)
                    captures.step(now)
                if args.fast:
                    if not args.headless:
                        time.sleep(.002)
                else:
                    time.sleep(max(0, wall_origin + now - time.perf_counter()))
    except traci.exceptions.FatalTraCIError as exc:
        status, error = "failed", str(exc)
    except Exception as exc:
        status, error = "failed", str(exc)
    finally:
        if connection:
            try:
                connection.close(wait=False)
            except (traci.exceptions.FatalTraCIError, OSError, TimeoutError):
                pass
            import subprocess
            try:
                connection._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                connection._process.terminate()
                connection._process.wait(timeout=3)
                status, error = "failed", "SUMO did not exit within 10 seconds after TraCI closed"
        if worker:
            try:
                worker.close()
            except Exception as exc:
                status, error = "failed", str(exc)
        log.close()
    exit_code = connection._process.returncode if connection else None
    if exit_code not in (None, 0) and status == "completed":
        status, error = "failed", f"SUMO exited with code {exit_code}"
    if not args.headless and exit_code == 0 and error == "Connection closed by SUMO.":
        status, error = "window_closed", None
    unchanged = before == input_hashes(SCENE)
    if not unchanged:
        status, error = "failed", "Authored simulation inputs changed during execution"
    if captures and status == "completed":
        try:
            captures.validate()
        except Exception as exc:
            status, error = "failed", str(exc)
    with (output / "traffic.jsonl").open("w", encoding="utf-8") as stream:
        for row in sampled:
            stream.write(json.dumps(row) + "\n")
    summary = {"status": status, "error": error, "mode": metadata["mode"],
        "source_mode": sources["source_mode"], "inference_executed": bool(sum(model_counts.values())),
        "model_forward_counts_consumed": dict(model_counts), "processed_hazards": sorted(processed_hazards),
        "scene":scene["scene"], "rsu_count":len(scene["rsus"]), "hazard_count":len(scene["hazards"]),
        "main_route_length_m":scene.get("main_route_length_m"),
        "bridge_observed_vehicle_counts":{level:len(ids) for level,ids in bridge_seen.items()},
        "bridge_observed_elevations_m":bridge_elevations,
        "sim_time_s": now, "wall_elapsed_s": time.perf_counter()-started,
        "counts": dict(counts), "sumo_exit_code": exit_code, "authored_files_unchanged": unchanged,
        "bypass_counter_definition": "fresh warning and request followed by observed lane response and bypass; not a counterfactual safety benefit",
        "active_vehicle_max": max((r["active_vehicles"] for r in sampled), default=0),
        "minimum_active_after_180s": min((r["active_vehicles"] for r in sampled if r["sim_time_s"] >=180), default=None),
        "minimum_moving_after_180s": min((r["moving_vehicles"] for r in sampled if r["sim_time_s"] >=180), default=None),
        "observed_types": dict(Counter(observed_types.values())),
        "tls_observed_states": {tls:sorted(states) for tls, states in signals.items()},
        "collision_vehicle_reports": collision_count, "starting_teleports": teleport_count,
        "screenshots": captures.files if captures else [], "screenshot_contexts": captures.contexts if captures else [],
        "limitations": ["Mapped still photographs; no SUMO camera or temporal video validation.",
            "LoRa delivery is modeled, not transmitted or range-qualified.",
            "Crossing a pothole is a measured route/lane traversal, not physical damage simulation."]}
    write_json(output / "summary.json", summary)
    if status == "failed":
        raise RuntimeError(f"{error}. Evidence: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=positive_seconds, default=3600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", choices=("all", "pothole", "construction", "accident_obstacle"), default="all")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--traffic-only", action="store_true", help="Explicit traffic verification without neural inference")
    parser.add_argument("--fast", action="store_true", help="Only for traffic-only verification; live inference uses wall-clock pacing")
    parser.add_argument("--smoke", action="store_true", help="Execute all three actual models on each scenario image without SUMO")
    args = parser.parse_args()
    if args.fast and not args.traffic_only:
        parser.error("--fast requires --traffic-only; live inference must preserve wall-clock pacing")
    if args.capture and (args.headless or args.smoke):
        parser.error("--capture requires native GUI simulation mode")
    if args.smoke and args.traffic_only:
        parser.error("--smoke cannot be combined with --traffic-only")
    try:
        result = run(args)
        if sys.stdout:
            print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        if sys.stderr:
            print(str(exc), file=sys.stderr)
        elif sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(exc), "Cooperative SUMO error", 0x10)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
