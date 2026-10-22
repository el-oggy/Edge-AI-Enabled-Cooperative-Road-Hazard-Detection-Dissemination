"""Launch the native Foundry map. N1 is a visual traffic demo, without AI warnings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid


SCENE = Path(__file__).resolve().parent
ROOT = SCENE.parents[1]


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Duration must be a positive finite number")
    return number


def sumo_installation(headless=False):
    home = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"))
    name = "sumo" if headless else "sumo-gui"
    binary = home / "bin" / (name + (".exe" if os.name == "nt" else ""))
    if not binary.is_file():
        found = shutil.which(name)
        if not found:
            raise FileNotFoundError(f"{name} is missing. Install SUMO or set SUMO_HOME.")
        binary = Path(found)
        home = binary.parent.parent
    if not (home / "tools" / "traci").is_dir():
        raise FileNotFoundError(f"SUMO TraCI tools are missing under {home}")
    return binary.resolve(), home / "tools"


def validate_scene(scene=SCENE):
    required = ("foundry.net.xml", "foundry.rou.xml", "foundry.scenery.add.xml",
                "foundry.view.xml", "follow.view.xml", "inspection.view.xml",
                "foundry.sumocfg", "scene.manifest.json")
    for name in required:
        if not (scene / name).is_file():
            raise FileNotFoundError(f"Missing authored scene file: {scene / name}")
    import xml.etree.ElementTree as ET
    for name in ("foundry.scenery.add.xml", "foundry.rou.xml"):
        for node in ET.parse(scene / name).iter():
            image = node.get("imgFile")
            if image and not (scene / image).is_file():
                raise FileNotFoundError(f"Missing scene artwork: {scene / image}")


def input_hashes(scene=SCENE):
    paths = [p for p in scene.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts]
    return {p.relative_to(scene).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def command_for(binary, run_dir, duration):
    return [str(binary), "-c", str(SCENE / "foundry.sumocfg"),
            "--gui-settings-file", str(SCENE / "foundry.view.xml"),
            "--end", str(duration), "--start", "--quit-on-end", "true",
            "--window-size", "1440,960", "--log", str(run_dir / "sumo.log"),
            "--error-log", str(run_dir / "sumo-errors.log"),
            "--tripinfo-output", str(run_dir / "trips.xml"),
            "--statistic-output", str(run_dir / "statistics.xml")]


def run(args):
    validate_scene()
    binary, tools = sumo_installation(args.headless)
    sys.path.insert(0, str(tools))
    import traci

    run_dir = args.output.resolve() if args.output else ROOT / "results" / "foundry_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True, exist_ok=False)
    hashes = input_hashes()
    metadata = {"mode": "native_visual_demo", "phase": "N1", "ai_inference": False,
                "cooperative_warning_service": False, "physical_lane_obstruction": False,
                "note": "Off-road samples illustrate categories; they are not neural detections.",
                "binary": str(binary), "command": command_for(binary, run_dir, args.duration),
                "pacing": "unpaced headless" if args.headless and args.fast else "fast GUI with event-loop yields" if args.fast else "wall clock target 1x",
                "created_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": hashes}
    (run_dir / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    connection = None
    started = time.perf_counter()
    samples = []
    observed_vehicles = {}
    collision_count = teleport_count = 0
    signal_states = {}
    followed = None
    views = {}
    screenshot_files = []
    screenshot_contexts = []
    capture_jobs = []
    capture_job = None
    capture_stage = "prepare"
    camera_settle_steps = 0
    overview_bounds = (-60, -55, 975, 825)
    status = "completed"
    error = None
    try:
        with (run_dir / "process.log").open("w", encoding="utf-8") as process_log:
            label = "foundry_" + uuid.uuid4().hex
            traci.start(metadata["command"], label=label, stdout=process_log)
            connection = traci.getConnection(label)
            # shortcut: TraCI exposes no public call timeout; bound this installed client's socket waits.
            connection._socket.settimeout(30)
            metadata["sumo_version"] = list(connection.getVersion())
            (run_dir / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if not args.headless:
                # Give the native window time to create its OpenGL context.
                # Fast stepping must not starve the desktop event loop.
                time.sleep(1)
                view_ids = connection.gui.getIDList()
                if not view_ids:
                    raise RuntimeError(f"Native views did not load: {view_ids}")
                views["overview"] = view_ids[0]
                connection.gui.setSchema(views["overview"], "Foundry Daylight")
                connection.gui.setBoundary(views["overview"], *overview_bounds)
                if args.capture:
                    manifest = json.loads((SCENE / "scene.manifest.json").read_text(encoding="utf-8"))
                    bounds = manifest["inspection_court"]["boundary_m"]
                    bounds = [bounds[0] - 8, bounds[1] - 8, bounds[2] + 8, bounds[3] + 8]
                    capture_jobs = [{"name": "overview", "bounds": overview_bounds},
                                    {"name": "follow"}, {"name": "court", "bounds": bounds}]
                    details = list(manifest["samples"])
                    details.append({"category": "junction", "position_m": list(connection.junction.getPosition("D2"))})
                    for fixture in details:
                        x, y = fixture["position_m"]
                        radius = 45 if fixture["category"] == "junction" else 10
                        capture_jobs.append({"name": "detail_" + fixture["category"],
                                             "bounds": [x - radius, y - radius * .7, x + radius, y + radius * .7]})
            wall_origin = time.perf_counter()
            with (run_dir / "traffic.jsonl").open("w", encoding="utf-8") as traffic_log:
                tick = 0
                while connection.simulation.getTime() < args.duration - 1e-6:
                    connection.simulationStep()
                    tick += 1
                    now = connection.simulation.getTime()
                    vehicles = connection.vehicle.getIDList()
                    for vehicle in vehicles:
                        if vehicle not in observed_vehicles:
                            observed_vehicles[vehicle] = connection.vehicle.getTypeID(vehicle)
                    collision_count += connection.simulation.getCollidingVehiclesNumber()
                    teleport_count += connection.simulation.getStartingTeleportNumber()
                    if tick % 10 == 0:
                        tls = {tls_id: connection.trafficlight.getRedYellowGreenState(tls_id)
                               for tls_id in connection.trafficlight.getIDList()}
                        for tls_id, state in tls.items():
                            signal_states.setdefault(tls_id, set()).add(state)
                        speeds = [connection.vehicle.getSpeed(v) for v in vehicles]
                        row = {"sim_time_s": now, "active_vehicles": len(vehicles),
                               "moving_vehicles": sum(speed > 0.5 for speed in speeds),
                               "mean_speed_mps": sum(speeds) / len(speeds) if speeds else 0,
                               "arrived": connection.simulation.getArrivedNumber(),
                               "tls": tls, "followed_vehicle": followed}
                        traffic_log.write(json.dumps(row) + "\n")
                        samples.append(row)
                    if args.capture and now >= min(160, args.duration / 2):
                        view = views["overview"]
                        if capture_stage == "settle":
                            # Camera changes are applied by the native event loop.
                            # Wait through multiple repaint opportunities before
                            # reading the front buffer, not just a socket reply.
                            camera_settle_steps += 1
                            time.sleep(.05)
                            if camera_settle_steps >= 4:
                                capture_stage = "request"
                        elif capture_stage == "request":
                            path = run_dir / (capture_job["name"] + ".png")
                            connection.gui.screenshot(view, str(path))
                            screenshot_files.append(path)
                            screenshot_contexts.append({"name": capture_job["name"],
                                "requested_sim_time_s": now, "followed_vehicle": followed,
                                "boundary": list(connection.gui.getBoundary(view))})
                            capture_stage = "advance"
                            time.sleep(.15)
                        elif capture_stage == "advance":
                            # screenshot() renders on the next simulationStep. Keep
                            # the camera fixed until that step has completed.
                            capture_job = None
                            screenshot_contexts[-1]["completed_after_sim_step_s"] = now
                            capture_stage = "prepare"
                            if not capture_jobs:
                                connection.gui.setBoundary(view, *overview_bounds)
                        elif capture_jobs:
                            capture_job = capture_jobs.pop(0)
                            connection.gui.trackVehicle(view, "")
                            followed = None
                            if capture_job["name"] == "follow":
                                followed = next((v for v in vehicles if v.startswith("bus")), vehicles[0])
                                x, y = connection.vehicle.getPosition(followed)
                                connection.gui.setBoundary(view, x - 45, y - 30, x + 45, y + 30)
                                connection.gui.trackVehicle(view, followed)
                            else:
                                connection.gui.setBoundary(view, *capture_job["bounds"])
                            capture_stage = "settle"
                            camera_settle_steps = 0
                            time.sleep(.15)
                    if not args.fast:
                        time.sleep(max(0, wall_origin + now - time.perf_counter()))
                    elif not args.headless:
                        time.sleep(.002)
    except traci.exceptions.FatalTraCIError as exc:
        if not args.headless and connection is not None and connection._process.poll() == 0:
            status = "window_closed"
        else:
            status, error = "failed", str(exc)
    except Exception as exc:
        status, error = "failed", str(exc)
    finally:
        if connection is not None:
            try:
                connection.close(wait=False)
            except (traci.exceptions.FatalTraCIError, TimeoutError, OSError):
                pass
            import subprocess
            try:
                connection._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                connection._process.terminate()
                connection._process.wait(timeout=3)
    unchanged = hashes == input_hashes()
    process_exit_code = connection._process.returncode if connection is not None else None
    # The socket can close just before the process reports its normal exit.
    if not args.headless and process_exit_code == 0 and error == "Connection closed by SUMO.":
        status, error = "window_closed", None
    missing_captures = [str(p) for p in screenshot_files if not p.is_file() or p.stat().st_size == 0]
    blank_captures = []
    if args.capture:
        from PIL import Image
        for path in screenshot_files:
            if str(path) not in missing_captures:
                with Image.open(path) as frame:
                    if frame.convert("RGB").getbbox() is None:
                        blank_captures.append(str(path))
    if not unchanged:
        status, error = "failed", "Authored scene files changed during execution"
    if missing_captures:
        status, error = "failed", (error + "; " if error else "") + "Requested native screenshots were not produced"
    if blank_captures:
        status, error = "failed", (error + "; " if error else "") + "Native screenshots were blank"
    if args.capture and (capture_jobs or capture_job or not screenshot_files) and error is None:
        status, error = "failed", "Run ended before all requested native screenshots"
    result = {"status": status, "error": error, "mode": "native_visual_demo", "ai_inference": False,
              "sumo_exit_code": process_exit_code,
              "wall_elapsed_s": time.perf_counter() - started,
              "last_sample_sim_time_s": samples[-1]["sim_time_s"] if samples else 0,
              "active_vehicle_min": min((r["active_vehicles"] for r in samples), default=0),
              "active_vehicle_max": max((r["active_vehicles"] for r in samples), default=0),
              "vehicle_types_observed": {kind: sum(value == kind for value in observed_vehicles.values())
                                         for kind in sorted(set(observed_vehicles.values()))},
              "minimum_active_after_180s": min((r["active_vehicles"] for r in samples if r["sim_time_s"] >= 180), default=None),
              "minimum_moving_after_180s": min((r["moving_vehicles"] for r in samples if r["sim_time_s"] >= 180), default=None),
              "collision_vehicle_reports": collision_count, "starting_teleports": teleport_count,
              "tls_observed_states": {k: sorted(v) for k, v in signal_states.items()},
              "authored_files_unchanged": unchanged, "screenshots": [str(p) for p in screenshot_files],
              "missing_screenshots": missing_captures, "blank_screenshots": blank_captures,
              "screenshot_contexts": screenshot_contexts}
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if sys.stdout is not None:
        print(json.dumps({"output": str(run_dir), **result}, indent=2))
    if status == "failed":
        raise RuntimeError(f"{error}. See {run_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Verify traffic without the native window")
    parser.add_argument("--fast", action="store_true", help="Run without wall-clock pacing; not a real-time demonstration")
    parser.add_argument("--duration", type=positive_seconds, default=3600)
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are rejected")
    parser.add_argument("--capture", action="store_true", help="Save actual native overview/follow/court screenshots")
    args = parser.parse_args()
    if args.capture and args.headless:
        parser.error("Native screenshots require GUI mode")
    try:
        run(args)
    except Exception as exc:
        if sys.stderr is not None:
            print(str(exc), file=sys.stderr)
        elif os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(exc), "Foundry SUMO could not start", 0x10)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
