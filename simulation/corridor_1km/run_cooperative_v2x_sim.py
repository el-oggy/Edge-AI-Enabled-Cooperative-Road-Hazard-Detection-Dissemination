"""
Edge-AI Cooperative Road Hazard Detection & V2X Dissemination — SUMO TraCI Engine
==================================================================================

Scenario:
  1.0 km arterial corridor with 2 signalized intersections (TL_01 at 300m, TL_02 at 800m).
  THREE distinct hazard zones on Lane 0:
    - CONSTRUCTION ZONE at x≈560m  (Model: construction_yolo11s_int8.onnx)
    - POTHOLE          at x≈650m  (Model: pothole_yolo11s_int8.onnx)
    - ACCIDENT/DEBRIS  at x≈731m  (Model: accident_obstacle_yolo11s_int8.onnx)

  Continuous vehicle flow. Each vehicle that enters the corridor acts as
  either a "detector" (first to see a hazard) or a "connected receiver"
  (warned by RSU). The RSU remembers all hazards reported and warns
  every subsequent vehicle.

  The SUMO-GUI stays open — you watch it, you close it when you want.

All 3 trained INT8 quantization models are represented in the simulation.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
if os.path.join(SUMO_HOME, "tools") not in sys.path:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

BASE_DIR = Path(__file__).resolve().parent
SUMOCFG = str(BASE_DIR / "corridor.sumocfg")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ── Hazard Definitions (mapped to our quantization models) ─────────────────
HAZARDS = [
    {
        "id": "CONSTRUCTION",
        "edge": "edge_2",
        "lane": 0,
        "pos": 250.0,        # pos along edge_2 → global x ≈ 550m
        "model": "construction_yolo11s_int8.onnx",
        "cause_code": 96,
        "sub_cause": 1,
        "description": "Road Construction / Barricade Zone",
        "severity": "HIGH",
        "advisory_action": "MERGE_ADJACENT_LANE",
        "advisory_speed_kmh": 25,
        "confidence": 0.927,
        "latency_ms": 36.1,
        "detect_range": 55.0,   # detect 55m before
        "color_detect": (255, 140, 0, 255),    # orange
        "color_warn": (255, 215, 0, 255),       # yellow
        "color_safe": (0, 255, 120, 255),        # green
    },
    {
        "id": "POTHOLE",
        "edge": "edge_2",
        "lane": 0,
        "pos": 350.0,        # pos along edge_2 → global x ≈ 650m
        "model": "pothole_yolo11s_int8.onnx",
        "cause_code": 97,
        "sub_cause": 1,
        "description": "Damaged Road Surface / Pothole",
        "severity": "HIGH",
        "advisory_action": "MANDATORY_LANE_CHANGE",
        "advisory_speed_kmh": 30,
        "confidence": 0.948,
        "latency_ms": 34.8,
        "detect_range": 50.0,
        "color_detect": (255, 100, 0, 255),
        "color_warn": (255, 200, 50, 255),
        "color_safe": (50, 220, 100, 255),
    },
    {
        "id": "ACCIDENT_OBSTACLE",
        "edge": "edge_2",
        "lane": 0,
        "pos": 430.0,        # pos along edge_2 → global x ≈ 730m
        "model": "accident_obstacle_yolo11s_int8.onnx",
        "cause_code": 95,
        "sub_cause": 2,
        "description": "Multi-Vehicle Collision / Road Debris",
        "severity": "CRITICAL",
        "advisory_action": "MANDATORY_LANE_CHANGE",
        "advisory_speed_kmh": 20,
        "confidence": 0.961,
        "latency_ms": 38.2,
        "detect_range": 60.0,
        "color_detect": (220, 30, 30, 255),
        "color_warn": (255, 180, 50, 255),
        "color_safe": (0, 200, 80, 255),
    },
]


def build_denm(station_id, hazard):
    """Build an ETSI EN 302 637-3 DENM packet for a hazard."""
    now_ms = int(time.time() * 1000)
    return {
        "header": {"protocolVersion": 2, "messageID": 1, "stationID": station_id},
        "denm": {
            "management": {
                "actionID": f"{station_id}_{hazard['id']}_{now_ms}",
                "detectionTime": now_ms,
                "eventPosition": {
                    "edgeID": hazard["edge"],
                    "laneIndex": hazard["lane"],
                    "longitudinalPosM": hazard["pos"],
                },
                "validityDurationSec": 300,
            },
            "situation": {
                "causeCode": hazard["cause_code"],
                "subCauseCode": hazard["sub_cause"],
                "description": hazard["description"],
                "severity": hazard["severity"],
                "advisoryAction": hazard["advisory_action"],
                "advisorySpeedKmh": hazard["advisory_speed_kmh"],
            },
        },
    }


def run(gui=True, max_steps=None, delay_ms=40):
    sumo_bin = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe" if gui else "sumo.exe")
    cmd = [sumo_bin, "-c", SUMOCFG, "--step-length", "0.1", "--collision.action", "warn"]

    if gui:
        # Do NOT auto-start, do NOT auto-quit — user controls everything
        # delay sets default playback speed so vehicles move visibly (can also be changed in GUI slider)
        cmd += ["--start", "false", "--quit-on-end", "false", "--delay", str(delay_ms)]

    print("=" * 70)
    print("  Edge-AI Cooperative Road Hazard Detection & V2X Dissemination")
    print("  Network : 1.0 km Signalized Corridor (2 Junctions, 2 Lanes)")
    print("  Hazards : Construction Zone (x≈550m) | Pothole (x≈650m) | Accident Debris (x≈730m)")
    print("  Models  : construction_int8 | pothole_int8 | accident_int8")
    print(f"  Mode    : {'SUMO-GUI — You control playback. Close when done.' if gui else 'Headless batch'}")
    print("=" * 70)

    traci.start(cmd)
    print("✓ TraCI connected. Press Play (▶) in SUMO-GUI to begin.\n")

    # ── RSU Memory: stores all known hazards, broadcasts to incoming vehicles
    rsu_known_hazards = {}        # hazard_id → DENM packet
    vehicle_states = {}           # veh_id → {detected: set(), warned: set(), merged_for: set()}
    detector_vehicles = set()     # first vehicles that detected each hazard
    telemetry = []
    prev_speeds = {}

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and (max_steps is None or step < max_steps):
            traci.simulationStep()
            sim_t = round(step * 0.1, 2)
            step += 1

            for vid in traci.vehicle.getIDList():
                edge = traci.vehicle.getRoadID(vid)
                lane = traci.vehicle.getLaneIndex(vid)
                pos = traci.vehicle.getLanePosition(vid)
                spd = traci.vehicle.getSpeed(vid)

                # Initialise tracking for new vehicles
                if vid not in vehicle_states:
                    vehicle_states[vid] = {"detected": set(), "warned": set(), "merged_for": set()}

                vs = vehicle_states[vid]

                # ── Check each hazard ────────────────────────────────────
                for hz in HAZARDS:
                    hid = hz["id"]
                    if edge != hz["edge"]:
                        continue

                    dist = hz["pos"] - pos

                    # ── DETECTION: vehicle on same lane, within detect range ──
                    if lane == hz["lane"] and 0 < dist <= hz["detect_range"] and hid not in vs["detected"]:
                        vs["detected"].add(hid)

                        # Is this the first vehicle to detect this hazard?
                        if hid not in rsu_known_hazards:
                            # First detector → Edge-AI inference
                            detector_vehicles.add(vid)
                            denm = build_denm("RSU_ZONE_01", hz)
                            rsu_known_hazards[hid] = denm

                            traci.vehicle.setColor(vid, hz["color_detect"])
                            traci.vehicle.changeLane(vid, 1, 3.5)

                            print(f"[t={sim_t:>6.1f}s] 🔍 {vid} DETECTED {hid} at pos={pos:.0f}m (dist={dist:.0f}m)")
                            print(f"           Model: {hz['model']} | Latency: {hz['latency_ms']}ms | Conf: {hz['confidence']:.1%}")
                            print(f"           → V2I uplink to RSU_ZONE_01. DENM broadcasted.")
                        else:
                            # Subsequent vehicle — RSU already knows, warn them
                            if hid not in vs["warned"]:
                                vs["warned"].add(hid)
                                traci.vehicle.setColor(vid, hz["color_warn"])
                                traci.vehicle.changeLane(vid, 1, 4.0)
                                traci.vehicle.slowDown(vid, hz["advisory_speed_kmh"] / 3.6, 3.0)
                                print(f"[t={sim_t:>6.1f}s] 📡 {vid} WARNED by RSU about {hid} at dist={dist:.0f}m — lane change + slow to {hz['advisory_speed_kmh']}km/h")

                    # ── Check if vehicle has safely merged past the hazard ──
                    if hid in vs["warned"] and hid not in vs["merged_for"] and lane == 1 and dist < 30.0 and dist > -20.0:
                        vs["merged_for"].add(hid)
                        traci.vehicle.setColor(vid, hz["color_safe"])
                        print(f"[t={sim_t:>6.1f}s] ✅ {vid} SAFELY PASSED {hid} in Lane 1 (clearance={dist:.0f}m)")

                # ── Telemetry logging (sample every 10 steps = 1s) ──
                if step % 10 == 0:
                    spd_kmh = round(spd * 3.6, 2)
                    prev_spd = prev_speeds.get(vid, spd)
                    accel = round((spd - prev_spd) / 0.1, 2) if step > 0 else 0.0
                    prev_speeds[vid] = spd

                    telemetry.append({
                        "step": step, "sim_time_s": sim_t, "veh_id": vid,
                        "edge_id": edge, "lane_idx": lane, "pos_m": round(pos, 1),
                        "speed_kmh": spd_kmh, "accel_mps2": accel,
                    })

    except traci.exceptions.FatalTraCIError:
        print("\n[INFO] SUMO-GUI was closed by the user.")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    # ── Save telemetry ──
    csv_path = OUTPUT_DIR / "v2x_telemetry.csv"
    if telemetry:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=telemetry[0].keys())
            w.writeheader()
            w.writerows(telemetry)
        print(f"\n✓ Telemetry saved: {csv_path} ({len(telemetry)} rows)")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total simulation steps : {step}")
    print(f"  Vehicles tracked       : {len(vehicle_states)}")
    print(f"  Hazards stored in RSU  : {len(rsu_known_hazards)}")
    for hid, denm in rsu_known_hazards.items():
        hz_info = denm["denm"]["situation"]
        print(f"    • {hid}: CauseCode={hz_info['causeCode']} | {hz_info['description']}")
    print(f"  First detectors        : {', '.join(detector_vehicles) if detector_vehicles else 'None'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO V2X Cooperative Hazard Simulation")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--gui", action="store_true", default=True, help="Run with SUMO-GUI (default)")
    parser.add_argument("--steps", type=int, default=None, help="Maximum simulation steps (default: runs till user closes GUI)")
    parser.add_argument("--delay", type=int, default=40, help="SUMO-GUI animation step delay in ms (default: 40ms)")
    args = parser.parse_args()

    use_gui = not args.headless
    run(gui=use_gui, max_steps=args.steps, delay_ms=args.delay)
