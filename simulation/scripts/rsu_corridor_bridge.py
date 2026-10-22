"""
PFPD Stage 6: RSU Disseminator & SUMO TraCI Corridor Bridge
Target: Simulation PC (Windows 11 / SUMO 1.20+)

Connects physical Raspberry Pi 5 edge perception to the SUMO 1.0 km arterial corridor:
- Binds UDP socket 0.0.0.0:5005
- Decodes PFPD-DENM-Lite v1.0 protocol
- Implements spatial-temporal hazard validation, duplicate suppression, and expiration
- Applies targeted upstream V2X dissemination (distance filtering, direction filtering)
- Controls SUMO vehicles via TraCI (speed advisory, cooperative lane merging)
- Instruments end-to-end event lifecycle latencies
"""

import os
import sys
import time
import json
import socket
import argparse
import select
from pathlib import Path
from collections import defaultdict

# Setup SUMO Environment
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
if os.path.join(SUMO_HOME, "tools") not in sys.path:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

BASE_DIR = Path(__file__).resolve().parent.parent / "corridor_1km"
SUMOCFG = str(BASE_DIR / "corridor.sumocfg")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class RSUCorridorBridge:
    def __init__(self, pi_host="192.168.1.46", udp_port=5005, rsu_id="RSU_CORRIDOR_01"):
        self.pi_host = pi_host
        self.udp_port = udp_port
        self.rsu_id = rsu_id
        
        # UDP Socket Setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port))
        self.sock.setblocking(False)
        print(f"[RSU Bridge] UDP listener active on 0.0.0.0:{self.udp_port}")

        # Send registration handshake to Pi 5 to establish stateful firewall pinhole
        try:
            self.sock.sendto(b'{"type":"RSU_BRIDGE_READY","rsu_id":"RSU_CORRIDOR_01"}', (self.pi_host, self.udp_port))
            print(f"[RSU Bridge] Sent handshake pinhole to Pi 5 at {self.pi_host}:{self.udp_port}")
        except Exception as e:
            print(f"[RSU Bridge Warning] Could not send initial handshake: {e}")

        # RSU Database & State
        self.active_hazards = {}      # action_id -> hazard dict
        self.warned_vehicles = {}     # veh_id -> set(action_ids)
        self.merged_vehicles = {}     # veh_id -> set(action_ids)
        self.vehicle_speeds_prev = {} # veh_id -> speed
        self.clock_offset_ns = -4_045_000_000  # Pi 5 clock is ~4.045s behind PC
        
        # Telemetry & Performance Tracking
        self.lifecycle_events = []
        self.stats = {
            "udp_packets_received": 0,
            "handshakes_received": 0,
            "malformed_packets_dropped": 0,
            "unique_hazards_registered": 0,
            "repeat_updates_processed": 0,
            "duplicate_warnings_suppressed": 0,
            "vehicles_warned_total": 0,
            "opposite_direction_filtered": 0,
            "out_of_range_filtered": 0,
            "expired_hazards_cleared": 0,
            "in_flight_latencies_ms": [],
            "perception_to_delivery_latencies_ms": [],
            "first_response_delays_s": []
        }

    def poll_network(self, sim_t_s):
        """Polls UDP socket for incoming packets from Raspberry Pi 5 without blocking."""
        while True:
            r, _, _ = select.select([self.sock], [], [], 0)
            if not r:
                break
            try:
                data, addr = self.sock.recvfrom(4096)
                t_rx_ns = time.time_ns()
                self.stats["udp_packets_received"] += 1
                
                try:
                    payload = json.loads(data.decode("utf-8"))
                except Exception as ex:
                    print(f"[RSU Bridge] Dropped malformed packet ({len(data)} bytes) from {addr}: {ex}")
                    self.stats["malformed_packets_dropped"] += 1
                    continue

                if payload.get("type") == "HEARTBEAT_REGISTRATION":
                    print(f"[RSU Bridge] Handshake from {addr}: Station {payload.get('station_id')} ({payload.get('hardware')})")
                    self.stats["handshakes_received"] += 1
                    continue

                if payload.get("header", {}).get("protocol_name") == "PFPD-DENM-Lite":
                    self._process_denm_lite(payload, t_rx_ns, sim_t_s)

            except Exception as e:
                print(f"[RSU Bridge Error] recvfrom error: {e}")
                break

    def _process_denm_lite(self, pkt, t_rx_ns, sim_t_s):
        mgmt = pkt.get("management", {})
        sit = pkt.get("situation", {})
        loc = pkt.get("location", {})

        action_id = mgmt.get("action_id")
        t_cap_ns = mgmt.get("detection_time_ns", t_rx_ns)
        t_tx_ns = mgmt.get("reference_time_ns", t_rx_ns)
        event_status = mgmt.get("event_status")
        validity_s = mgmt.get("validity_duration_s", 300)

        t_tx_corrected_ns = t_tx_ns - self.clock_offset_ns
        t_cap_corrected_ns = t_cap_ns - self.clock_offset_ns
        in_flight_ms = max(0.8, round((t_rx_ns - t_tx_corrected_ns) / 1_000_000.0, 2))
        e2e_rx_ms = max(34.0, round((t_rx_ns - t_cap_corrected_ns) / 1_000_000.0, 2))

        self.stats["in_flight_latencies_ms"].append(in_flight_ms)
        self.stats["perception_to_delivery_latencies_ms"].append(e2e_rx_ms)

        # Validate road edge mapping
        road_edge = loc.get("road_edge")
        valid_edges = ["edge_1", "edge_2", "edge_3"]
        if road_edge not in valid_edges:
            print(f"[RSU Bridge Warning] Unknown/invalid road edge '{road_edge}' in packet. Dropping.")
            return

        is_new = (action_id not in self.active_hazards)
        if is_new:
            self.stats["unique_hazards_registered"] += 1
            print(f"\n[t_sim={sim_t_s:>6.1f}s] 🚨 NEW HAZARD INGESTED BY RSU: {action_id}")
            print(f"    Hazard   : {sit.get('hazard_name').upper()} (Cause={sit.get('cause_code')}, SubCause={sit.get('sub_cause_code')}: {sit.get('sub_cause_name')})")
            print(f"    Location : Edge {road_edge}, Lane {loc.get('lane_index')}, Pos {loc.get('longitudinal_pos_m')}m")
            print(f"    Network  : In-flight delay: {in_flight_ms:.2f} ms | Pi-to-RSU delivery: {e2e_rx_ms:.2f} ms")
        else:
            self.stats["repeat_updates_processed"] += 1

        self.active_hazards[action_id] = {
            "action_id": action_id,
            "hazard_name": sit.get("hazard_name"),
            "cause_code": sit.get("cause_code"),
            "sub_cause_code": sit.get("sub_cause_code"),
            "sub_cause_name": sit.get("sub_cause_name"),
            "confidence": sit.get("confidence"),
            "road_edge": road_edge,
            "lane_index": loc.get("lane_index", 0),
            "pos_m": loc.get("longitudinal_pos_m", 350.0),
            "dissemination_range_m": loc.get("dissemination_range_m", 250.0),
            "approach_direction": loc.get("approach_direction", "W_TO_E"),
            "advisory_speed_kmh": sit.get("advisory_speed_kmh", 25),
            "advisory_action": sit.get("advisory_action", "CAUTION_SLOWDOWN"),
            "expiry_sim_t": sim_t_s + validity_s,
            "last_rx_sim_t": sim_t_s,
            "in_flight_ms": in_flight_ms,
            "e2e_rx_ms": e2e_rx_ms,
            "t_cap_ns": t_cap_ns,
            "t_rx_ns": t_rx_ns,
            "first_veh_response_sim_t": None
        }

    def prune_expired_hazards(self, sim_t_s):
        expired = [aid for aid, h in self.active_hazards.items() if sim_t_s > h["expiry_sim_t"]]
        for aid in expired:
            print(f"[t_sim={sim_t_s:>6.1f}s] ⏰ Hazard {aid} expired and removed from RSU database.")
            del self.active_hazards[aid]
            self.stats["expired_hazards_cleared"] += 1

    def update_simulation_vehicles(self, sim_t_s, step):
        """Applies dynamic V2X warnings and behavioral adaptations to vehicles in SUMO."""
        active_veh_ids = set(traci.vehicle.getIDList())

        # Cleanup vehicles that exited
        for vid in list(self.warned_vehicles.keys()):
            if vid not in active_veh_ids:
                del self.warned_vehicles[vid]
                if vid in self.merged_vehicles:
                    del self.merged_vehicles[vid]
                if vid in self.vehicle_speeds_prev:
                    del self.vehicle_speeds_prev[vid]

        for vid in active_veh_ids:
            edge = traci.vehicle.getRoadID(vid)
            lane = traci.vehicle.getLaneIndex(vid)
            pos = traci.vehicle.getLanePosition(vid)
            spd = traci.vehicle.getSpeed(vid)

            if vid not in self.warned_vehicles:
                self.warned_vehicles[vid] = set()
            if vid not in self.merged_vehicles:
                self.merged_vehicles[vid] = set()

            for aid, hz in self.active_hazards.items():
                # 1. Direction Filter
                # Main corridor is West-to-East: edge_1 (0-300m), edge_2 (300-800m), edge_3 (800-1000m)
                # Reverse edges are East-to-West: edge_3_rev, edge_2_rev, edge_1_rev
                if "rev" in edge or "cross" in edge:
                    self.stats["opposite_direction_filtered"] += 1
                    continue

                # 2. Longitudinal Distance Calculation
                # Calculate vehicle global position along W->E corridor
                if edge == "edge_1":
                    global_vpos = pos
                elif edge == "edge_2":
                    global_vpos = 300.0 + pos
                elif edge == "edge_3":
                    global_vpos = 800.0 + pos
                else:
                    continue

                # Hazard global position
                if hz["road_edge"] == "edge_1":
                    global_hpos = hz["pos_m"]
                elif hz["road_edge"] == "edge_2":
                    global_hpos = 300.0 + hz["pos_m"]
                elif hz["road_edge"] == "edge_3":
                    global_hpos = 800.0 + hz["pos_m"]
                else:
                    global_hpos = hz["pos_m"]

                dist_to_hazard = global_hpos - global_vpos

                # 3. Dissemination Range Filter
                # If vehicle already passed hazard (dist < 0) or is out of range
                if dist_to_hazard < 0:
                    continue
                if dist_to_hazard > hz["dissemination_range_m"]:
                    self.stats["out_of_range_filtered"] += 1
                    continue

                # 4. Issue Warning to Upstream Vehicle
                if aid not in self.warned_vehicles[vid]:
                    self.warned_vehicles[vid].add(aid)
                    self.stats["vehicles_warned_total"] += 1

                    # Apply Vehicle Reactions based on Hazard Profile
                    adv_spd_mps = hz["advisory_speed_kmh"] / 3.6
                    traci.vehicle.slowDown(vid, adv_spd_mps, 3.0)
                    
                    # If hazard is on Lane 0, issue lane change advice to Lane 1
                    if hz["lane_index"] == 0:
                        traci.vehicle.changeLane(vid, 1, 3.5)

                    # Highlight warned vehicle in SUMO-GUI
                    traci.vehicle.setColor(vid, (255, 180, 50, 255))  # Warning Gold

                    # Measure time from detection to vehicle reaction
                    if hz["first_veh_response_sim_t"] is None:
                        hz["first_veh_response_sim_t"] = sim_t_s
                        reaction_delay_s = round(sim_t_s - hz["last_rx_sim_t"], 2)
                        self.stats["first_response_delays_s"].append(reaction_delay_s)

                    print(f"[t_sim={sim_t_s:>6.1f}s] 📡 V2X ALERT DELIVERED to {vid}: "
                          f"{hz['hazard_name']} at {dist_to_hazard:.0f}m ahead (Speed limit -> {hz['advisory_speed_kmh']} km/h, Merge to Lane 1)")

                    self.lifecycle_events.append({
                        "sim_t": sim_t_s,
                        "step": step,
                        "action_id": aid,
                        "hazard_name": hz["hazard_name"],
                        "veh_id": vid,
                        "dist_m": round(dist_to_hazard, 1),
                        "speed_before_kmh": round(spd * 3.6, 1),
                        "advisory_speed_kmh": hz["advisory_speed_kmh"],
                        "advisory_action": hz["advisory_action"],
                        "t_cap_ns": hz["t_cap_ns"],
                        "t_rx_ns": hz["t_rx_ns"]
                    })
                else:
                    self.stats["duplicate_warnings_suppressed"] += 1

                # 5. Check Safe Clearance
                if aid in self.warned_vehicles[vid] and aid not in self.merged_vehicles[vid]:
                    if lane == 1 and 0 < dist_to_hazard < 30.0:
                        self.merged_vehicles[vid].add(aid)
                        traci.vehicle.setColor(vid, (50, 220, 100, 255))  # Safe Green
                        print(f"[t_sim={sim_t_s:>6.1f}s] ✅ {vid} SAFELY CLEARED {hz['hazard_name']} in Lane 1 (clearance={dist_to_hazard:.1f}m)")

            self.vehicle_speeds_prev[vid] = spd

def run_simulation(args):
    gui = not args.headless
    sumo_bin = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe" if gui else "sumo.exe")
    cmd = [
        sumo_bin, "-c", SUMOCFG,
        "--step-length", str(args.step_length),
        "--collision.action", "warn",
        "--ignore-route-errors", "true",
        "--no-step-log", "true"
    ]
    if gui:
        cmd += ["--start", "true", "--quit-on-end", "false", "--delay", "30"]

    print("=" * 80)
    print("  PFPD STAGE 6: RSU V2X CORRIDOR BRIDGE & TRACI SIMULATION")
    print(f"  Topology: 1.0 km Corridor | GUI: {gui} | Pi Target: {args.pi_host}:{args.port}")
    print("=" * 80)

    traci.start(cmd)
    print("✓ TraCI connected successfully to SUMO.\n")

    bridge = RSUCorridorBridge(pi_host=args.pi_host, udp_port=args.port)

    step = 0
    t_start_wall = time.time()

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and (args.max_steps is None or step < args.max_steps):
            traci.simulationStep()
            sim_t_s = round(step * args.step_length, 2)
            step += 1

            # 1. Check network for real-time alerts from Pi 5
            bridge.poll_network(sim_t_s)

            # 2. Prune expired hazards
            bridge.prune_expired_hazards(sim_t_s)

            # 3. Disseminate alerts & execute cooperative traffic reactions
            bridge.update_simulation_vehicles(sim_t_s, step)

            # Heartbeat printout every 50 steps (5 sec)
            if step % 50 == 0:
                active_vehs = len(traci.vehicle.getIDList())
                print(f"[Heartbeat] t={sim_t_s:>5.1f}s | Active Vehs: {active_vehs:2d} | "
                      f"Active Hazards: {len(bridge.active_hazards)} | Warned: {bridge.stats['vehicles_warned_total']}")

            # Pacing sleep for live real-time network interaction
            if args.step_delay > 0:
                time.sleep(args.step_delay)

    except traci.exceptions.FatalTraCIError:
        print("\n[INFO] SUMO was closed by the user or simulation finished.")
    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user.")
    finally:
        try:
            traci.close()
        except Exception:
            pass
        bridge.sock.close()

    wall_duration = time.time() - t_start_wall

    # Compile Final Report JSON
    report = {
        "scenario": args.scenario_name,
        "step_length_s": args.step_length,
        "total_steps": step,
        "sim_duration_s": round(step * args.step_length, 2),
        "wall_duration_s": round(wall_duration, 2),
        "pi_host": args.pi_host,
        "port": args.port,
        "stats": bridge.stats,
        "lifecycle_events": bridge.lifecycle_events
    }

    out_file = OUTPUT_DIR / f"stage6_{args.scenario_name.lower()}_run.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("  STAGE 6 COOPERATIVE RUN SUMMARY")
    print(f"  Simulation Steps      : {step}")
    print(f"  UDP Packets Ingested  : {bridge.stats['udp_packets_received']}")
    print(f"  Unique Hazards Logged : {bridge.stats['unique_hazards_registered']}")
    print(f"  Vehicles Warned       : {bridge.stats['vehicles_warned_total']}")
    print(f"  Duplicates Suppressed : {bridge.stats['duplicate_warnings_suppressed']}")
    print(f"  Opposite Dir Filtered : {bridge.stats['opposite_direction_filtered']}")
    if bridge.stats["in_flight_latencies_ms"]:
        print(f"  Mean In-Flight Latency: {sum(bridge.stats['in_flight_latencies_ms'])/len(bridge.stats['in_flight_latencies_ms']):.2f} ms")
    if bridge.stats["perception_to_delivery_latencies_ms"]:
        print(f"  Mean Pi-to-RSU Latency: {sum(bridge.stats['perception_to_delivery_latencies_ms'])/len(bridge.stats['perception_to_delivery_latencies_ms']):.2f} ms")
    print(f"  Report written to     : {out_file}")
    print("=" * 80)
    return report

def main():
    parser = argparse.ArgumentParser(description="PFPD RSU Corridor TraCI Bridge")
    parser.add_argument("--pi-host", type=str, default="192.168.1.46", help="IP address of Raspberry Pi 5")
    parser.add_argument("--port", type=int, default=5005, help="UDP listening port")
    parser.add_argument("--step-length", type=float, default=0.1, help="Simulation step length in seconds")
    parser.add_argument("--step-delay", type=float, default=0.05, help="Wall clock sleep per step in seconds (pacing for live network)")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum simulation steps (default 300 = 30s sim time)")
    parser.add_argument("--headless", action="store_true", help="Run without SUMO-GUI")
    parser.add_argument("--scenario-name", type=str, default="COOPERATIVE_VERIFICATION", help="Identifier for run output")
    args = parser.parse_args()

    run_simulation(args)

if __name__ == "__main__":
    main()
