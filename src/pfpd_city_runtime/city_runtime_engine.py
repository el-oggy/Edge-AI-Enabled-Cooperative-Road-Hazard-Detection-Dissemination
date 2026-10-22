"""
PFPD Smart City Runtime Engine (Phase SC1)
TraCI simulation controller managing 4-way signalized SUMO junction,
synchronizing fixed-camera perception, and orchestrating RSU advisory broadcasts.
"""

import os
import sys
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Ensure SUMO tools are available in path
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
if os.path.exists(SUMO_TOOLS) and SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

import traci

from pfpd_city_protocol.contracts import (
    HazardDetection,
    VehicleState,
    TrafficLightState,
    RSUBeacon,
    SimulationSnapshot
)
from pfpd_city_perception.camera_perception import CityCameraPerception

@dataclass
class RuntimeConfig:
    sumocfg_path: Path
    use_gui: bool = False
    step_length_s: float = 0.1
    rsu_id: str = "RSU_J0"
    rsu_pos: Tuple[float, float] = (0.0, 0.0)
    rsu_range_m: float = 250.0
    tls_id: str = "TL_J0"
    advisory_speed_mps: float = 5.55  # 20 km/h slowdown for hazards


class CityRuntimeEngine:
    """
    Sole TraCI coordinator for Phase SC1 4-way junction simulation.
    Orchestrates continuous multi-class traffic, traffic signals, camera perception,
    and vehicle cooperative slowdown / lane change responses.
    """

    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        perception: Optional[CityCameraPerception] = None
    ):
        if config is None:
            repo_root = Path(__file__).resolve().parents[2]
            sumocfg = repo_root / "simulation" / "smart_city" / "smart_city_junction.sumocfg"
            config = RuntimeConfig(sumocfg_path=sumocfg)

        self.config = config
        self.perception = perception or CityCameraPerception()
        
        self.is_running = False
        self.current_tick = 0
        self.sim_time_s = 0.0
        
        # State stores
        self.active_hazards: List[HazardDetection] = []
        self.snapshots: List[SimulationSnapshot] = []
        self.advisory_log: List[Dict[str, Any]] = []
        self.vehicles_warned: set = set()

    def start(self) -> None:
        """Launches SUMO TraCI server and establishes controller connection."""
        if self.is_running:
            return

        sumo_binary = "sumo-gui" if self.config.use_gui else "sumo"
        # Check standard installation path if not in global PATH
        sumo_exec = os.path.join(SUMO_HOME, "bin", f"{sumo_binary}.exe")
        if not os.path.exists(sumo_exec):
            sumo_exec = sumo_binary

        sumo_cmd = [
            sumo_exec,
            "-c", str(self.config.sumocfg_path),
            "--step-length", str(self.config.step_length_s),
            "--start",
            "--quit-on-end",
            "--no-warnings", "true"
        ]

        traci.start(sumo_cmd)
        self.is_running = True
        self.current_tick = 0
        self.sim_time_s = 0.0
        self.vehicles_warned.clear()

    def step(self) -> SimulationSnapshot:
        """Advances simulation by one step, collects telemetry, and processes RSU dissemination."""
        if not self.is_running:
            raise RuntimeError("Simulation engine is not running. Call start() first.")

        traci.simulationStep()
        self.current_tick += 1
        self.sim_time_s = traci.simulation.getTime()

        # 1. Collect Vehicle Telemetry
        vehicle_states: List[VehicleState] = []
        veh_ids = traci.vehicle.getIDList()

        for vid in veh_ids:
            x, y = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            heading = traci.vehicle.getAngle(vid)
            lane_id = traci.vehicle.getLaneID(vid)
            vtype = traci.vehicle.getTypeID(vid)

            # Check communication range to RSU_J0
            dist_to_rsu = math.hypot(x - self.config.rsu_pos[0], y - self.config.rsu_pos[1])
            applied_advisories = []

            # 2. Cooperative Dissemination & Vehicle Response
            if dist_to_rsu <= self.config.rsu_range_m and self.active_hazards:
                for hazard in self.active_hazards:
                    applied_advisories.append(hazard.advisory)
                    if vid not in self.vehicles_warned:
                        self._apply_cooperative_response(vid, hazard, lane_id)
                        self.vehicles_warned.add(vid)

            vehicle_states.append(VehicleState(
                vehicle_id=vid,
                x=float(x),
                y=float(y),
                speed_mps=float(speed),
                heading_deg=float(heading),
                lane_id=lane_id,
                vehicle_type=vtype,
                active_advisories=applied_advisories
            ))

        # 3. Collect Traffic Light State
        tls_states: List[TrafficLightState] = []
        try:
            current_state = traci.trafficlight.getRedYellowGreenState(self.config.tls_id)
            current_phase = traci.trafficlight.getPhase(self.config.tls_id)
            duration = traci.trafficlight.getPhaseDuration(self.config.tls_id)
            tls_states.append(TrafficLightState(
                tls_id=self.config.tls_id,
                phase_index=int(current_phase),
                state_string=current_state,
                duration_s=float(duration)
            ))
        except traci.TraCIException:
            pass

        # 4. Create RSU Beacon
        rsu_beacon = RSUBeacon(
            rsu_id=self.config.rsu_id,
            position_xy=self.config.rsu_pos,
            range_m=self.config.rsu_range_m,
            active_hazards=list(self.active_hazards),
            timestamp=self.sim_time_s
        )

        snapshot = SimulationSnapshot(
            tick=self.current_tick,
            sim_time_s=self.sim_time_s,
            vehicles=vehicle_states,
            traffic_lights=tls_states,
            rsus=[rsu_beacon]
        )
        self.snapshots.append(snapshot)
        return snapshot

    def _apply_cooperative_response(self, vid: str, hazard: HazardDetection, current_lane: str) -> None:
        """Applies realistic deceleration and lane diversion without bypassing SUMO safety."""
        action = hazard.advisory
        try:
            if action == "REDUCE_SPEED":
                # Gradual safe slowdown to advisory speed over 2.5s
                traci.vehicle.slowDown(vid, self.config.advisory_speed_mps, duration=2.5)
                self.advisory_log.append({
                    "vehicle_id": vid,
                    "tick": self.current_tick,
                    "sim_time": self.sim_time_s,
                    "action": "SLOWDOWN",
                    "target_speed": self.config.advisory_speed_mps,
                    "hazard_id": hazard.hazard_id
                })
            elif action in ("MERGE_LANE", "MANDATORY_LANE_CHANGE"):
                # If vehicle is on the inner lane (index 1), divert to outer lane (index 0)
                lane_index = traci.vehicle.getLaneIndex(vid)
                target_lane = 0 if lane_index > 0 else 1
                traci.vehicle.changeLane(vid, target_lane, duration=3.0)
                # Also reduce speed moderately during maneuver
                traci.vehicle.slowDown(vid, self.config.advisory_speed_mps * 1.2, duration=2.0)
                self.advisory_log.append({
                    "vehicle_id": vid,
                    "tick": self.current_tick,
                    "sim_time": self.sim_time_s,
                    "action": "LANE_CHANGE",
                    "target_lane": target_lane,
                    "hazard_id": hazard.hazard_id
                })
        except traci.TraCIException as e:
            # Rejection by SUMO due to safety constraint is logged, not suppressed
            self.advisory_log.append({
                "vehicle_id": vid,
                "tick": self.current_tick,
                "sim_time": self.sim_time_s,
                "action": "REJECTED_BY_SUMO_SAFETY",
                "error": str(e)
            })

    def process_camera_frame(
        self,
        frame: np.ndarray,
        location_m: Tuple[float, float] = (0.0, 0.0)
    ) -> List[HazardDetection]:
        """Runs neural inference on junction camera frame and registers active hazards."""
        detections = self.perception.detect_hazards(
            image=frame,
            rsu_id=self.config.rsu_id,
            location_m=location_m
        )
        self.active_hazards = detections
        return detections

    def inject_hazard(self, hazard: HazardDetection) -> None:
        """Directly registers a hazard into the active RSU dissemination queue."""
        self.active_hazards.append(hazard)

    def clear_hazards(self) -> None:
        """Clears all active hazards from the RSU broadcast."""
        self.active_hazards.clear()
        self.vehicles_warned.clear()

    def close(self) -> None:
        """Gracefully disconnects from TraCI and shuts down SUMO."""
        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                self.is_running = False
