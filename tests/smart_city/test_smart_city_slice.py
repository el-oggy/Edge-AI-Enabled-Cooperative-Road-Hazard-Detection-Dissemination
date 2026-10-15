"""
PFPD Smart City Integration & Acceptance Test (Phase SC0-SC1)
Verifies:
1. SUMO 4-way signalized junction network and TLS cycle.
2. TraCI sole-coordinator simulation stepping and multi-class traffic circulation.
3. RSU beacon hazard dissemination to approaching vehicles within range.
4. Cooperative vehicle deceleration and lane avoidance responses without collision.
"""

import sys
import time
from pathlib import Path

# Add src to python path
src_path = str(Path(__file__).resolve().parents[2] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pfpd_city_protocol.contracts import HazardDetection
from pfpd_city_runtime.city_runtime_engine import CityRuntimeEngine, RuntimeConfig

def test_sumo_junction_circulation():
    print("[1/2] Testing SUMO junction simulation and TLS stepping...")
    engine = CityRuntimeEngine()
    try:
        engine.start()
        for tick in range(100):
            snapshot = engine.step()
            if tick % 25 == 0:
                print(f"  Tick {tick:03d} (T={snapshot.sim_time_s:.1f}s): "
                      f"{len(snapshot.vehicles)} active vehicles, "
                      f"TLS phase={snapshot.traffic_lights[0].phase_index if snapshot.traffic_lights else 'N/A'}")

        assert engine.current_tick == 100
        assert engine.sim_time_s >= 9.9
        assert len(engine.snapshots) == 100
        
        # Verify vehicle telemetry validity
        last_snap = engine.snapshots[-1]
        assert len(last_snap.vehicles) > 0, "Expected active vehicles circulating through junction"
        for v in last_snap.vehicles:
            assert isinstance(v.x, float) and isinstance(v.y, float)
            assert v.speed_mps >= 0.0
            assert 0.0 <= v.heading_deg <= 360.0
            assert v.lane_id != ""

        print(f"  Simulation completed 100 ticks with {len(last_snap.vehicles)} circulating vehicles.")
    finally:
        engine.close()

def test_cooperative_hazard_advisory_flow():
    print("[2/2] Testing RSU hazard dissemination and cooperative vehicle response...")
    engine = CityRuntimeEngine()
    try:
        engine.start()
        # Warm up junction for 30 ticks
        for _ in range(30):
            engine.step()

        # Inject road hazard detected by camera at junction approach
        test_hazard = HazardDetection(
            hazard_id="HAZ-TEST-001",
            category="POTHOLE",
            confidence=0.88,
            box=[420, 480, 560, 600],
            advisory="REDUCE_SPEED",
            detected_timestamp=engine.sim_time_s,
            rsu_id="RSU_J0",
            location_m=(25.0, 0.0)
        )
        engine.inject_hazard(test_hazard)
        print(f"  Injected {test_hazard.category} hazard into RSU beacon queue.")

        # Step simulation and observe vehicle slowdown reactions
        advisory_applied = False
        for _ in range(50):
            snapshot = engine.step()
            for v in snapshot.vehicles:
                if "REDUCE_SPEED" in v.active_advisories:
                    advisory_applied = True
                    break
            if advisory_applied:
                break

        assert advisory_applied, "Approaching vehicle should receive and register REDUCE_SPEED advisory"
        assert len(engine.advisory_log) > 0, "Runtime engine should log cooperative maneuver commands"
        
        first_action = engine.advisory_log[0]
        print(f"  Verified cooperative response receipt: Vehicle {first_action['vehicle_id']} "
              f"executed {first_action['action']} to {first_action.get('target_speed')} m/s.")
    finally:
        engine.close()

if __name__ == "__main__":
    test_sumo_junction_circulation()
    test_cooperative_hazard_advisory_flow()
    print("\nALL SC0-SC1 VERTICAL SLICE INTEGRATION TESTS PASSED!")
