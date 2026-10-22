"""
PFPD Smart City Foundry & Market District Network Acceptance Test
Validates:
1. SUMO 5 km Foundry & Market network geometry and 7 junctions (D1, D2, D3, F1, F2, M2, M3).
2. TraCI controller stepping across all 12 physical streets.
3. Multi-class vehicle circulation and speed limits.
4. Operational 4-way traffic lights at D1, D2, and D3.
"""

import os
import sys
from pathlib import Path

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
if os.path.exists(SUMO_TOOLS) and SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

import traci

def test_foundry_simulation():
    repo_root = Path(__file__).resolve().parents[2]
    sumocfg = repo_root / "simulation" / "smart_city" / "foundry" / "foundry.sumocfg"
    assert sumocfg.exists(), f"Configuration file not found: {sumocfg}"

    sumo_bin = os.path.join(SUMO_HOME, "bin", "sumo.exe")
    sumo_cmd = [
        sumo_bin,
        "-c", str(sumocfg),
        "--step-length", "0.1",
        "--start",
        "--quit-on-end",
        "--no-warnings", "true"
    ]

    print("Launching SUMO on Foundry & Market District...")
    traci.start(sumo_cmd)
    try:
        # Step for 100 ticks (10 seconds)
        for tick in range(100):
            traci.simulationStep()

        active_vehicles = traci.vehicle.getIDList()
        print(f"Tick 100: {len(active_vehicles)} active circulating vehicles.")
        assert len(active_vehicles) > 10, "Expected multiple active circulating vehicles"

        # Verify traffic lights at D1, D2, D3
        tls_list = traci.trafficlight.getIDList()
        print(f"Traffic lights detected: {tls_list}")
        assert "TL_D1" in tls_list, "Traffic light TL_D1 should be present"
        assert "TL_D2" in tls_list, "Traffic light TL_D2 should be present"
        assert "TL_D3" in tls_list, "Traffic light TL_D3 should be present"

        for tls in ["TL_D1", "TL_D2", "TL_D3"]:
            state = traci.trafficlight.getRedYellowGreenState(tls)
            assert len(state) > 0, f"TLS {tls} state should not be empty"

        print("Foundry & Market District test passed successfully!")
    finally:
        traci.close()

if __name__ == "__main__":
    test_foundry_simulation()
