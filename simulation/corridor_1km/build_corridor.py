import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 1. Define Nodes (Junctions)
# Total length = 1000m. Two signalized intersections at 300m and 800m.
nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <!-- Main Corridor Nodes -->
    <node id="N_entry" x="0.0" y="0.0" type="priority"/>
    <node id="TL_01" x="300.0" y="0.0" type="traffic_light"/>
    <node id="TL_02" x="800.0" y="0.0" type="traffic_light"/>
    <node id="N_exit" x="1000.0" y="0.0" type="priority"/>

    <!-- Cross Street 1 Nodes (at 300m) -->
    <node id="N_cross1_north" x="300.0" y="100.0" type="priority"/>
    <node id="N_cross1_south" x="300.0" y="-100.0" type="priority"/>

    <!-- Cross Street 2 Nodes (at 800m) -->
    <node id="N_cross2_north" x="800.0" y="100.0" type="priority"/>
    <node id="N_cross2_south" x="800.0" y="-100.0" type="priority"/>
</nodes>
"""

# 2. Define Edges (Roads)
# Main arterial: 2 lanes each direction, speed limit 13.89 m/s (50 km/h)
# Cross streets: 1 lane each direction, speed limit 11.11 m/s (40 km/h)
edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Main Corridor (West to East) -->
    <edge id="edge_1" from="N_entry" to="TL_01" numLanes="2" speed="13.89" priority="3"/>
    <edge id="edge_2" from="TL_01" to="TL_02" numLanes="2" speed="13.89" priority="3"/>
    <edge id="edge_3" from="TL_02" to="N_exit" numLanes="2" speed="13.89" priority="3"/>

    <!-- Main Corridor Reverse (East to West) -->
    <edge id="edge_3_rev" from="N_exit" to="TL_02" numLanes="2" speed="13.89" priority="3"/>
    <edge id="edge_2_rev" from="TL_02" to="TL_01" numLanes="2" speed="13.89" priority="3"/>
    <edge id="edge_1_rev" from="TL_01" to="N_entry" numLanes="2" speed="13.89" priority="3"/>

    <!-- Cross Street 1 -->
    <edge id="cross1_in_n" from="N_cross1_north" to="TL_01" numLanes="1" speed="11.11" priority="2"/>
    <edge id="cross1_out_s" from="TL_01" to="N_cross1_south" numLanes="1" speed="11.11" priority="2"/>
    <edge id="cross1_in_s" from="N_cross1_south" to="TL_01" numLanes="1" speed="11.11" priority="2"/>
    <edge id="cross1_out_n" from="TL_01" to="N_cross1_north" numLanes="1" speed="11.11" priority="2"/>

    <!-- Cross Street 2 -->
    <edge id="cross2_in_n" from="N_cross2_north" to="TL_02" numLanes="1" speed="11.11" priority="2"/>
    <edge id="cross2_out_s" from="TL_02" to="N_cross2_south" numLanes="1" speed="11.11" priority="2"/>
    <edge id="cross2_in_s" from="N_cross2_south" to="TL_02" numLanes="1" speed="11.11" priority="2"/>
    <edge id="cross2_out_n" from="TL_02" to="N_cross2_north" numLanes="1" speed="11.11" priority="2"/>
</edges>
"""

# 3. Define Additional Elements (Visual Hazard POI, RSU station marker)
additional_xml = """<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <!-- RSU Location at mid-corridor (x=500, y=10) -->
    <poi id="RSU_ZONE_01" color="0,200,255" layer="10" x="500.0" y="12.0" width="8.0" height="8.0" type="RSU_STATION"/>

    <!-- Pothole Hazard Location on edge_2 at x=650 (350m into edge_2) on Lane 0 -->
    <poi id="HAZARD_POTHOLE_01" color="255,50,50" layer="10" lane="edge_2_0" pos="350.0" width="3.2" height="3.2" type="ROAD_HAZARD"/>
</additional>
"""

# 4. Define Routes & Vehicle Types
routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicle Types -->
    <vType id="car_lead_detector" accel="2.6" decel="4.5" sigma="0.2" length="4.8" minGap="2.5" maxSpeed="13.89" color="0,150,255" lcStrategic="1.0" lcCooperative="1.0"/>
    <vType id="car_trail_connected" accel="2.6" decel="4.5" sigma="0.2" length="4.8" minGap="2.5" maxSpeed="13.89" color="0,200,100" lcStrategic="1.0" lcCooperative="1.0"/>
    <vType id="car_trail_unconnected" accel="2.6" decel="6.5" sigma="0.5" length="4.8" minGap="2.5" maxSpeed="13.89" color="180,180,180" lcStrategic="0.1" lcCooperative="0.1"/>
    <vType id="car_ambient" accel="2.2" decel="4.0" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="13.89" color="200,200,200"/>

    <!-- Corridor Route (West to East) -->
    <route id="corridor_we" edges="edge_1 edge_2 edge_3"/>
    <route id="corridor_ew" edges="edge_3_rev edge_2_rev edge_1_rev"/>
    <route id="cross1_ns" edges="cross1_in_n cross1_out_s"/>
    <route id="cross2_sn" edges="cross2_in_s cross2_out_n"/>

    <!-- Experimental Vehicles -->
    <!-- Vehicle A: Leading Edge-AI Detector -->
    <vehicle id="veh_A_detector" type="car_lead_detector" route="corridor_we" depart="0.0" departLane="0" departSpeed="11.11"/>

    <!-- Vehicle B: Trailing Connected Vehicle (receives DENM) -->
    <vehicle id="veh_B_connected" type="car_trail_connected" route="corridor_we" depart="14.0" departLane="0" departSpeed="11.11"/>

    <!-- Vehicle C: Trailing Unconnected Vehicle (Baseline contrast) -->
    <vehicle id="veh_C_baseline" type="car_trail_unconnected" route="corridor_we" depart="28.0" departLane="0" departSpeed="11.11"/>

    <!-- Ambient traffic flows for realistic corridor activity -->
    <flow id="flow_reverse" type="car_ambient" route="corridor_ew" begin="2.0" end="150.0" period="12" departLane="random" departSpeed="11.11"/>
    <flow id="flow_cross1" type="car_ambient" route="cross1_ns" begin="5.0" end="150.0" period="20" departSpeed="10.0"/>
    <flow id="flow_cross2" type="car_ambient" route="cross2_sn" begin="10.0" end="150.0" period="25" departSpeed="10.0"/>
</routes>
"""

# 5. Define SUMO Configuration
sumocfg_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="corridor.net.xml"/>
        <route-files value="corridor.rou.xml"/>
        <additional-files value="corridor.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="180"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
        <collision.stoptime value="2"/>
    </processing>
    <gui_only>
        <start value="false"/>
    </gui_only>
</configuration>
"""

def main():
    print(f"[BuildCorridor] Writing XML descriptors in {BASE_DIR}...")
    (BASE_DIR / "corridor.nod.xml").write_text(nodes_xml, encoding="utf-8")
    (BASE_DIR / "corridor.edg.xml").write_text(edges_xml, encoding="utf-8")
    (BASE_DIR / "corridor.add.xml").write_text(additional_xml, encoding="utf-8")
    (BASE_DIR / "corridor.rou.xml").write_text(routes_xml, encoding="utf-8")
    (BASE_DIR / "corridor.sumocfg").write_text(sumocfg_xml, encoding="utf-8")

    netconvert_path = r"C:\Program Files (x86)\Eclipse\Sumo\bin\netconvert.exe"
    net_output = BASE_DIR / "corridor.net.xml"

    cmd = [
        netconvert_path,
        "--node-files", str(BASE_DIR / "corridor.nod.xml"),
        "--edge-files", str(BASE_DIR / "corridor.edg.xml"),
        "--output-file", str(net_output),
        "--junctions.join",
        "--tls.default-type", "actuated",
        "--tls.cycle.time", "60"
    ]

    print("[BuildCorridor] Running netconvert...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        size_kb = os.path.getsize(net_output) / 1024
        print(f"✓ Successfully generated {net_output.name} ({size_kb:.1f} KB)")
    else:
        print(f"✗ netconvert failed:\n{res.stderr}")

if __name__ == "__main__":
    main()
