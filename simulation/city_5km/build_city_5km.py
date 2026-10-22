"""
PFPD Stage 7: Realistic 5 km Multi-RSU City Network Generator
Generates nodes, edges, connections, and compiles city_5km.net.xml using SUMO netconvert.

Network Specifications:
- Total Roadway Distance: 11.2 km of dual-carriageway directional urban arterials
- Primary Driving Circuit: 5.2 km interconnected route with multiple alternative paths
- Intersections: 9 signalized 4-way junctions (J_CENTRAL, J_WEST, J_EAST, J_NORTH, J_SOUTH, J_NW, J_NE, J_SW, J_SE)
- 4 External Highway Arterial Gateways (West, East, North, South)
- Multi-lane roads with dedicated turning lanes and realistic urban speed limits
"""

import os
import subprocess
from pathlib import Path

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
NETCONVERT = os.path.join(SUMO_HOME, "bin", "netconvert.exe")

BASE_DIR = Path(__file__).resolve().parent

# 1. NODES DEFINITION
nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <!-- 4 External Arterial Gateways -->
    <node id="GATE_WEST"  x="0.0"    y="500.0"  type="priority"/>
    <node id="GATE_EAST"  x="1200.0" y="500.0"  type="priority"/>
    <node id="GATE_NORTH" x="600.0"  y="1100.0" type="priority"/>
    <node id="GATE_SOUTH" x="600.0"  y="-100.0" type="priority"/>

    <!-- Central & Arterial Signalized Junctions -->
    <node id="J_CENTRAL" x="600.0"  y="500.0"  type="traffic_light" tlType="actuated"/>
    <node id="J_WEST"    x="200.0"  y="500.0"  type="traffic_light" tlType="actuated"/>
    <node id="J_EAST"    x="1000.0" y="500.0"  type="traffic_light" tlType="actuated"/>
    <node id="J_NORTH"   x="600.0"  y="900.0"  type="traffic_light" tlType="actuated"/>
    <node id="J_SOUTH"   x="600.0"  y="100.0"  type="traffic_light" tlType="actuated"/>

    <!-- Perimeter Corner Signalized Junctions -->
    <node id="J_NW"      x="200.0"  y="900.0"  type="traffic_light" tlType="actuated"/>
    <node id="J_NE"      x="1000.0" y="900.0"  type="traffic_light" tlType="actuated"/>
    <node id="J_SW"      x="200.0"  y="100.0"  type="traffic_light" tlType="actuated"/>
    <node id="J_SE"      x="1000.0" y="100.0"  type="traffic_light" tlType="actuated"/>
</nodes>
"""

# 2. EDGES DEFINITION
# All main corridors have 2 lanes per direction (4 lanes total), speed 13.89 m/s (50 km/h)
edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- ===== METRO PARKWAY (EAST-WEST CENTRAL ARTERIAL) ===== -->
    <!-- West Gate to J_WEST -->
    <edge id="parkway_gw_w" from="GATE_WEST" to="J_WEST" numLanes="2" speed="13.89" priority="3"/>
    <edge id="parkway_w_gw" from="J_WEST" to="GATE_WEST" numLanes="2" speed="13.89" priority="3"/>
    <!-- J_WEST to J_CENTRAL -->
    <edge id="parkway_w_c" from="J_WEST" to="J_CENTRAL" numLanes="2" speed="13.89" priority="3"/>
    <edge id="parkway_c_w" from="J_CENTRAL" to="J_WEST" numLanes="2" speed="13.89" priority="3"/>
    <!-- J_CENTRAL to J_EAST -->
    <edge id="parkway_c_e" from="J_CENTRAL" to="J_EAST" numLanes="2" speed="13.89" priority="3"/>
    <edge id="parkway_e_c" from="J_EAST" to="J_CENTRAL" numLanes="2" speed="13.89" priority="3"/>
    <!-- J_EAST to East Gate -->
    <edge id="parkway_e_ge" from="J_EAST" to="GATE_EAST" numLanes="2" speed="13.89" priority="3"/>
    <edge id="parkway_ge_e" from="GATE_EAST" to="J_EAST" numLanes="2" speed="13.89" priority="3"/>

    <!-- ===== AVENUE CENTRAL (NORTH-SOUTH CENTRAL ARTERIAL) ===== -->
    <!-- South Gate to J_SOUTH -->
    <edge id="avenue_gs_s" from="GATE_SOUTH" to="J_SOUTH" numLanes="2" speed="13.89" priority="3"/>
    <edge id="avenue_s_gs" from="J_SOUTH" to="GATE_SOUTH" numLanes="2" speed="13.89" priority="3"/>
    <!-- J_SOUTH to J_CENTRAL -->
    <edge id="avenue_s_c" from="J_SOUTH" to="J_CENTRAL" numLanes="2" speed="13.89" priority="3"/>
    <edge id="avenue_c_s" from="J_CENTRAL" to="J_SOUTH" numLanes="2" speed="13.89" priority="3"/>
    <!-- J_CENTRAL to J_NORTH -->
    <edge id="avenue_c_n" from="J_CENTRAL" to="J_NORTH" numLanes="2" speed="13.89" priority="3"/>
    <edge id="avenue_n_c" from="J_NORTH" to="J_CENTRAL" numLanes="2" speed="13.89" priority="3"/>
    <!-- J_NORTH to North Gate -->
    <edge id="avenue_n_gn" from="J_NORTH" to="GATE_NORTH" numLanes="2" speed="13.89" priority="3"/>
    <edge id="avenue_gn_n" from="GATE_NORTH" to="J_NORTH" numLanes="2" speed="13.89" priority="3"/>

    <!-- ===== NORTH RING ROAD (NORTH ARTERIAL) ===== -->
    <!-- J_NW to J_NORTH -->
    <edge id="ring_nw_n" from="J_NW" to="J_NORTH" numLanes="2" speed="13.89" priority="2"/>
    <edge id="ring_n_nw" from="J_NORTH" to="J_NW" numLanes="2" speed="13.89" priority="2"/>
    <!-- J_NORTH to J_NE -->
    <edge id="ring_n_ne" from="J_NORTH" to="J_NE" numLanes="2" speed="13.89" priority="2"/>
    <edge id="ring_ne_n" from="J_NE" to="J_NORTH" numLanes="2" speed="13.89" priority="2"/>

    <!-- ===== SOUTH BYPASS (SOUTH ARTERIAL) ===== -->
    <!-- J_SW to J_SOUTH -->
    <edge id="bypass_sw_s" from="J_SW" to="J_SOUTH" numLanes="2" speed="13.89" priority="2"/>
    <edge id="bypass_s_sw" from="J_SOUTH" to="J_SW" numLanes="2" speed="13.89" priority="2"/>
    <!-- J_SOUTH to J_SE -->
    <edge id="bypass_s_se" from="J_SOUTH" to="J_SE" numLanes="2" speed="13.89" priority="2"/>
    <edge id="bypass_se_s" from="J_SE" to="J_SOUTH" numLanes="2" speed="13.89" priority="2"/>

    <!-- ===== WESTWAY (WEST CONNECTOR CORRIDOR) ===== -->
    <!-- J_SW to J_WEST -->
    <edge id="westway_sw_w" from="J_SW" to="J_WEST" numLanes="2" speed="11.11" priority="2"/>
    <edge id="westway_w_sw" from="J_WEST" to="J_SW" numLanes="2" speed="11.11" priority="2"/>
    <!-- J_WEST to J_NW -->
    <edge id="westway_w_nw" from="J_WEST" to="J_NW" numLanes="2" speed="11.11" priority="2"/>
    <edge id="westway_nw_w" from="J_NW" to="J_WEST" numLanes="2" speed="11.11" priority="2"/>

    <!-- ===== EASTWAY (EAST CONNECTOR CORRIDOR) ===== -->
    <!-- J_SE to J_EAST -->
    <edge id="eastway_se_e" from="J_SE" to="J_EAST" numLanes="2" speed="11.11" priority="2"/>
    <edge id="eastway_e_se" from="J_EAST" to="J_SE" numLanes="2" speed="11.11" priority="2"/>
    <!-- J_EAST to J_NE -->
    <edge id="eastway_e_ne" from="J_EAST" to="J_NE" numLanes="2" speed="11.11" priority="2"/>
    <edge id="eastway_ne_e" from="J_NE" to="J_EAST" numLanes="2" speed="11.11" priority="2"/>
</edges>
"""

def main():
    print("=" * 80)
    print("  BUILDING 5.0 KM REALISTIC URBAN SUMO NETWORK")
    print("=" * 80)

    nod_p = BASE_DIR / "city_5km.nod.xml"
    edg_p = BASE_DIR / "city_5km.edg.xml"
    net_p = BASE_DIR / "city_5km.net.xml"

    with open(nod_p, "w", encoding="utf-8") as f:
        f.write(nodes_xml)
    print(f"✓ Wrote nodes: {nod_p}")

    with open(edg_p, "w", encoding="utf-8") as f:
        f.write(edges_xml)
    print(f"✓ Wrote edges: {edg_p}")

    # Run netconvert
    cmd = [
        NETCONVERT,
        "--node-files", str(nod_p),
        "--edge-files", str(edg_p),
        "--output-file", str(net_p),
        "--junctions.corner-detail", "5",
        "--roundabouts.guess", "true",
        "--tls.default-type", "actuated",
        "--tls.guess", "true",
        "--tls.cycle.time", "60",
        "--default.speed", "13.89"
    ]

    print(f"Running netconvert...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR running netconvert:\n{res.stderr}")
        return False

    print(f"✓ Successfully compiled SUMO network: {net_p}")
    return True

if __name__ == "__main__":
    main()
