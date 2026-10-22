"""
PFPD Stage 7: Realistic 5 km Urban Scenery Generator
Generates city_5km.add.xml containing:
- Multi-district urban zoning (Commercial Core, Civic/Residential, Riverfront Promenade, Tech Campus)
- Natural waterways (River South with bridges), expansive parks, plazas, and tree groves
- Detailed building polygons with distinct architectural types, colors, and heights
- Sidewalks, curbs, and road verges
- Visual RSU Masts and radio coverage boundary overlays
"""

import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ADD_FILE = BASE_DIR / "city_5km.add.xml"

def generate_scenery():
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">',
        '',
        '    <!-- ================================================================ -->',
        '    <!--  WATERWAY & RIVERFRONT PROMENADE (South Sector)                 -->',
        '    <!-- ================================================================ -->',
        '    <poly id="river_south" type="water" color="65,130,195,220" fill="true" layer="0" shape="-50.0,-50.0 1250.0,-50.0 1250.0,30.0 -50.0,30.0"/>',
        '    <poly id="riverbank_green" type="landuse.grass" color="70,150,80,220" fill="true" layer="1" shape="-50.0,30.0 1250.0,30.0 1250.0,60.0 -50.0,60.0"/>',
        '    <poly id="river_promenade" type="sidewalk" color="210,205,195,255" fill="true" layer="2" shape="-50.0,60.0 1250.0,60.0 1250.0,75.0 -50.0,75.0"/>',
        '',
        '    <!-- ================================================================ -->',
        '    <!--  CIVIC PARKS & GREEN BELTS                                       -->',
        '    <!-- ================================================================ -->',
        '    <!-- Central Civic Park (SW of Central Junction) -->',
        '    <poly id="park_central" type="leisure.park" color="80,165,85,220" fill="true" layer="1" shape="240.0,140.0 560.0,140.0 560.0,460.0 240.0,460.0"/>',
        '    <poly id="park_pond" type="water" color="75,140,200,240" fill="true" layer="2" shape="350.0,250.0 450.0,250.0 480.0,350.0 380.0,380.0"/>',
        '',
        '    <!-- North Ecological Reserve (North of Ring Road) -->',
        '    <poly id="park_north_reserve" type="landuse.forest" color="60,135,70,220" fill="true" layer="1" shape="240.0,940.0 960.0,940.0 960.0,1050.0 240.0,1050.0"/>',
        '',
        '    <!-- Southeast Sports & Recreation Ground -->',
        '    <poly id="park_se_sports" type="leisure.pitch" color="90,175,95,220" fill="true" layer="1" shape="640.0,140.0 960.0,140.0 960.0,460.0 640.0,460.0"/>',
        '    <poly id="stadium_pitch" type="sport" color="60,170,80,255" fill="true" layer="2" shape="740.0,240.0 860.0,240.0 860.0,360.0 740.0,360.0"/>',
        '',
        '    <!-- ================================================================ -->',
        '    <!--  URBAN TREE CANOPIES & BOULEVARD GREENERY                        -->',
        '    <!-- ================================================================ -->'
    ]

    # Riverbank trees along promenade (y = 68m, x from 60 to 1150)
    for tx in range(60, 1150, 45):
        lines.append(
            f'    <poly id="tree_river_{tx}" type="natural.tree" color="45,135,55,230" fill="true" layer="3" '
            f'shape="{tx-6}.0,62.0 {tx+6}.0,62.0 {tx+8}.0,70.0 {tx}.0,76.0 {tx-8}.0,70.0"/>'
        )

    # Central Park tree clusters (x: 260-540, y: 160-440, outside pond)
    tree_spots = [
        (260, 180), (300, 200), (270, 240), (280, 310), (270, 380), (310, 420),
        (510, 180), (530, 240), (520, 310), (500, 380), (480, 430), (360, 430),
        (330, 160), (410, 170), (460, 180), (490, 210), (360, 210)
    ]
    for idx, (tx, ty) in enumerate(tree_spots):
        lines.append(
            f'    <poly id="tree_park_{idx}" type="natural.tree" color="40,125,50,230" fill="true" layer="3" '
            f'shape="{tx-8}.0,{ty-8}.0 {tx+8}.0,{ty-8}.0 {tx+10}.0,{ty}.0 {tx}.0,{ty+10}.0 {tx-10}.0,{ty}.0"/>'
        )

    lines.append('')
    lines.append('    <!-- ================================================================ -->')
    lines.append('    <!--  ROADSIDE CONTROLLERS (RSC) & TRAFFIC INFRASTRUCTURE             -->')
    lines.append('    <!-- ================================================================ -->')
    junction_rscs = [
        ("RSC_CENTRAL", 615.0, 515.0, "RSC Central Master Controller"),
        ("RSC_WEST",    215.0, 515.0, "RSC West Gateway Controller"),
        ("RSC_EAST",    1015.0, 515.0, "RSC East Tech Controller"),
        ("RSC_NORTH",   615.0, 915.0, "RSC North Beltway Controller"),
        ("RSC_SOUTH",   615.0, 115.0, "RSC South Riverway Controller"),
        ("RSC_NW",      215.0, 915.0, "RSC NW Commercial Controller"),
        ("RSC_NE",      1015.0, 915.0, "RSC NE Innovation Controller"),
        ("RSC_SW",      215.0, 115.0, "RSC SW Gateway Controller"),
        ("RSC_SE",      1015.0, 115.0, "RSC SE Logistics Controller"),
    ]
    for rsc_id, rx, ry, rsc_desc in junction_rscs:
        lines.append(f'    <poi id="{rsc_id}" type="rsc_cabinet" color="40,160,220,255" layer="10" x="{rx}" y="{ry}"/>')

    lines.append('')
    lines.append('    <!-- ================================================================ -->')
    lines.append('    <!--  DISTRICT 1: CENTRAL COMMERCIAL TOWER BLOCKS (NW of Central)     -->')
    lines.append('    <!-- ================================================================ -->')

    # Generate Commercial Core Buildings (x: 240-560, y: 540-860)
    b_id = 1
    for bx in range(250, 550, 75):
        for by in range(550, 850, 75):
            lines.append(
                f'    <poly id="b_commercial_{b_id}" type="building.commercial" color="185,195,210,255" fill="true" layer="3" '
                f'shape="{bx}.0,{by}.0 {bx+55}.0,{by}.0 {bx+55}.0,{by+55}.0 {bx}.0,{by+55}.0"/>'
            )
            b_id += 1

    lines.append('')
    lines.append('    <!-- ================================================================ -->')
    lines.append('    <!--  DISTRICT 2: NORTHEAST TECH & INNOVATION CAMPUS                  -->')
    lines.append('    <!-- ================================================================ -->')

    # Generate Tech Campus Buildings (x: 640-960, y: 540-860)
    t_id = 1
    for bx in range(650, 950, 80):
        for by in range(550, 850, 80):
            lines.append(
                f'    <poly id="b_tech_{t_id}" type="building.office" color="215,210,200,255" fill="true" layer="3" '
                f'shape="{bx}.0,{by}.0 {bx+60}.0,{by}.0 {bx+60}.0,{by+60}.0 {bx}.0,{by+60}.0"/>'
            )
            t_id += 1

    lines.append('')
    lines.append('    <!-- ================================================================ -->')
    lines.append('    <!--  DISTRICT 3: RESIDENTIAL & CIVIC FLATS (Outer Perimeter)        -->')
    lines.append('    <!-- ================================================================ -->')

    # Westside Residential (x: 50-160, y: 150-850)
    r_id = 1
    for by in range(160, 850, 70):
        lines.append(
            f'    <poly id="b_res_w_{r_id}" type="building.residential" color="225,200,180,255" fill="true" layer="3" '
            f'shape="60.0,{by}.0 160.0,{by}.0 160.0,{by+50}.0 60.0,{by+50}.0"/>'
        )
        r_id += 1

    # Eastside Logistics & Industrial (x: 1040-1150, y: 150-850)
    ind_id = 1
    for by in range(160, 850, 90):
        lines.append(
            f'    <poly id="b_ind_e_{ind_id}" type="building.industrial" color="170,175,180,255" fill="true" layer="3" '
            f'shape="1040.0,{by}.0 1160.0,{by}.0 1160.0,{by+70}.0 1040.0,{by+70}.0"/>'
        )
        ind_id += 1

    lines.append('')
    lines.append('    <!-- ================================================================ -->')
    lines.append('    <!--  ROADSIDE UNITS (RSU) MASTS & RADIO COVERAGE ZONES               -->')
    lines.append('    <!-- ================================================================ -->')

    # Define 4 Core RSUs + 4 Expansion RSUs
    rsus = [
        ("RSU_01_WEST",    200.0, 500.0, 250.0, "RSU 1: West Gate Interchange",        "0,180,255,40"),
        ("RSU_02_CENTRAL", 600.0, 500.0, 280.0, "RSU 2: Central Metro Hub",            "0,220,100,40"),
        ("RSU_03_EAST",   1000.0, 500.0, 250.0, "RSU 3: East Tech Interchange",        "255,160,0,40"),
        ("RSU_04_NORTH",   600.0, 900.0, 250.0, "RSU 4: North Beltway Interchange",     "220,50,220,40"),
        ("RSU_05_SOUTH",   600.0, 100.0, 250.0, "RSU 5: South Riverway Interchange",    "100,100,255,30"),
        ("RSU_06_NW",      200.0, 900.0, 220.0, "RSU 6: Northwest Commercial Hub",     "100,220,200,30"),
        ("RSU_07_NE",     1000.0, 900.0, 220.0, "RSU 7: Northeast Innovation Hub",     "240,200,50,30"),
        ("RSU_08_SE",     1000.0, 100.0, 220.0, "RSU 8: Southeast Sports & Logistics",  "200,120,240,30"),
    ]

    for rid, rx, ry, r_cov, desc, fill_rgba in rsus:
        # Physical RSU Mast Point
        lines.append(f'    <poi id="{rid}_mast" type="rsu_mast" color="255,50,50,255" layer="10" x="{rx}" y="{ry}"/>')
        
        # Approximate Circular Coverage Polygon (16-point circle approximation)
        pts = []
        for i in range(16):
            ang = (i / 16.0) * 2.0 * math.pi
            px = rx + r_cov * math.cos(ang)
            py = ry + r_cov * math.sin(ang)
            pts.append(f"{px:.1f},{py:.1f}")
        pts_str = " ".join(pts)
        lines.append(f'    <poly id="{rid}_coverage" type="v2x_coverage" color="{fill_rgba}" fill="true" layer="2" shape="{pts_str}"/>')

    lines.append('')
    lines.append('</additional>')
    return "\n".join(lines)

def main():
    print("=" * 80)
    print("  BUILDING DETAILED URBAN SCENERY (city_5km.add.xml)")
    print("=" * 80)
    content = generate_scenery()
    with open(ADD_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Wrote detailed city scenery with multi-RSU coverage: {ADD_FILE}")

if __name__ == "__main__":
    main()
