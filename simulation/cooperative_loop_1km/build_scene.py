"""Explicitly build the native cooperative-loop SUMO package.

Run from any working directory with the project Python:
    python simulation/cooperative_loop_1km/build_scene.py

This builder writes only files inside this package. The launcher must load the
checked-in artifacts and must never invoke this script implicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
N1_ASSETS = PROJECT / "simulation" / "foundry_market" / "assets"
SUMO_HOME_CANDIDATES = [
    Path(os.environ.get("SUMO_HOME", "")),
    Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
    Path(r"C:\Program Files\Eclipse\Sumo"),
]
SUMO_HOME = next((p for p in SUMO_HOME_CANDIDATES if p and (p / "tools" / "sumolib").is_dir()), None)
if SUMO_HOME is None:
    raise RuntimeError("Could not locate installed SUMO tools; set SUMO_HOME")
sys.path.insert(0, str(SUMO_HOME / "tools"))

import sumolib  # noqa: E402
from sumolib.geomhelper import positionAtShapeOffset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

NETCONVERT = SUMO_HOME / "bin" / ("netconvert.exe" if os.name == "nt" else "netconvert")
if not NETCONVERT.is_file():
    raise RuntimeError(f"netconvert executable missing: {NETCONVERT}")

BG = "#C5B391"
COLORS = {
    "sand": "#C5B391", "yard": "#B5A589", "road": "#464747",
    "curb": "#D8CDB7", "rust": "#875843", "steel": "#61605A",
    "ochre": "#C18D40", "olive": "#72785A", "cream": "#E5D6B6",
}

RING = ["SW", "S1", "J1", "S3", "SE", "E", "NE", "N3", "J2", "N1", "NW", "W"]
CW = [f"cw_{a}_{b}" for a,b in zip(RING, RING[1:]+RING[:1])]
REVERSE_RING = RING[:1] + list(reversed(RING[1:]))
CCW = [f"ccw_{a}_{b}" for a,b in zip(REVERSE_RING, REVERSE_RING[1:]+REVERSE_RING[:1])]
BOUNDS = [-180, -240, 2440, 1200]
PANEL = {"position_m": [2070, 520], "width_m": 620, "height_m": 475,
         "bounds_m": [1760, 282.5, 2380, 757.5], "reserved_for": "live controller image POI"}
BRIDGE = {"id": "foundry_viaduct", "upper_edges": ["bridge_W_E", "bridge_E_W"],
          "lower_edges": ["underpass_J1_C2", "underpass_C2_J2", "underpass_J2_C2", "underpass_C2_J1"],
          "deck_height_m": 8, "crossing_position_m": [800,450], "bounds_m": [630,420,970,480]}
TARGETS = [
    ("pothole", "pothole", "cw_SW_S1", 210),
    ("construction", "construction", "cw_S3_SE", 160),
    ("accident_obstacle", "accident_obstacle", "cw_E_NE", 210),
    ("pothole_02", "pothole", "cw_J2_N1", 240),
    ("construction_02", "construction", "cw_NW_W", 180),
    ("accident_obstacle_02", "accident_obstacle", "cw_W_SW", 240),
    ("pothole_03", "pothole", "street_S1_C1", 200),
    ("construction_03", "construction", "bridge_E_C3", 110),
    ("accident_obstacle_03", "accident_obstacle", "underpass_C2_J2", 220),
]


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_geometry() -> None:
    nodes = ET.Element("nodes")
    specs = [
        ("SW",0,0), ("S1",400,-40), ("J1",800,0), ("S3",1200,60),
        ("SE",1600,120), ("E",1660,480), ("NE",1560,960), ("N3",1180,1000),
        ("J2",800,960), ("N1",400,990), ("NW",0,900), ("W",-70,450),
        ("C1",400,450), ("C2",800,450), ("C3",1200,450),
        ("B_W",650,450), ("B_E",950,450),
        ("S_END",800,-170), ("N_END",800,1140),
    ]
    lights = {"S1","J1","S3","E","N3","J2","N1","W","C1","C3"}
    for node_id,x,y in specs:
        ET.SubElement(nodes,"node",{"id":node_id,"x":str(x),"y":str(y),
            "z":"8" if node_id in ("B_W","B_E") else "0",
            "type":"traffic_light" if node_id in lights else "priority"})
    write_xml(nodes, HERE / "loop.nod.xml")

    edges = ET.Element("edges")
    pairs = [(edge,a,b,"Foundry Ring Road") for edge,a,b in zip(CW,RING,RING[1:]+RING[:1])]
    pairs += [(edge,a,b,"Foundry Ring Road") for edge,a,b in zip(CCW,REVERSE_RING,REVERSE_RING[1:]+REVERSE_RING[:1])]
    links = [("street","S1","C1"),("street","C1","N1"),
             ("street","S3","C3"),("street","C3","N3"),
             ("avenue","W","C1"),("avenue","C3","E"),
             ("bridge","C1","B_W"),("bridge","B_W","B_E"),("bridge","B_E","C3"),
             ("underpass","J1","C2"),("underpass","C2","J2"),
             ("spur","S_END","J1"),("spur","N_END","J2")]
    for category,a,b in links:
        for source,dest in ((a,b),(b,a)):
            short_source,short_dest = source.replace("B_",""),dest.replace("B_","")
            pairs.append((f"{category}_{short_source}_{short_dest}",source,dest,
                          "Foundry Viaduct" if category=="bridge" else category.title()+" Avenue"))
    for edge_id, source, dest, name in pairs:
        ET.SubElement(edges, "edge", {
            "id": edge_id, "from": source, "to": dest, "name": name,
            "numLanes": "2", "speed": "13.89" if edge_id.startswith("bridge_") else "11.11", "width": "3.2",
            "priority": "4", "spreadType": "right",
        })
    write_xml(edges, HERE / "loop.edg.xml")

    cmd = [str(NETCONVERT), "--node-files", "loop.nod.xml", "--edge-files", "loop.edg.xml",
           "--output-file", "loop.net.xml", "--tls.default-type", "static",
           "--junctions.corner-detail", "4", "--geometry.remove", "false",
           "--offset.disable-normalization", "true"]
    subprocess.run(cmd, cwd=HERE, check=True)


def make_vehicle_types(routes: ET.Element) -> None:
    colors = {
        "car": ("passenger", "4.5", "1.8", "11.11", "218,217,204", "vehicle_car.png", "passenger/sedan"),
        "ev": ("passenger", "4.5", "1.8", "11.11", "65,171,173", "vehicle_ev.png", "passenger/sedan"),
        "motorcycle": ("motorcycle", "2.2", "0.8", "10.0", "101,119,115", "vehicle_motorcycle.png", "motorcycle"),
        "small_passenger": ("passenger", "2.8", "1.4", "8.33", "179,145,76", "vehicle_small_passenger.png", "passenger/van"),
        "bus": ("bus", "10", "2.5", "8.33", "183,131,59", "vehicle_bus.png", "bus"),
        "truck": ("truck", "8", "2.4", "8.33", "130,92,73", "vehicle_truck.png", "truck"),
    }
    for type_id, (vclass, length, width, vmax, color, image, shape) in colors.items():
        ET.SubElement(routes, "vType", {
            "id": type_id, "vClass": vclass, "length": length, "width": width,
            "maxSpeed": vmax, "color": color, "accel": "2", "decel": "4.5",
            "sigma": "0.3", "minGap": "2.5", "guiShape": shape,
            "imgFile": f"assets/{image}",
        })


def build_routes() -> dict:
    root = ET.Element("routes")
    make_vehicle_types(root)
    ring_n = 24
    route_edges = {}
    for label,cycle in (("cw",CW),("ccw",CCW)):
        for index in range(len(cycle)):
            route_edges[f"route_{label}_{index:02d}"] = (cycle[index:]+cycle[:index])*ring_n
    cycles = {
        "viaduct_east": ["avenue_W_C1","bridge_C1_W","bridge_W_E","bridge_E_C3","avenue_C3_E",
            "cw_E_NE","cw_NE_N3","cw_N3_J2","cw_J2_N1","cw_N1_NW","cw_NW_W"],
        "viaduct_west": ["avenue_E_C3","bridge_C3_E","bridge_E_W","bridge_W_C1","avenue_C1_W",
            "ccw_W_NW","ccw_NW_N1","ccw_N1_J2","ccw_J2_N3","ccw_N3_NE","ccw_NE_E"],
        "underpass_north": ["underpass_J1_C2","underpass_C2_J2","cw_J2_N1","cw_N1_NW",
            "cw_NW_W","cw_W_SW","cw_SW_S1","cw_S1_J1"],
        "underpass_south": ["underpass_J2_C2","underpass_C2_J1","ccw_J1_S1","ccw_S1_SW",
            "ccw_SW_W","ccw_W_NW","ccw_NW_N1","ccw_N1_J2"],
        "market_street": ["street_S1_C1","street_C1_N1","cw_N1_NW","cw_NW_W","cw_W_SW","cw_SW_S1"],
        "foundry_street": ["street_S3_C3","street_C3_N3","ccw_N3_NE","ccw_NE_E","ccw_E_SE","ccw_SE_S3"],
    }
    route_edges.update({f"route_{name}":cycle*ring_n for name,cycle in cycles.items()})
    route_edges["route_south_spur"] = ["spur_S_END_J1"]+cycles["underpass_north"]*ring_n
    route_edges["route_north_spur"] = ["spur_N_END_J2"]+cycles["underpass_south"]*ring_n
    for route_id, edge_list in route_edges.items():
        ET.SubElement(root, "route", {"id": route_id, "edges": " ".join(edge_list)})
    counts = {"car": 56, "ev": 24, "motorcycle": 24, "small_passenger": 16, "bus": 12, "truck": 12}
    types = []
    while any(counts.values()):
        for type_id in counts:
            if counts[type_id]:
                types.append(type_id)
                counts[type_id] -= 1
    type_totals = {key: types.count(key) for key in ("car", "ev", "motorcycle", "small_passenger", "bus", "truck")}
    for index, type_id in enumerate(types):
        route_id = list(route_edges)[index % len(route_edges)]
        ET.SubElement(root, "vehicle", {
            "id": f"actor_{index+1:03d}", "type": type_id, "route": route_id,
            "depart": str(index * 1.5), "departLane": "best", "departSpeed": "desired",
        })
    write_xml(root, HERE / "loop.rou.xml")
    return {"vehicles_total": len(types), "vehicles_by_type": type_totals,
            "route_loops": ring_n, "depart_interval_s": 1.5, "depart_last_s": (len(types)-1)*1.5}


def asset_copy_and_ev() -> None:
    HERE.joinpath("assets").mkdir(exist_ok=True)
    asset_names = [
        "vehicle_car.png", "vehicle_motorcycle.png", "vehicle_small_passenger.png",
        "vehicle_bus.png", "vehicle_truck.png", "sample_pothole.png",
        "sample_construction.png", "sample_incident_car.png", "sample_debris.png",
        "rsu_mast.png", "sign_foundry.png", "sign_market.png", "tree_symbol.png",
    ]
    for name in asset_names:
        src = N1_ASSETS / name
        if not src.is_file():
            raise FileNotFoundError(f"N1 source artwork missing: {src}")
        shutil.copy2(src, HERE / "assets" / name)
        if name in ("sample_pothole.png", "sample_construction.png", "sample_incident_car.png"):
            # Native captures showed sand-colored backing around these sprites.
            # Give road fixtures an asphalt backing so they sit on the lane.
            path = HERE / "assets" / name
            with Image.open(path) as original:
                foreground = original.convert("RGBA")
                Image.alpha_composite(Image.new("RGBA", foreground.size, COLORS["road"]), foreground).save(path)
    base = Image.open(HERE / "assets" / "vehicle_car.png").convert("RGBA")
    d = ImageDraw.Draw(base)
    # Small grayscale EV mark on the roof; the rest remains tintable grayscale.
    d.rounded_rectangle((72, 184, 108, 230), radius=7, fill=(228,228,228,235), outline=(48,48,48,255), width=3)
    d.polygon([(93,190),(82,207),(91,207),(86,223),(101,203),(92,203)], fill=(65,65,65,255))
    base.save(HERE / "assets" / "vehicle_ev.png", optimize=True)


def read_net_and_interpolate():
    net = sumolib.net.readNet(str(HERE / "loop.net.xml"), withInternal=False)
    def lane_for(edge_id: str, lane_index: int = 0):
        return net.getEdge(edge_id).getLane(lane_index)
    def convert2D(edge_id: str, pos: float, lane_index: int = 0):
        """Interpolate on sumolib's compiled lane shape, equivalent to convert2D."""
        lane = lane_for(edge_id, lane_index)
        return positionAtShapeOffset(lane.getShape(), pos)
    return net, lane_for, convert2D


def point_polyline_distance(point, shape) -> float:
    x, y = point
    best = float("inf")
    for a, b in zip(shape, shape[1:]):
        ax, ay = a[:2]; bx, by = b[:2]
        dx, dy = bx-ax, by-ay
        t = max(0.0, min(1.0, ((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy or 1.0)))
        best = min(best, math.hypot(x-(ax+t*dx), y-(ay+t*dy)))
    return best


def tangent_angle(shape, pos: float, total_length: float) -> float:
    lo, hi = max(0.0, pos-0.75), min(total_length, pos+0.75)
    a = positionAtShapeOffset(shape, lo); b = positionAtShapeOffset(shape, hi)
    # SUMO shape angles are measured clockwise from north: east = 90 degrees.
    return round((math.degrees(math.atan2(b[0]-a[0], b[1]-a[1])) + 360) % 360, 2)


def hazard_locations(lane_for, convert2D):
    hazards = []
    dimensions = {"pothole":(1.1,.9,False,"sample_pothole.png"),
                  "construction":(3.0,6.0,True,"sample_construction.png"),
                  "accident_obstacle":(1.8,4.3,True,"sample_incident_car.png")}
    for hazard_id, kind, edge_id, local_pos in TARGETS:
        width,length,blocking,icon = dimensions[kind]
        lane = lane_for(edge_id, 0)
        if not length < local_pos < lane.getLength()-length:
            raise RuntimeError(f"Hazard {hazard_id} does not fit its compiled lane")
        position = convert2D(edge_id, local_pos, 0)
        angle = tangent_angle(lane.getShape(), local_pos, lane.getLength())
        hazards.append({
            "id": hazard_id, "kind": kind, "edge": edge_id, "lane_index": 0,
            "lane_id": lane.getID(), "pos_m": round(local_pos, 3),
            "position_m": [round(position[0], 3), round(position[1], 3)],
            "angle_deg": angle, "width_m": width, "length_m": length,
            "blocking": blocking, "icon": icon,
        })
    return hazards


def add_poly(root, object_id, points, color, layer, fill=True, line_width=None, kind="scene"):
    attrs = {"id": object_id, "type": kind, "color": color,
             "fill": "1" if fill else "0", "layer": str(layer),
             "shape": " ".join(f"{x:.2f},{y:.2f}" for x,y in points)}
    if line_width is not None:
        attrs["lineWidth"] = str(line_width)
    ET.SubElement(root, "poly", attrs)


def add_rect(root, object_id, x, y, w, h, color, layer, kind="scene"):
    return add_poly(root, object_id, [(x-w/2,y-h/2),(x+w/2,y-h/2),(x+w/2,y+h/2),(x-w/2,y+h/2)], color, layer, True, kind=kind)


def circle_shape(x, y, r, steps=28):
    return [(x+r*math.cos(2*math.pi*i/steps), y+r*math.sin(2*math.pi*i/steps)) for i in range(steps)]


def create_scenery(lane_for, convert2D, traffic_counts):
    root = ET.Element("additional")
    x0,y0,x1,y1 = BOUNDS
    add_poly(root,"district-ground",[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],BG,-30,kind="terrain")
    net,_,_ = read_net_and_interpolate()
    lane_shapes = [lane.getShape() for edge in net.getEdges() for lane in edge.getLanes()]
    def road_clear(cx,cy,w,h):
        samples=[(cx-w/2,cy-h/2),(cx+w/2,cy-h/2),(cx+w/2,cy+h/2),(cx-w/2,cy+h/2),(cx,cy)]
        return min(point_polyline_distance(p,shape) for p in samples for shape in lane_shapes)

    # Eight walkable-looking desert blocks, separated by the actual road grid.
    buildings_written = 0
    for col,cx in enumerate((180,600,1000,1420)):
        for row,cy in enumerate((220,710)):
            district = "market" if row else "foundry"
            add_rect(root,f"block-{col}-{row}",cx,cy,320,290,"#BBA98A",-25,"district-yard")
            for ix,dx in enumerate((-105,-35,35,105)):
                for iy,dy in enumerate((-65,5,75)):
                    x,y,w,h=cx+dx,cy+dy,46,26
                    if road_clear(x,y,w+8,h+8)<14 or 680<x<780:
                        continue
                    oid=f"{district}-{col}-{ix}-{iy}"
                    color=("#875843","#61605A","#B78C46")[(ix+iy)%3] if not row else ("#C18D40","#875843")[(ix+iy)%2]
                    add_rect(root,oid+"-apron",x,y,w+8,h+9,"#D1C1A2",-17,"loading-apron")
                    add_rect(root,oid+"-shadow",x+4,y-4,w,h,"#8E7C60",-16,"roof-shadow")
                    add_rect(root,oid+"-roof",x,y,w,h,color,-12,"building")
                    for offset in (-w/3,0,w/3):
                        if row:
                            add_rect(root,oid+f"-awning-{offset}",x+offset,y+h/2-1,w/3-1,4,"#E0AE58" if ix%2 else "#B75D3D",-11,"market-awning")
                        else:
                            add_poly(root,oid+f"-seam-{offset}",[(x+offset,y-h/2+2),(x+offset,y+h/2-2)],"#C5A27A",-11,False,.7,"roof-seam")
                    if not row and ix%2==0:
                        add_poly(root,oid+"-vent",circle_shape(x+12,y,3),"#D0B994",-10,kind="roof-vent")
                        add_poly(root,oid+"-vent-mouth",circle_shape(x+12,y,1.8),"#343532",-9,kind="roof-vent")
                    buildings_written+=1

    # A seasonal wash makes the viaduct legible in the desert landscape.
    wash=[(701,100),(714,260),(705,450),(719,650),(705,870),(748,870),(755,650),(744,450),(754,260),(741,100)]
    add_poly(root,"seasonal-wash-bed",wash,"#B29D78",-21,kind="wash")
    add_poly(root,"seasonal-wash-water",[(720,120),(731,260),(723,450),(735,650),(723,850),(738,850),(744,650),(735,450),(743,260),(732,120)],"#6F8A83",-20,kind="seasonal-water")

    # Layered native polygons show the raised deck without creating a junction.
    add_rect(root,"viaduct-shadow",808,441,312,24,"#897C65",-1,"bridge-shadow")
    add_rect(root,"viaduct-deck",800,450,300,13.0,"#505352",5,"bridge-deck")
    for y in (442.8,457.2):
        add_poly(root,f"viaduct-rail-{y}",[(649,y),(951,y)],"#E4D3AF",7,False,1.1,"bridge-parapet")
        for x in range(650,951,20):
            add_rect(root,f"viaduct-post-{x}-{y}",x,y,2,3,"#B58D58",8,"bridge-post")
    for x in (650,950):
        add_poly(root,f"viaduct-joint-{x}",[(x,443.5),(x,456.5)],"#9E967F",6,False,.6,"bridge-joint")
    add_poly(root,"viaduct-centerline",[(650,450),(950,450)],"#E2DAC9",6,False,.2,"bridge-marking")
    for x in range(654,949,18):
        for y in (446.8,453.2):
            add_poly(root,f"viaduct-dash-{x}-{y}",[(x,y),(x+7,y)],"#E2DAC9",6,False,.2,"bridge-marking")
    for x in (660,940):
        for y in (432,468):
            add_rect(root,f"viaduct-pier-{x}-{y}",x,y,9,10,"#9F947D",-2,"bridge-pier")

    for oid,image,x,y,w,h,title in (
        ("district-sign-foundry","assets/sign_foundry.png",230,-165,330,55,"Foundry Works"),
        ("district-sign-market","assets/sign_market.png",2050,1050,620,95,"Foundry Market"),
    ):
        ET.SubElement(root,"poi",{"id":oid,"x":str(x),"y":str(y),"type":"district-sign","color":"#FFFFFF","layer":"20","name":title,"width":str(w),"height":str(h),"imgFile":image})
    ET.SubElement(root,"poi",{"id":"viaduct-title","x":"800","y":"490","type":"bridge-label","name":"FOUNDRY VIADUCT +8m","color":"#A77D42","layer":"12","width":"2","height":"2"})

    hazards=hazard_locations(lane_for,convert2D)
    for index,hazard in enumerate(hazards):
        x,y=hazard["position_m"]
        add_poly(root,"hazard_marker_"+hazard["id"],circle_shape(x,y,7),"#B79555",3,False,.08,"hazard-locator")
        ET.SubElement(root,"poi",{"id":"hazard_"+hazard["id"],"x":str(x),"y":str(y),"type":"scenario_"+hazard["kind"],"color":"#FFFFFF","layer":"15","name":f"{index+1:02d} {hazard['kind'].upper()}","width":str(hazard["width_m"]),"height":str(hazard["length_m"]),"angle":str(hazard["angle_deg"]),"imgFile":"assets/"+hazard["icon"]})

    rsus=[{"id":"RSU_"+name,"position_m":list(position)} for name,position in (
        ("SOUTH_GATE",(775,-22)),("FOUNDRY_EAST",(1220,39)),("EAST_GATE",(1683,480)),
        ("NORTH_GATE",(820,983)),("MARKET_WEST",(380,1014)),("WEST_GATE",(-93,450)),
        ("BRIDGE_WEST",(380,475)),("BRIDGE_EAST",(1220,475)),
    )]
    for rsu in rsus:
        x,y=rsu["position_m"]
        ET.SubElement(root,"poi",{"id":rsu["id"],"x":str(x),"y":str(y),"type":"roadside_unit_fixture","color":"#FFFFFF","layer":"14","name":rsu["id"],"width":"18","height":"14","imgFile":"assets/rsu_mast.png"})
        add_poly(root,"rsu-pad-"+rsu["id"],circle_shape(x,y,13),"#AA9876",-10,kind="rsu-plinth")

    for index,(x,y) in enumerate(( (x,y) for x in range(80,1580,80) for y in (405,495))):
        if road_clear(x,y,8,8)<12 or 680<x<780 or 630<x<970:
            continue
        ET.SubElement(root,"poi",{"id":f"olive-tree-{index:02d}","x":str(x),"y":str(y),"type":"olive_tree","color":"#FFFFFF","layer":"-8","name":"","width":"7","height":"6","imgFile":"assets/tree_symbol.png"})
    write_xml(root,HERE/"loop.scenery.add.xml")
    return hazards,rsus,buildings_written


def write_gui_and_config():
    view = ET.Element("viewsettings")
    scheme = ET.SubElement(view, "scheme", {"name":"Foundry Daylight"})
    ET.SubElement(scheme,"background",{"backgroundColor":"197,179,145","showGrid":"false"})
    edges = ET.SubElement(scheme,"edges",{"laneEdgeMode":"0","laneShowBorders":"true","showLinkDecals":"true",
        "showLinkRules":"true","widthExaggeration":"1","minSize":"0","disableHideByZoom":"true"})
    edge_colors=ET.SubElement(edges,"colorScheme",{"name":"uniform"})
    ET.SubElement(edge_colors,"entry",{"color":"70,71,71","threshold":"0","name":"road"})
    ET.SubElement(scheme,"vehicles",{"vehicleMode":"0","vehicleQuality":"3","vehicle_minSize":"0",
        "vehicle_exaggeration":"1.0","showBlinker":"true"})
    junctions=ET.SubElement(scheme,"junctions",{"junctionMode":"0","drawShape":"true","showLane2Lane":"false"})
    jc=ET.SubElement(junctions,"colorScheme",{"name":"uniform"})
    ET.SubElement(jc,"entry",{"color":"70,71,71","threshold":"0"})
    ET.SubElement(scheme,"pois",{"poi_minSize":"0","poi_exaggeration":"1","poiType_show":"false",
        "poiName_show":"true","poiName_size":"10","poiName_color":"69,62,52",
        "poiName_constantSize":"true","poiName_onlySelected":"false"})
    ET.SubElement(scheme,"polys",{"polyName_show":"false","polyType_show":"false"})
    ET.SubElement(view,"viewport",{"zoom":"45","x":"1130","y":"480"})
    ET.SubElement(view,"delay",{"value":"0"})
    write_xml(view,HERE/"loop.view.xml")

    cfg=ET.Element("configuration")
    inp=ET.SubElement(cfg,"input")
    ET.SubElement(inp,"net-file",{"value":"loop.net.xml"})
    ET.SubElement(inp,"route-files",{"value":"loop.rou.xml"})
    ET.SubElement(inp,"additional-files",{"value":"loop.scenery.add.xml"})
    ET.SubElement(inp,"gui-settings-file",{"value":"loop.view.xml"})
    time=ET.SubElement(cfg,"time")
    ET.SubElement(time,"begin",{"value":"0"});ET.SubElement(time,"end",{"value":"3600"});ET.SubElement(time,"step-length",{"value":"0.1"})
    proc=ET.SubElement(cfg,"processing")
    ET.SubElement(proc,"collision.action",{"value":"warn"})
    gui=ET.SubElement(cfg,"gui_only")
    ET.SubElement(gui,"start",{"value":"true"});ET.SubElement(gui,"quit-on-end",{"value":"true"})
    write_xml(cfg,HERE/"loop.sumocfg")


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    asset_copy_and_ev()
    write_geometry()
    traffic_counts=build_routes()
    write_gui_and_config()
    net,lane_for,convert2D=read_net_and_interpolate()
    hazards,rsus,building_count=create_scenery(lane_for,convert2D,traffic_counts)
    compiled=ET.parse(HERE/"loop.net.xml").getroot()
    lanes={lane.get("id"):float(lane.get("length")) for edge in compiled.findall("edge") for lane in edge.findall("lane")}
    connections={(c.get("from"),c.get("to"),c.get("fromLane"),c.get("toLane")):c for c in compiled.findall("connection")}
    main_length=sum(lane_for(edge).getLength() for edge in CW)
    for before,after in zip(CW,CW[1:]+CW[:1]):
        via=connections[(before,after,"0","0")].get("via")
        while via:
            main_length+=lanes[via]
            internal,lane_index=via.rsplit("_",1)
            via=connections[(internal,after,lane_index,"0")].get("via")
    net_hash=hashlib.sha256((HERE/"loop.net.xml").read_bytes()).hexdigest()
    scene={
        "scene":"Foundry District - 5 km Viaduct Loop",
        "nominal_loop_length_m":5000,
        "main_route_length_m":round(main_length,3),
        "bounds":BOUNDS,
        "main_route":CW,
        "opposite_route":CCW,
        "hazards":hazards,
        "rsus":rsus,
        "bridge":BRIDGE,
        "evidence_panel":PANEL,
        "counts":{**traffic_counts,"buildings":building_count,"rsus":len(rsus),"hazards":len(hazards),
                  "signalized_junctions":len(compiled.findall("tlLogic"))},
        "net_sha256":net_hash,
        "provenance":{"netconvert":str(NETCONVERT),"sumolib":sumolib.__file__,
                       "coordinates":"compiled lane shapes interpolated with positionAtShapeOffset"},
        "limits":["Scenery hazard POIs are visual fixtures; controller-owned stopped blockers provide occupancy.",
                   "The evidence panel is blank static ground reserved for a live controller image."],
    }
    (HERE/"scene.json").write_text(json.dumps(scene,indent=2)+"\n",encoding="utf-8")
    print(f"Built network {net.getEdges()[0].getID()}.. with SHA256 {net_hash}")
    print(f"Routes: {traffic_counts['vehicles_total']} vehicles; depart 0..{traffic_counts['depart_last_s']} s; route loops {traffic_counts['route_loops']}")
    print("Hazards:")
    for h in hazards:
        print(f"  {h['id']}: {h['edge']} lane {h['lane_index']} pos {h['pos_m']:.2f} m at {h['position_m']}")
    print(f"Outer loop including internal junction lanes: {main_length:.2f} m")
    print(f"Scenery: {building_count} roofs; {len(rsus)} RSUs; {len(list((HERE/'assets').glob('*.png')))} copied/generated PNG assets")


if __name__ == "__main__":
    main()
