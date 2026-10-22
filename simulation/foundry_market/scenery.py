"""Build the deterministic, native SUMO scenery for Foundry and Market.

Run explicitly with the project Python: python scenery.py. Outputs are local to
this directory; no launcher or simulation process runs this generator.
"""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
NET = HERE / "foundry.net.xml"
ASSETS = HERE / "assets"
OUT_XML = HERE / "foundry.scenery.add.xml"
OUT_MANIFEST = HERE / "scene.manifest.json"
PALETTE = {
    "sand": "#C5B391", "dust": "#A99A80", "road": "#464747",
    "curb": "#D8CDB7", "rust": "#875843", "steel": "#61605A",
    "ochre": "#C18D40", "olive": "#72785A", "cream": "#E5D6B6",
}


def rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def make_icon(name: str, draw_fn) -> None:
    im = Image.new("RGBA", (160, 112), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    draw_fn(d)
    im.save(ASSETS / f"{name}.png", optimize=True)


def font(size: int, bold: bool = False):
    paths = [r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
             r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"]
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_signs() -> None:
    """Create readable transparent bitmap labels; these do not rely on POI text."""
    im = Image.new("RGBA", (1500, 250), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((5, 8, 1494, 242), radius=22, fill=(48, 47, 42, 232), outline=(224, 203, 163, 255), width=8)
    d.rectangle((25, 28, 65, 222), fill=rgba("#C18D40"))
    d.text((100, 38), "FOUNDRY WORKS", font=font(93, True), fill=rgba("#F0E2C5"))
    d.text((104, 153), "IRON   ·   REPAIR   ·   FREIGHT", font=font(43, True), fill=rgba("#D5B986"))
    im.save(ASSETS / "sign_foundry.png", optimize=True)

    im = Image.new("RGBA", (1300, 250), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((5, 8, 1294, 242), radius=22, fill=(58, 48, 38, 232), outline=(217, 177, 108, 255), width=8)
    for x in range(42, 1250, 122):
        d.polygon([(x, 31), (x+53, 31), (x+77, 89), (x+29, 89)], fill=rgba("#C18D40"))
        d.line((x+30, 95, x+96, 95), fill=rgba("#E5D6B6"), width=9)
    d.text((70, 118), "FOUNDRY MARKET", font=font(82, True), fill=rgba("#F0E2C5"))
    im.save(ASSETS / "sign_market.png", optimize=True)

    # A large world-space plaque with actual sample icons and category copy.
    im = Image.new("RGBA", (1800, 620), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((4, 4, 1796, 616), radius=28, fill=(47, 46, 41, 240), outline=(226, 209, 174, 255), width=8)
    d.text((42, 22), "VISUAL SAMPLES  -  N1", font=font(66, True), fill=rgba("#F2E5C9"))
    d.text((44, 95), "SCENERY ONLY   /   NO TRAFFIC EFFECT", font=font(30, True), fill=rgba("#D7B779"))
    entries = [
        ("Pothole", "sample_pothole.png", "visual sample"),
        ("Puddle", "sample_puddle.png", "comparison"),
        ("Manhole", "sample_manhole.png", "comparison"),
        ("Asphalt patch", "sample_asphalt_patch.png", "comparison"),
        ("Fallen tree", "sample_fallen_tree.png", "visual sample"),
        ("Debris", "sample_debris.png", "visual sample"),
        ("Incident car", "sample_incident_car.png", "visual sample"),
        ("Construction", "sample_construction.png", "visual sample"),
    ]
    cell_w, cell_h = 420, 205
    for i, (label, icon_name, role) in enumerate(entries):
        col, row = i % 4, i // 4
        x, y = 30 + col * 440, 150 + row * 220
        d.rounded_rectangle((x, y, x+cell_w, y+cell_h), radius=15,
                            fill=(83, 75, 62, 210), outline=(142, 126, 99, 255), width=3)
        icon = Image.open(ASSETS / icon_name).convert("RGBA")
        icon.thumbnail((180, 130))
        im.alpha_composite(icon, (x+10, y+30))
        d.text((x+200, y+34), label, font=font(33, True), fill=rgba("#F0E2C5"))
        d.text((x+200, y+91), role.upper(), font=font(22, True), fill=rgba("#D7B779"))
    im.save(ASSETS / "n1_sample_legend.png", optimize=True)


def draw_vehicle_sprites() -> None:
    """Make front-up grayscale top-down sprites for SUMO raster vehicle mode."""
    specs = {
        "vehicle_car.png": (180, 450, "car"),       # 4.5 m / 1.8 m = 2.5
        "vehicle_motorcycle.png": (144, 396, "bike"),# 2.2 m / 0.8 m = 2.75
        "vehicle_small_passenger.png": (180, 360, "van"), # 2.8 m / 1.4 m = 2.0
        "vehicle_bus.png": (150, 600, "bus"),       # 10 m / 2.5 m = 4.0
        "vehicle_truck.png": (180, 600, "truck"),   # 8 m / 2.4 m = 3.33
    }
    for filename, (w, h, kind) in specs.items():
        scale = 2
        im = Image.new("RGBA", (w*scale, h*scale), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        def box(coords): return tuple(int(v*scale) for v in coords)
        def line(coords, fill, width): d.line([tuple(int(v*scale) for v in pt) for pt in coords], fill=fill, width=int(width*scale), joint="curve")
        def rounded(coords, radius, fill, outline=None, width=1):
            d.rounded_rectangle(box(coords), radius=int(radius*scale), fill=fill, outline=outline,
                                width=max(1, int(width*scale)) if outline else 1)
        def rect(coords, fill, outline=None, width=1):
            d.rectangle(box(coords), fill=fill, outline=outline,
                        width=max(1, int(width*scale)) if outline else 1)
        def ellipse(coords, fill, outline=None, width=1):
            d.ellipse(box(coords), fill=fill, outline=outline,
                      width=max(1, int(width*scale)) if outline else 1)

        # Paint a soft monochrome ground shadow and four black rubber wheels.
        if kind in ("car", "van"):
            bw = 112 if kind == "car" else 122
            x0, x1 = (w-bw)/2, (w+bw)/2
            y0, y1 = 20, h-18
            body = (174,174,174,255) if kind == "car" else (164,164,164,255)
            shadow = (24,24,24,82)
            rounded((x0+6,y0+7,x1+7,y1+7), 38 if kind=="car" else 28, shadow)
            # Wheel silhouettes sit just outside the body shell.
            wheel_w = 19 if kind=="car" else 21
            for wy in (y0+66, y1-126):
                rounded((x0-wheel_w/2,wy,x0+wheel_w/2,wy+70), 7, (22,22,22,255))
                rounded((x1-wheel_w/2,wy,x1+wheel_w/2,wy+70), 7, (22,22,22,255))
            rounded((x0,y0,x1,y1), 39 if kind=="car" else 27, body, (75,75,75,255), 4)
            # Front/nose is at the top. Lights and windshield are distinct.
            rounded((x0+16,y0+8,x1-16,y0+42), 9, (218,218,218,255))
            for lx in (x0+10,x1-25):
                rounded((lx,y0+14,lx+15,y0+27), 4, (238,238,238,255))
            if kind == "car":
                rounded((x0+19,y0+82,x1-19,y0+164), 18, (65,65,65,255), (218,218,218,255), 3)
                rounded((x0+22,y0+179,x1-22,y0+269), 18, (112,112,112,255), (211,211,211,255), 3)
                rounded((x0+20,y0+283,x1-20,y0+339), 12, (61,61,61,255), (205,205,205,255), 3)
                line([(x0+13,y0+171),(x1-13,y0+171)], (235,235,235,255), 3)
            else:
                # Squared van cab, broad side windows, visible cargo roof.
                rounded((x0+13,y0+62,x1-13,y0+139), 14, (57,57,57,255), (218,218,218,255), 3)
                rounded((x0+12,y0+153,x1-12,y1-45), 15, (137,137,137,255), (218,218,218,255), 3)
                for wy in (y0+168, y0+230, y0+292):
                    line([(x0+14,wy),(x1-14,wy)], (197,197,197,255), 3)
            # Tail lamps and bumper help distinguish the rear.
            for lx in (x0+10,x1-25):
                rounded((lx,y1-29,lx+15,y1-13), 3, (75,75,75,255))
            line([(x0+17,y1-40),(x1-17,y1-40)], (229,229,229,255), 4)

        elif kind == "bike":
            cx=w/2
            ellipse((cx-17,13,cx+17,105), (24,24,24,255))
            ellipse((cx-17,h-105,cx+17,h-13), (24,24,24,255))
            ellipse((cx-10,25,cx+10,93), (145,145,145,255), (56,56,56,255), 4)
            ellipse((cx-10,h-93,cx+10,h-25), (145,145,145,255), (56,56,56,255), 4)
            # Front fork, handlebars and headlamp establish the upward heading.
            line([(cx-22,91),(cx,124),(cx+22,91)], (72,72,72,255), 8)
            line([(cx-37,95),(cx+37,95)], (42,42,42,255), 7)
            ellipse((cx-15,106,cx+15,136), (220,220,220,255), (52,52,52,255), 4)
            rounded((cx-28,136,cx+28,228), 20, (117,117,117,255), (44,44,44,255), 4)
            rounded((cx-22,226,cx+22,303), 16, (71,71,71,255), (35,35,35,255), 4)
            rounded((cx-34,294,cx+34,329), 13, (96,96,96,255), (34,34,34,255), 4)
            line([(cx-19,330),(cx-10,h-99)], (58,58,58,255), 5)
            line([(cx+19,330),(cx+10,h-99)], (58,58,58,255), 5)
            ellipse((cx-16,106,cx+16,139), (230,230,230,255))
            ellipse((cx-12,h-72,cx+12,h-50), (207,207,207,255))

        elif kind == "bus":
            x0,x1=22,w-22; y0,y1=12,h-12
            rounded((x0+5,y0+8,x1+5,y1+8), 25, (22,22,22,80))
            for wy in (y0+70,y1-147):
                rounded((x0-10,wy,x0+9,wy+84), 6, (20,20,20,255))
                rounded((x1-9,wy,x1+10,wy+84), 6, (20,20,20,255))
            rounded((x0,y0,x1,y1), 24, (171,171,171,255), (68,68,68,255), 4)
            rounded((x0+9,y0+15,x1-9,y0+76), 12, (53,53,53,255), (228,228,228,255), 3)
            # Repeated broad passenger windows along both sides.
            for wy in range(y0+93,y1-178,61):
                rounded((x0+8,wy,x0+29,wy+49), 7, (57,57,57,255), (214,214,214,255), 2)
                rounded((x1-29,wy,x1-8,wy+49), 7, (57,57,57,255), (214,214,214,255), 2)
            rounded((x0+13,y1-160,x1-13,y1-87), 12, (67,67,67,255), (222,222,222,255), 3)
            line([(x0+8,y1-79),(x1-8,y1-79)], (232,232,232,255), 4)
            for lx in (x0+5,x1-17): rounded((lx,y0+8,lx+12,y0+18),3,(230,230,230,255))
            for lx in (x0+5,x1-17): rounded((lx,y1-18,lx+12,y1-8),3,(75,75,75,255))

        else: # box truck: cab at the top, ribbed cargo bed behind it
            x0,x1=22,w-22; y0,y1=12,h-12
            rounded((x0+5,y0+8,x1+5,y1+8), 18, (22,22,22,82))
            for wy in (y0+80,y1-152):
                rounded((x0-10,wy,x0+9,wy+78), 6, (20,20,20,255))
                rounded((x1-9,wy,x1+10,wy+78), 6, (20,20,20,255))
            rounded((x0,y0+5,x1,y1), 16, (152,152,152,255), (59,59,59,255), 4)
            # Cargo section, separated from the forward cab by a hard bulkhead.
            rounded((x0+8,y0+22,x1-8,y0+294), 12, (180,180,180,255), (73,73,73,255), 3)
            for wy in range(y0+54,y0+278,36):
                line([(x0+12,wy),(x1-12,wy)], (121,121,121,255), 3)
            line([(x0+7,y0+302),(x1-7,y0+302)], (45,45,45,255), 7)
            rounded((x0+13,y0+316,x1-13,y0+405), 15, (110,110,110,255), (47,47,47,255), 3)
            rounded((x0+21,y0+326,x1-21,y0+379), 12, (52,52,52,255), (223,223,223,255), 3)
            for lx in (x0+6,x1-18): rounded((lx,y0+15,lx+13,y0+27),3,(235,235,235,255))
            line([(x0+10,y0+28),(x1-10,y0+28)], (218,218,218,255), 4)
            for lx in (x0+5,x1-17): rounded((lx,y1-20,lx+12,y1-9),3,(70,70,70,255))

        im = im.resize((w,h), Image.Resampling.LANCZOS)
        im.save(ASSETS / filename, optimize=True)


def draw_icons() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    def pothole(d):
        d.ellipse((17, 20, 145, 94), fill=(69, 65, 57, 170))
        d.polygon([(22, 53), (38, 30), (69, 24), (88, 15), (121, 28), (145, 48), (132, 77), (102, 88), (73, 82), (44, 96), (23, 77)], fill=rgba("#B99467"))
        d.polygon([(31, 55), (47, 37), (70, 32), (91, 24), (117, 35), (134, 51), (123, 69), (101, 77), (76, 72), (49, 83), (33, 70)], fill=rgba("#292A29"))
        d.line([(42, 35), (31, 20), (26, 15)], fill=rgba("#554B3D"), width=4)
        d.line([(111, 32), (128, 17), (139, 17)], fill=rgba("#554B3D"), width=4)
        d.line([(60, 78), (53, 97), (42, 104)], fill=rgba("#554B3D"), width=3)
        d.ellipse((49, 45, 70, 58), fill=(123, 111, 92, 80))

    def puddle(d):
        d.ellipse((14, 24, 146, 91), fill=(62, 68, 67, 110))
        d.polygon([(19, 52), (35, 31), (59, 34), (76, 21), (100, 27), (123, 23), (144, 44), (135, 68), (115, 82), (88, 77), (64, 89), (36, 80)], fill=(92, 119, 126, 220))
        d.ellipse((32, 36, 133, 75), outline=(180, 197, 193, 190), width=3)
        d.line([(52, 48), (85, 39), (104, 40)], fill=(225, 229, 214, 200), width=4)
        d.line([(84, 67), (114, 59)], fill=(161, 188, 184, 200), width=3)

    def manhole(d):
        d.ellipse((32, 9, 128, 103), fill=(37, 39, 38, 170))
        d.ellipse((27, 6, 123, 100), fill=rgba("#383B39"), outline=rgba("#B0A58D"), width=5)
        d.ellipse((39, 18, 111, 88), fill=rgba("#686B65"), outline=rgba("#242725"), width=4)
        for y in (34, 45, 56, 67, 78):
            d.line((49, y, 101, y), fill=rgba("#303432"), width=3)
        d.arc((42, 20, 107, 85), 205, 330, fill=rgba("#D3C5A9"), width=3)

    def patch(d):
        d.polygon([(22, 25), (126, 18), (145, 38), (136, 83), (112, 96), (26, 88), (14, 63)], fill=(44, 44, 41, 145))
        d.polygon([(23, 22), (125, 16), (139, 34), (132, 78), (111, 91), (25, 84), (17, 61)], fill=rgba("#555650"))
        d.line((29, 33, 121, 27), fill=rgba("#8C897A"), width=3)
        d.line((27, 73, 112, 81), fill=rgba("#6F6E65"), width=3)

    def fallen_tree(d):
        d.line((21, 84, 138, 32), fill=(39, 35, 29, 150), width=17)
        d.line((17, 77, 134, 25), fill=rgba("#68472F"), width=12)
        d.line((70, 53, 55, 22), fill=rgba("#805A3C"), width=7)
        d.line((95, 42, 113, 70), fill=rgba("#805A3C"), width=7)
        for box, col in [((4, 9, 52, 44), "#72785A"), ((44, 1, 88, 34), "#858567"), ((109, 4, 157, 39), "#747A5C"), ((116, 59, 157, 99), "#858567"), ((31, 70, 71, 109), "#747A5C")]:
            d.ellipse(box, fill=rgba(col))
        d.ellipse((10, 13, 45, 39), fill=(172, 170, 127, 105))

    def debris(d):
        d.polygon([(25, 82), (35, 51), (53, 62), (60, 31), (78, 43), (93, 19), (105, 49), (124, 38), (140, 76), (112, 91), (74, 83), (47, 97)], fill=(37, 35, 32, 150))
        d.polygon([(20, 74), (31, 44), (50, 55), (56, 25), (73, 39), (89, 13), (102, 44), (123, 31), (137, 69), (107, 84), (72, 76), (45, 90)], fill=rgba("#817663"))
        d.polygon([(49, 58), (56, 38), (72, 52), (66, 71)], fill=rgba("#B3986B"))
        d.polygon([(91, 55), (108, 43), (116, 68), (99, 76)], fill=rgba("#554C40"))

    def incident_car(d):
        d.polygon([(18, 43), (31, 28), (52, 24), (67, 17), (109, 19), (129, 30), (142, 43), (142, 69), (127, 81), (34, 82), (18, 70)], fill=(32, 31, 29, 155))
        d.polygon([(15, 37), (28, 23), (50, 20), (67, 13), (108, 16), (127, 26), (139, 38), (139, 64), (124, 76), (31, 77), (15, 65)], fill=rgba("#A44736"))
        d.polygon([(50, 25), (69, 18), (105, 21), (119, 31), (111, 42), (44, 41)], fill=rgba("#B9C4BA"))
        d.polygon([(43, 48), (116, 49), (123, 64), (37, 64)], fill=rgba("#784536"))
        for x in (37, 115):
            d.rounded_rectangle((x, 27, x+9, 38), radius=3, fill=rgba("#252625"))
            d.rounded_rectangle((x, 63, x+9, 73), radius=3, fill=rgba("#252625"))
        d.line((63, 17, 83, 39), fill=rgba("#E5C8A0"), width=3)
        d.line((28, 84, 14, 100), fill=rgba("#333230"), width=4)

    def construction(d):
        # orange cones and a clearly striped, low barrier
        for x in (24, 58, 92, 126):
            d.polygon([(x, 63), (x+8, 32), (x+16, 63)], fill=rgba("#D9792F"))
            d.line((x+4, 49, x+12, 49), fill=rgba("#F1D4A4"), width=4)
            d.line((x-3, 65, x+19, 65), fill=rgba("#514638"), width=4)
        d.rounded_rectangle((19, 76, 141, 91), radius=3, fill=rgba("#B44E35"), outline=rgba("#413C34"), width=3)
        for x in range(24, 136, 20):
            d.line((x, 78, x+10, 89), fill=rgba("#E8D5AE"), width=5)

    def rsu(d):
        d.ellipse((44, 82, 111, 105), fill=(32, 31, 28, 115))
        d.line((78, 23, 78, 87), fill=rgba("#343A36"), width=7)
        d.line((63, 48, 94, 48), fill=rgba("#343A36"), width=5)
        d.rounded_rectangle((59, 61, 98, 88), radius=4, fill=rgba("#D8CDB7"), outline=rgba("#454944"), width=4)
        d.rectangle((68, 68, 89, 82), fill=rgba("#79827A"))
        d.arc((44, 6, 111, 71), 215, 325, fill=rgba("#C18D40"), width=4)
        d.arc((54, 14, 101, 61), 215, 325, fill=rgba("#C18D40"), width=4)

    for name, fn in (("sample_pothole", pothole), ("sample_puddle", puddle),
                     ("sample_manhole", manhole), ("sample_asphalt_patch", patch),
                     ("sample_fallen_tree", fallen_tree), ("sample_debris", debris),
                     ("sample_incident_car", incident_car), ("sample_construction", construction),
                     ("rsu_mast", rsu)):
        make_icon(name, fn)


def read_network():
    root = ET.parse(NET).getroot()
    segments, junctions = [], []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        for lane in edge.findall("lane"):
            pts = [tuple(map(float, p.split(","))) for p in lane.get("shape", "").split()]
            segments.extend(zip(pts, pts[1:]))
    for j in root.findall("junction"):
        if j.get("type") != "internal":
            junctions.append((j.get("id", ""), float(j.get("x", 0)), float(j.get("y", 0))))
    return root, segments, junctions


def road_distance(x: float, y: float, segments) -> float:
    best = float("inf")
    for (ax, ay), (bx, by) in segments:
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x-ax)*dx + (y-ay)*dy) / (dx*dx + dy*dy or 1.0)))
        best = min(best, math.hypot(x-(ax+t*dx), y-(ay+t*dy)))
    return best


def main() -> None:
    if not NET.is_file():
        raise FileNotFoundError(f"Expected compiled native network: {NET}")
    net_root, segments, junctions = read_network()
    if not segments:
        raise RuntimeError("Compiled network contains no external lane geometry")
    draw_icons()
    draw_vehicle_sprites()

    root = ET.Element("additional")
    ET.SubElement(root, "poly", {"id": "district-ground", "type": "terrain", "color": PALETTE["sand"],
        "fill": "1", "layer": "-30", "shape": "-20,-20 940,-20 940,790 -20,790"})
    # Compose each district independently so no early lot consumes the global
    # building budget. The yard treatment is broken into small aprons instead
    # of broad, empty rectangles.
    lots = [
        {"name": "foundry-north", "kind": "foundry", "box": (292, 38, 452, 205), "count": 5, "style": "industrial"},
        {"name": "foundry-east", "kind": "foundry", "box": (465, 28, 640, 215), "count": 6, "style": "industrial"},
        {"name": "market-west", "kind": "market", "box": (38, 105, 205, 230), "count": 5, "style": "shop"},
        {"name": "market-west-road", "kind": "market", "box": (70, 390, 215, 480), "count": 5, "style": "shop"},
        {"name": "market-southwest", "kind": "market", "box": (110, 540, 270, 700), "count": 5, "style": "shop"},
        {"name": "market-south", "kind": "market", "box": (275, 535, 450, 700), "count": 10, "style": "shop"},
        {"name": "market-southeast", "kind": "market", "box": (465, 560, 620, 735), "count": 9, "style": "shop"},
        {"name": "service-northeast", "kind": "service", "box": (700, 30, 880, 170), "count": 7, "style": "shed"},
        {"name": "service-southeast", "kind": "service", "box": (710, 530, 890, 660), "count": 6, "style": "shed"},
    ]
    roofs = []
    for li, lot in enumerate(lots):
        x0, y0, x1, y1 = lot["box"]
        style = lot["style"]
        step_x, step_y = (42, 38) if style == "industrial" else ((30, 31) if style == "shop" else (36, 34))
        w, h = (36, 23) if style == "industrial" else ((19, 13) if style == "shop" else (26, 18))
        possible = []
        for row, yy in enumerate(range(y0 + h//2 + 9, y1 - h//2 - 4, step_y)):
            for col, xx in enumerate(range(x0 + w//2 + 8, x1 - w//2 - 5, step_x)):
                cx, cy = xx, yy
                if road_distance(cx, cy, segments) < math.hypot(w*1.24, h*1.24)/2 + 9:
                    continue
                if any(math.hypot(cx-jx, cy-jy) < 28 for _, jx, jy in junctions):
                    continue
                reserved_tanks = ((430, 140, 11), (290, 35, 9))
                if style == "industrial" and any(math.hypot(cx-tx, cy-ty) < rad+max(w,h)/2+8 for tx,ty,rad in reserved_tanks):
                    continue
                possible.append((cx, cy, w, h, lot["kind"], style, lot["name"]))
        # Reserve the whole district's budget using stable row-major placement.
        roofs.extend(possible[:lot["count"]])

    roofcolors = ("#875843", "#8D6049", "#61605A", "#9A7658", "#B07A47")
    for i, (cx, cy, w, h, kind, style, lot_name) in enumerate(roofs):
        color = ("#875843" if i % 3 else "#61605A") if style == "industrial" else ("#C18D40" if style == "shop" else roofcolors[i % len(roofcolors)])
        # Small aprons hug each structure; continuous flat yard blocks are omitted.
        aw, ah = w*1.42, h*1.52
        ET.SubElement(root, "poly", {"id": f"shed-{lot_name}-{i:02d}-apron", "type": "loading-apron",
            "color": "#B8A788", "fill": "1", "layer": "-17",
            "shape": f"{cx-aw/2},{cy-ah/2} {cx+aw/2},{cy-ah/2} {cx+aw/2},{cy+ah/2} {cx-aw/2},{cy+ah/2}"})
        ET.SubElement(root, "poly", {"id": f"shed-{lot_name}-{i:02d}-shadow", "type": "building-shadow",
            "color": "#6B5C49", "fill": "1", "layer": "-16",
            "shape": f"{cx-w/2+2},{cy-h/2+3} {cx+w/2+2},{cy-h/2+3} {cx+w/2+2},{cy+h/2+3} {cx-w/2+2},{cy+h/2+3}"})
        ET.SubElement(root, "poly", {"id": f"shed-{lot_name}-{i:02d}-roof", "type": "building",
            "color": color, "fill": "1", "layer": "-12", "name": lot_name,
            "shape": f"{cx-w/2},{cy-h/2} {cx+w/2},{cy-h/2} {cx+w/2},{cy+h/2} {cx-w/2},{cy+h/2}"})
        if style == "shop":
            # Cloth awning and a little recessed shopfront along the south edge.
            awn = "#B75D3D" if i % 2 else "#E0AE58"
            ET.SubElement(root, "poly", {"id": f"shop-{lot_name}-{i:02d}-awning", "type": "market-awning",
                "color": awn, "fill": "1", "layer": "-11",
                "shape": f"{cx-w/2+1},{cy+h/2-3} {cx+w/2-1},{cy+h/2-3} {cx+w/2-1},{cy+h/2+1} {cx-w/2+1},{cy+h/2+1}"})
            ET.SubElement(root, "poly", {"id": f"shop-{lot_name}-{i:02d}-door", "type": "shopfront",
                "color": "#D8CDB7", "fill": "0", "lineWidth": "0.5", "layer": "-10",
                "shape": f"{cx-1},{cy+h/2-1} {cx+1},{cy+h/2-1}"})
        else:
            seam_color = "#C5A27A" if style == "industrial" else "#D0B68B"
            for k in (1, 2):
                x = cx-w/2+k*w/3
                ET.SubElement(root, "poly", {"id": f"shed-{lot_name}-{i:02d}-seam-{k}", "type": "roof-detail",
                    "color": seam_color, "fill": "0", "lineWidth": "0.5", "layer": "-11",
                    "shape": f"{x},{cy-h/2+1} {x},{cy+h/2-1}"})

    # A few tanks and stack footprints make the foundry district read from above.
    def circle_shape(cx, cy, rx, ry, steps=28):
        return " ".join(f"{cx+rx*math.cos(2*math.pi*k/steps):.2f},{cy+ry*math.sin(2*math.pi*k/steps):.2f}" for k in range(steps))

    landmark_points = [(430, 140, 11), (290, 35, 9)]
    tank_count = 0
    for k, (x, y, rad) in enumerate(landmark_points):
        if road_distance(x, y, segments) < rad + 8 or any(math.hypot(x-cx, y-cy) < rad+math.hypot(w,h)/2+4 for cx,cy,w,h,*_ in roofs):
            continue
        tank_count += 1
        ET.SubElement(root, "poly", {"id": f"foundry-tank-{k:02d}-shadow", "type": "tank-shadow",
            "color": "#6D5D49", "fill": "1", "layer": "-15", "shape": circle_shape(x+2, y+3, rad+1, rad+1)})
        ET.SubElement(root, "poly", {"id": f"foundry-tank-{k:02d}-rim", "type": "storage-tank",
            "color": "#CDBB99", "fill": "1", "layer": "-13", "shape": circle_shape(x, y, rad+1, rad+1)})
        ET.SubElement(root, "poly", {"id": f"foundry-tank-{k:02d}-top", "type": "storage-tank-roof",
            "color": "#74746B", "fill": "1", "layer": "-12", "shape": circle_shape(x, y, rad-1, rad-1)})
        ET.SubElement(root, "poly", {"id": f"foundry-tank-{k:02d}-seam", "type": "tank-seam",
            "color": "#C9B58D", "fill": "0", "lineWidth": "0.6", "layer": "-11",
            "shape": f"{x-rad+2},{y} {x+rad-2},{y}"})
    # Put three compact stacks directly on foundry roofs, with pads and mouths.
    stack_count = 0
    industrial_roofs = [(cx,cy,w,h) for cx,cy,w,h,_,style,_ in roofs if style == "industrial"]
    for k, (x, y, w, h) in enumerate(industrial_roofs[:3]):
        if any(math.hypot(x-tx, y-ty) < rad+12 for tx,ty,rad in landmark_points):
            continue
        stack_count += 1
        ET.SubElement(root, "poly", {"id": f"foundry-stack-{k:02d}-pad", "type": "chimney-pad",
            "color": "#D0C09F", "fill": "1", "layer": "-15",
            "shape": f"{x-7},{y-7} {x+7},{y-7} {x+7},{y+7} {x-7},{y+7}"})
        ET.SubElement(root, "poly", {"id": f"foundry-stack-{k:02d}", "type": "chimney-footprint",
            "color": "#815B43", "fill": "1", "layer": "-12",
            "shape": f"{x-4.5},{y-4.5} {x+4.5},{y-4.5} {x+4.5},{y+4.5} {x-4.5},{y+4.5}"})
        ET.SubElement(root, "poly", {"id": f"foundry-stack-{k:02d}-mouth", "type": "chimney-mouth",
            "color": "#393A36", "fill": "1", "layer": "-11", "shape": circle_shape(x, y, 2.7, 2.7, 16)})

    # District names are rendered into transparent local PNGs so they remain
    # visible even when native POI-name display is turned off.
    draw_signs()
    # Full-overview sign proportions: roughly 200 world-meters wide, placed on
    # clear ground beside their respective low-y foundry and high-y market lots.
    signs = [("district-sign-foundry", "assets/sign_foundry.png", 410, 209, 200, 34),
             ("district-sign-market", "assets/sign_market.png", 440, 800, 200, 38)]
    for sid, image, x, y, w, h in signs:
        ET.SubElement(root, "poi", {"id": sid, "x": str(x), "y": str(y), "type": "district-sign",
            "color": "#FFFFFF", "layer": "20", "name": "", "width": str(w), "height": str(h), "imgFile": image})

    # Lot boundary lines, planted pockets and sparse top-down olive trees.
    trees = []
    for i, (x, y) in enumerate(((35, 292), (51, 315), (35, 340), (52, 365), (35, 390), (52, 416),
                                  (875, 250), (858, 273), (875, 298), (857, 323), (875, 347), (858, 374),
                                  (35, 700), (875, 705), (198, 565), (198, 610), (198, 655),
                                  (650, 110), (667, 134), (650, 158), (666, 181),
                                  (277, 102), (294, 128), (278, 153), (294, 179),
                                  (635, 570), (650, 594), (635, 620), (650, 646))):
        if road_distance(x, y, segments) < 15:
            continue
        trees.append((x, y))
        ET.SubElement(root, "poly", {"id": f"tree-bed-{i:02d}", "type": "planting", "color": "#8C846B",
            "fill": "1", "layer": "-17", "shape": f"{x-3},{y-2} {x+3},{y-2} {x+3},{y+2} {x-3},{y+2}"})
        # Native compact tree silhouette rather than a huge map marker.
        ET.SubElement(root, "poi", {"id": f"olive-tree-{i:02d}", "x": str(x), "y": str(y),
            "type": "olive_tree", "color": "#FFFFFF", "layer": "-8", "name": "Olive tree",
            "width": "4.2", "height": "3.2", "imgFile": "assets/tree_symbol.png"})

    # One compact reusable tree symbol.
    im = Image.new("RGBA", (112, 96), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse((20, 21, 94, 83), fill=(41, 38, 31, 105))
    d.ellipse((25, 14, 87, 74), fill=rgba("#62694E"), outline=rgba("#494A3C"), width=3)
    for box in ((29, 17, 57, 46), (54, 18, 82, 44), (38, 39, 75, 67)):
        d.ellipse(box, fill=rgba("#828467"))
    d.ellipse((34, 20, 52, 35), fill=(194, 174, 126, 100))
    im.save(ASSETS / "tree_symbol.png", optimize=True)

    # Find a true roadside candidate near each compiled junction while leaving
    # both a measurable lane gap and a clear junction radius.
    rsu_rows = []
    for idx, (jid, jx, jy) in enumerate(junctions):
        choices = []
        for radius in range(19, 42):
            for deg in range(0, 360, 5):
                a = math.radians(deg)
                x0, y0 = jx+radius*math.cos(a), jy+radius*math.sin(a)
                lane_gap = road_distance(x0, y0, segments)
                if lane_gap < 7 or lane_gap > 24 or x0 < 5 or y0 < 5 or x0 > 912 or y0 > 763:
                    continue
                choices.append((abs(lane_gap-12)+abs(radius-26)*.35, x0, y0, lane_gap, radius))
        if not choices:
            raise RuntimeError(f"No clear roadside placement found for junction {jid}")
        _, x, y, lane_gap, junction_gap = min(choices)
        rsu_rows.append({"id": f"rsu-{jid}", "junction": jid, "x": round(x, 3), "y": round(y, 3), "dimensions_m": [3.2, 2.4]})
        ET.SubElement(root, "poi", {"id": f"rsu-{jid}", "x": f"{x:.3f}", "y": f"{y:.3f}",
            "type": "roadside_unit_fixture", "color": "#FFFFFF", "layer": "14", "name": f"RSU {jid} · visual mast",
            "width": "3.2", "height": "2.4", "imgFile": "assets/rsu_mast.png"})
        rsu_rows[-1]["lane_clearance_m"] = round(lane_gap, 2)
        rsu_rows[-1]["junction_clearance_m"] = round(junction_gap, 2)

    court = (680, 644, 910, 775)
    ET.SubElement(root, "poly", {"id": "n1-inspection-court", "type": "visual-inspection-court",
        "color": "#897E68", "fill": "0", "lineWidth": "1.1", "layer": "8",
        "shape": f"{court[0]},{court[1]} {court[2]},{court[1]} {court[2]},{court[3]} {court[0]},{court[3]}"})
    ET.SubElement(root, "poi", {"id": "n1-court-legend", "x": "771", "y": "699", "type": "visual-only-legend",
        "color": "#FFFFFF", "layer": "19", "name": "", "width": "174", "height": "60", "imgFile": "assets/n1_sample_legend.png"})

    samples = [
        ("sample-pothole", "pothole", "POTHOLE · 1.1×0.9 m", 711, 763, 1.1, .9, "sample_pothole.png", 0),
        ("sample-puddle", "puddle", "PUDDLE · 1.5×1.1 m", 772, 763, 1.5, 1.1, "sample_puddle.png", 0),
        ("sample-manhole", "manhole", "MANHOLE · 0.9×0.9 m", 833, 763, .9, .9, "sample_manhole.png", 0),
        ("sample-asphalt-patch", "asphalt_patch", "PATCH · 1.8×1.3 m", 894, 763, 1.8, 1.3, "sample_asphalt_patch.png", 0),
        ("sample-fallen-tree", "fallen_tree", "TREE · 5.0×1.8 m", 711, 744, 5.0, 1.8, "sample_fallen_tree.png", -20),
        ("sample-debris", "debris", "DEBRIS · 2.4×1.4 m", 772, 744, 2.4, 1.4, "sample_debris.png", 0),
        ("sample-incident-car", "incident_car", "INCIDENT CAR · 4.3×1.8 m", 833, 744, 4.3, 1.8, "sample_incident_car.png", 90),
        ("sample-construction", "construction", "WORKS · 6.0×3.0 m", 894, 744, 6.0, 3.0, "sample_construction.png", 0),
    ]
    sample_manifest = []
    for sid, category, label, x, y, w, h, icon, angle in samples:
        ET.SubElement(root, "poi", {"id": sid, "x": str(x), "y": str(y), "type": f"n1_visual_sample_{category}",
            "color": "#FFFFFF", "layer": "15", "name": label,
            "width": str(w), "height": str(h), "angle": str(angle), "imgFile": f"assets/{icon}"})
        sample_manifest.append({"id": sid, "category": category, "position_m": [x, y],
                               "dimensions_m": [w, h], "angle_deg": angle, "label": label})

    ET.indent(root, space="    ")
    ET.ElementTree(root).write(OUT_XML, encoding="utf-8", xml_declaration=True)
    manifest = {
        "scene": "Foundry and Market native SUMO scenery N1",
        "provenance": {"generator": "scenery.py", "compiled_network": NET.name,
            "compiled_network_sha256": hashlib.sha256(NET.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "placement_basis": "compiled external lane polylines and junction coordinates"},
        "palette": PALETTE,
        "inspection_court": {"boundary_m": list(court), "meaning": "off-road visual-only fixtures; no lane blockage, collision, detector, or AI claim"},
        "samples": sample_manifest,
        "counts": {"buildings": len(roofs), "olive_trees": len(trees), "rsu_masts": len(rsu_rows),
                   "storage_tanks": tank_count, "chimney_footprints": stack_count},
        "buildings_by_area": {lot["name"]: sum(1 for row in roofs if row[6] == lot["name"]) for lot in lots},
        "rsu_fixtures": rsu_rows,
        "limits": ["All N1 objects are scenery; SUMO does not treat their images as obstacles.",
                   "The inspection court is off-road and has no detector or traffic-control semantics."],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_XML.name}, {OUT_MANIFEST.name}, {len(list(ASSETS.glob('*.png')))} PNG assets")
    print(f"Placed {len(roofs)} buildings, {len(trees)} olive trees, {len(rsu_rows)} RSUs, {len(samples)} visual samples")


if __name__ == "__main__":
    main()
