import os
import sys
import math
import concurrent.futures
import requests
import xml.etree.ElementTree as ET

try:
    import sumolib
except ImportError:
    sys.exit("Please install sumolib: pip install sumolib")

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def download_tile(url, filepath):
    if os.path.exists(filepath):
        return filepath
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
    except Exception as e:
        print(f"Failed: {e}")
    return None

def get_quadkey(x, y, zoom):
    quadkey = ""
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey

def main():
    net_file = "maps/map_16_9.net.xml"
    out_dir = "satellite_highres"
    zoom = 16  # Lower zoom = fewer tiles = SUMO doesn't freeze
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Reading network file for coordinate conversion...")
    net = sumolib.net.readNet(net_file)
    
    # Cover the entire network area (16:9 bounds)
    lon_min, lat_min = 84.7505, 19.2867
    lon_max, lat_max = 84.8351, 19.3317
    
    x_min, y_max = deg2num(lat_min, lon_min, zoom)
    x_max, y_min = deg2num(lat_max, lon_max, zoom)
    
    if x_max < x_min: x_min, x_max = x_max, x_min
    if y_max < y_min: y_min, y_max = y_max, y_min
    
    tiles_to_download = []
    servers = ["t0", "t1", "t2", "t3"]
    
    # Build the decals XML
    xml_path = "xml_configs/satellite_decals.xml"
    with open(xml_path, "w") as f:
        f.write('<viewsettings>\n')
        f.write('    <scheme name="real world (satellite)">\n')
        f.write('    </scheme>\n')
        f.write('    <decals>\n')
        
        counter = 0
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                server = servers[counter % len(servers)]
                q = get_quadkey(x, y, zoom)
                # Google Maps Hybrid (Satellite with labels) URL
                url = f"https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={zoom}"
                filepath = os.path.join(out_dir, f"tile_{zoom}_{x}_{y}.jpg")
                tiles_to_download.append((url, filepath))
                
                # Get the geo coordinates of this tile's corners
                lat_top, lon_left = num2deg(x, y, zoom)
                lat_bottom, lon_right = num2deg(x + 1, y + 1, zoom)
                
                # Convert to SUMO XY coordinates
                sumo_x_left, sumo_y_top = net.convertLonLat2XY(lon_left, lat_top)
                sumo_x_right, sumo_y_bottom = net.convertLonLat2XY(lon_right, lat_bottom)
                
                # Add a 5% overlap (multiply by 1.05) to eliminate white grid gaps
                width = (abs(sumo_x_right - sumo_x_left)) * 1.05
                height = (abs(sumo_y_top - sumo_y_bottom)) * 1.05
                
                center_x = (sumo_x_left + sumo_x_right) / 2
                center_y = (sumo_y_top + sumo_y_bottom) / 2
                
                f.write(f'        <decal file="../{out_dir}/tile_{zoom}_{x}_{y}.jpg" ')
                f.write(f'centerX="{center_x:.2f}" centerY="{center_y:.2f}" ')
                f.write(f'width="{width:.2f}" height="{height:.2f}" layer="-100"/>\n')
                
                counter += 1
        
        f.write('    </decals>\n')
        f.write('</viewsettings>\n')
    
    total = len(tiles_to_download)
    print(f"Downloading {total} tiles at Zoom Level {zoom}...")
    
    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_tile, url, fp): fp for url, fp in tiles_to_download}
        for future in concurrent.futures.as_completed(futures):
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"  Downloaded {downloaded}/{total} tiles...")
    
    print(f"\nDone! {downloaded}/{total} tiles downloaded.")
    print(f"Satellite decals written to {xml_path}")

if __name__ == "__main__":
    main()
