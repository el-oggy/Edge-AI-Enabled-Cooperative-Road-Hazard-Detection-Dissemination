import urllib.request
import urllib.parse
import sys

# Focused 16:9 bounds centered on the dense road area of Berhampur
south, west = 19.221139, 84.750235
north, east = 19.326165, 84.838428

# Use Overpass QL query (more reliable than /api/map endpoint)
query = f"""
[out:xml][timeout:300];
(
  node({south},{west},{north},{east});
  way({south},{west},{north},{east});
  relation({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""

url = "https://overpass-api.de/api/interpreter"
data = urllib.parse.urlencode({"data": query}).encode("utf-8")

print(f"Downloading OSM data for bbox: {south},{west},{north},{east}")
print("This may take 1-2 minutes for a large area...")

req = urllib.request.Request(url, data=data, headers={"User-Agent": "PFPD-LL/1.0"})
try:
    with urllib.request.urlopen(req, timeout=300) as response:
        osm_data = response.read()
        outfile = "maps/map_focused.osm"
        with open(outfile, "wb") as f:
            f.write(osm_data)
        print(f"Success! Downloaded {len(osm_data):,} bytes to {outfile}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
