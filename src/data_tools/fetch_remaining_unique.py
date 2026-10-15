"""
Fetch Remaining Unique Benchmark Images to reach exactly 150 unique images
"""
import urllib.request
import urllib.parse
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE = ROOT_DIR / "data" / "benchmark_150"

TARGETS = [
    {
        "cat": "potholes/dry",
        "target": 25,
        "queries": [
            "pothole asphalt", "road asphalt damage", "cracked road pavement",
            "pothole street pavement", "pothole crater road", "bad road asphalt hole"
        ]
    },
    {
        "cat": "construction/with_symbols",
        "target": 25,
        "queries": [
            "traffic cone road", "road work sign", "traffic barrels construction",
            "road barrier construction", "highway work zone sign", "under construction road sign"
        ]
    },
    {
        "cat": "construction/without_symbols",
        "target": 25,
        "queries": [
            "road workers asphalt paving", "road resurfacing excavator",
            "asphalt roller road work", "highway construction machinery",
            "road repair crew asphalt", "workers paving street"
        ]
    },
    {
        "cat": "accidents_obstacles/collisions",
        "target": 20,
        "queries": [
            "car crash highway collision", "overturned car road accident",
            "traffic collision vehicle damage", "traffic accident car wreck",
            "damaged vehicle road crash"
        ]
    },
    {
        "cat": "accidents_obstacles/fallen_trees",
        "target": 15,
        "queries": [
            "fallen tree road storm", "tree branch blocking street",
            "fallen trunk roadway traffic", "windstorm fallen tree road",
            "hurricane tree across street", "fallen timber roadway"
        ]
    },
    {
        "cat": "accidents_obstacles/boulders",
        "target": 15,
        "queries": [
            "rockfall on road", "rockslide blocking roadway", "boulder on highway",
            "landslide rocks on road", "falling rocks mountain road",
            "rock avalanche roadway", "stone on road hazard"
        ]
    }
]

def search_wikimedia(query, max_results=30):
    urls = []
    headers = {"User-Agent": "RoadHazardResearchBot/3.0 (academic@hazard.edu)"}
    for offset in [0, 20, 40, 60]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrlimit": "20",
            "gsroffset": str(offset),
            "prop": "imageinfo",
            "iiprop": "url|mime"
        }
        url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    break
                for pid, pdata in pages.items():
                    info = pdata.get("imageinfo", [{}])[0]
                    u = info.get("url")
                    mime = info.get("mime", "").lower()
                    if u and ("jpeg" in mime or "jpg" in mime or "png" in mime):
                        clean_u = u.split("?")[0]
                        if clean_u not in urls:
                            urls.append(clean_u)
        except Exception:
            pass
        if len(urls) >= max_results:
            break
    return urls

def download_img(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RoadHazardResearchBot/3.0 (academic@hazard.edu)"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            arr = np.asarray(bytearray(raw), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None and img.shape[0] >= 120 and img.shape[1] >= 120:
                h, w = img.shape[:2]
                if max(h, w) > 1920:
                    scale = 1920 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                hsh = hashlib.md5(raw).hexdigest()
                return img, hsh
    except Exception:
        pass
    return None, None

def main():
    # Build global hash set
    all_files = list(BASE.rglob("*.*"))
    existing_hashes = set(hashlib.md5(f.read_bytes()).hexdigest() for f in all_files if f.is_file())
    print(f"Initial unique existing images: {len(existing_hashes)}")

    for t in TARGETS:
        folder = BASE / t["cat"]
        folder.mkdir(parents=True, exist_ok=True)
        current_files = list(folder.glob("*.*"))
        needed = t["target"] - len(current_files)
        print(f"\nChecking {t['cat']}: Have {len(current_files)}, Need {needed} more...")

        if needed <= 0:
            print(f"  ✓ {t['cat']} already full!")
            continue

        candidate_urls = []
        for q in t["queries"]:
            urls = search_wikimedia(q, max_results=30)
            for u in urls:
                if u not in candidate_urls:
                    candidate_urls.append(u)
            if len(candidate_urls) >= needed * 3:
                break

        print(f"  Found {len(candidate_urls)} candidate URLs for {t['cat']}. Downloading...")
        added = 0
        for u in candidate_urls:
            if added >= needed:
                break
            img, hsh = download_img(u)
            if img is not None and hsh not in existing_hashes:
                existing_hashes.add(hsh)
                count = len(list(folder.glob("*.*"))) + 1
                prefix = folder.name
                fn = f"{prefix}_{count:02d}.jpg"
                cv2.imwrite(str(folder / fn), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                added += 1

        print(f"  -> Added {added} unique images to {t['cat']}. Total now: {len(list(folder.glob('*.*')))}")

    # Final audit
    final_files = list(BASE.rglob("*.*"))
    final_hashes = set(hashlib.md5(f.read_bytes()).hexdigest() for f in final_files if f.is_file())
    print("\n" + "=" * 70)
    print(f"FINAL AUDIT: {len(final_files)} Total Files, {len(final_hashes)} Unique Hashes")
    print("=" * 70)

if __name__ == "__main__":
    main()
