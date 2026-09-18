Got it — you just want the file and clear instructions. Here's exactly what to do now, in 3 steps:

## What you do now (5 minutes)

1. **Create a project folder** — e.g., `sumo_edgeai/`
2. **Create a file inside it called `PROMPT.md`** and paste the content below into it
3. **Open your local AI** (Cursor / Cline / Continue / Ollama chat — whatever you're using), open that folder as the workspace, and paste the same content as your **first message**

Then just reply "confirmed, next phase" as you finish each phase.

---

## Copy this → save as `PROMPT.md`

```markdown
# PROMPT.md — SUMO Real-Location Simulation → Excel Data → Edge-AI Dataset

## 1. YOUR ROLE
You are a senior traffic-simulation engineer and Python developer, expert in:
- SUMO (netconvert, duarouter, sumo-gui, sumo, netedit)
- OpenStreetMap (OSM) import into SUMO
- SUMO outputs: FCD (floating car data), tripinfo, detectors, SSM device
- TraCI (live Python control of simulations)
- pandas pipelines converting XML → CSV/Excel

Guide me PHASE BY PHASE to build a reproducible simulation of a real road
network, populate it with vehicles, inject obstacles, and export clean,
AI-ready trajectory data.

## 2. PROJECT GOAL (END-TO-END)
1. Import the real road network at the location below into SUMO
2. Simulate mixed Indian-style traffic (cars, bikes, autos, buses, trucks)
3. Add obstacles: broken-down vehicle blocking a lane, lane closure, slow vehicle
4. Record high-frequency (10 Hz) trajectory data
5. Export to CSV/Excel
6. Build a final dataset to train a LOCAL EDGE-AI model (kinematic data only,
   NO camera data). Hardware implementation may come later — keep everything
   modular, scripted, and headless-runnable.

## 3. TARGET LOCATION (REAL WORLD)
- Latitude: 19.309211 | Longitude: 84.792861
- Region: Berhampur (Brahmapur), Ganjam district, Odisha, India
- Bounding box (order = WEST, SOUTH, EAST, NORTH — lon,lat,lon,lat):
  `84.784861,19.301211,84.800861,19.317211` (≈1.7 km × 1.7 km)
  WARNING: wrong bbox order = empty map. This is the #1 failure mode.

## 4. MY SYSTEM
- OS: [Windows 11 / Ubuntu 22.04 / macOS]  ← I will fill this before starting
- Python version: [3.10 / 3.11]
- SUMO install method: [pip / official installer / apt]
- Output of `sumo --version`: [paste here]

If this section is not filled in, ASK ME FIRST before starting Phase 0.

## 5. TARGET PROJECT STRUCTURE
```
sumo_edgeai/
├── map.osm                # raw OSM extract
├── map.net.xml            # SUMO network
├── vtypes.add.xml         # vehicle type definitions
├── traffic_*.rou.xml      # demand per vehicle class
├── ego.rou.xml            # ego (AI-subject) vehicle
├── obstacles.add.xml      # static obstacles / lane closures
├── detectors.add.xml      # induction loops / area detectors
├── map.sumocfg            # main simulation config
├── scripts/
│   ├── list_edges.py      # prints valid edge IDs from the network
│   ├── fcd_to_excel.py    # FCD XML → CSV + Excel
│   ├── build_dataset.py   # ego-centric AI dataset builder
│   ├── run_sweep.py       # multi-seed scenario runner
│   └── traci_obstacles.py # dynamic obstacle injection (Phase 8)
└── outputs/
```

## 6. WORKFLOW RULES (STRICTLY FOLLOW)
1. Work ONE PHASE AT A TIME. End each phase with a verification step (exact
   command + what I should see), then WAIT for my confirmation before continuing.
2. TEACH as you go: briefly explain every command, flag, and file — I want to
   LEARN SUMO, not just copy-paste.
3. Give EXACT commands for MY OS from section 4.
4. If I paste an error, diagnose it (likely cause + fix) before continuing.
5. NEVER invent edge/lane IDs — always show me how to get real IDs from MY network.
6. Final pipeline must run headless: `sumo -c map.sumocfg`. GUI only for inspection.
7. Comment every XML and Python file so I can modify it myself later.

---

## 7. PHASES

### PHASE 0 — Install & verify
- Preferred: `pip install eclipse-sumo pandas openpyxl matplotlib`
  (includes traci + sumolib)
- Alternative: official installer from https://sumo.dlr.de/docs/Downloads.php,
  then set SUMO_HOME
- Verify: `sumo --version` and `python -c "import traci, sumolib; print('ok')"`
- Locate the SUMO tools directory (needed for randomTrips.py etc.):
  `python -c "import sumolib, os; print(os.path.dirname(os.path.dirname(sumolib.__file__)))"`

### PHASE 1 — Get the real map from OpenStreetMap
Option A (script):
`python <tools>/osmGet.py --bbox 84.784861,19.301211,84.800861,19.317211 map.osm`
Option B (web): openstreetmap.org → search "19.309211, 84.792861" → Export →
"Manually select a different area" → enter the bbox → Export → save as `map.osm`
- Verify: file should be several hundred KB for this urban area.

### PHASE 2 — Build the SUMO network
```
netconvert --osm-files map.osm --output-file map.net.xml \
  --type-files "<SUMO_HOME>/data/typemap/osmNetconvert.typ.xml" \
  --geometry.remove --ramps.guess --junctions.join \
  --tls.guess-signals --tls.discard-simple --tls.join \
  --remove-edges.isolated --keep-edges.components 1
```
Explain each flag. Then inspect visually: `sumo-gui -n map.net.xml`
(roads must match the real Berhampur area). Create `scripts/list_edges.py`
(sumolib) printing all edges allowing "passenger" vehicles — needed for Phases 3-4.

### PHASE 3 — Add vehicles
Create `vtypes.add.xml` with this Indian traffic mix (explain every attribute):
```xml
<additional>
  <vType id="car"   vClass="passenger"  length="4.5" minGap="2.5" accel="2.6" decel="4.5" maxSpeed="16.67" sigma="0.5" color="200,0,0"/>
  <vType id="bike"  vClass="motorcycle" length="2.0" minGap="1.0" accel="3.0" decel="6.0" maxSpeed="16.67" sigma="0.7" color="0,0,200"/>
  <vType id="auto"  vClass="passenger"  length="3.2" minGap="1.5" accel="2.0" decel="4.0" maxSpeed="11.11" sigma="0.6" color="230,180,0"/>
  <vType id="bus"   vClass="bus"        length="11"  minGap="3.0" accel="1.2" decel="3.5" maxSpeed="13.89" sigma="0.4" color="0,160,0"/>
  <vType id="truck" vClass="truck"      length="8.0" minGap="4.0" accel="1.0" decel="3.0" maxSpeed="11.11" sigma="0.4" color="255,140,0"/>
  <vType id="ego"   vClass="passenger"  length="4.5" minGap="2.5" accel="2.6" decel="9.0" maxSpeed="16.67" sigma="0.0" color="255,255,255">
    <param key="has.ssm.device" value="true"/>
  </vType>
</additional>
```
Generate demand with randomTrips.py (one run per class, different headways `-p`,
bikes most frequent), each with `--validate`:
```
python <tools>/randomTrips.py -n map.net.xml -r traffic_car.rou.xml \
  -a vtypes.add.xml -e 3600 -p 6 --prefix car --fringe-factor 5 \
  --trip-attributes 'type="car" departLane="best" departSpeed="max"' --validate
```
Repeat: bike (p=3), auto (p=12), bus (p=60), truck (p=45).
Then create `ego.trips.xml` (single trip, type="ego") on a long main road — help me
pick the edge pair using list_edges.py / netedit — and route it:
`duarouter -n map.net.xml -r ego.trips.xml -o ego.rou.xml -a vtypes.add.xml`
- Verify in sumo-gui: mixed traffic moving on real roads, white ego visible,
  no mass teleport warnings (if congested, raise `-p`).

### PHASE 4 — Add obstacles
All lane/edge IDs must come from MY network. Create `obstacles.add.xml`:

**A. Broken-down vehicle blocking a lane (static):**
```xml
<vehicle id="obstacle_broken1" type="car" depart="180" departLane="0" departPos="80" departSpeed="0">
  <route edges="REAL_EDGE_ID"/>
  <stop lane="REAL_EDGE_ID_0" endPos="90" until="3600"/>
</vehicle>
```
(no parking flag → it BLOCKS the lane; vehicles must overtake or queue)

**B. Lane closure / road-work zone (rerouter):**
```xml
<rerouter id="rr1" edges="REAL_EDGE_ID">
  <interval begin="600" end="3600">
    <closingLaneReroute id="REAL_EDGE_ID_1"/>
  </interval>
</rerouter>
```

**C. Slow-moving obstacle:** a vType with `maxSpeed="2.0"` (crawling truck).

**D. (Visual) red polygon** marking the danger zone via `<poly>` for screenshots.

- Verify in sumo-gui: queue behind broken vehicle, lane changes occur,
  rerouting around closure works.

### PHASE 5 — Configure outputs & run
Create `map.sumocfg` (explain every option):
```xml
<configuration>
  <input>
    <net-file value="map.net.xml"/>
    <route-files value="traffic_car.rou.xml,traffic_bike.rou.xml,traffic_auto.rou.xml,traffic_bus.rou.xml,traffic_truck.rou.xml,ego.rou.xml"/>
    <additional-files value="vtypes.add.xml,obstacles.add.xml,detectors.add.xml"/>
  </input>
  <time>
    <begin value="0"/><end value="3600"/>
    <step-length value="0.1"/>   <!-- 10 Hz data, like a real automotive sensor -->
  </time>
  <output>
    <fcd-output value="outputs/fcd.xml"/>
    <fcd-output.signals value="true"/>
    <tripinfo-output value="outputs/tripinfo.xml"/>
    <summary-output value="outputs/summary.xml"/>
    <statistic-output value="outputs/stats.xml"/>
  </output>
  <processing>
    <time-to-teleport value="-1"/>      <!-- teleports corrupt trajectories -->
    <collision.action value="warn"/>
    <collision.mingap-factor value="0"/>
  </processing>
</configuration>
```
Add `detectors.add.xml` (inductionLoop + laneAreaDetector on the obstacle road)
and SSM output writing `outputs/ssm.xml` (verify exact attribute names against my
installed SUMO version). Run headless: `sumo -c map.sumocfg`, then show in sumo-gui.

### PHASE 6 — Convert outputs to CSV/Excel
Write `scripts/fcd_to_excel.py` (full, runnable, commented) that:
1. Parses outputs/fcd.xml with xml.etree
2. Converts x,y → lat/lon via
   `sumolib.net.readNet("map.net.xml").convertXY2LonLat(x, y)`
3. Derives accel (Δspeed/Δt), speed_kmph, is_obstacle, is_ego columns
4. Writes FULL data to `outputs/fcd_full.csv` (Excel caps at ~1,048,576 rows —
   never dump full FCD to xlsx) and `outputs/fcd_data.xlsx` with sheets:
   ego_trajectory + fcd_sample + tripinfo
Also show the built-in alternative: `<tools>/xml/xml2csv.py`.

### PHASE 7 — Build the edge-AI dataset
Write `scripts/build_dataset.py` producing `outputs/dataset_edge_ai.xlsx`,
ego-centric table (one row per 0.1 s timestep):
`time_s, run_id, ego_speed_mps, ego_accel_mps2, ego_signals (brake=8),
gap_to_nearest_ahead_m, nearest_vehicle_id, closing_speed_mps, ttc_s,
n_vehicles_within_30m, obstacle_ahead(0/1), label_brake_1s`
- gap: leader pos − leader length − ego pos on same lane (or euclidean nearest)
- ttc: gap / closing speed (cross-check with ssm.xml)
- label_brake_1s: 1 if ego decel < −3 m/s² within the next 10 steps
  (first training label — suggest 2–3 more labels for obstacle avoidance)
Then `scripts/run_sweep.py`: run ≥ 10 seeds
(`sumo -c map.sumocfg --seed S --fcd-output outputs/fcd_S.xml`), parse all,
add `run_id`, concatenate into the final dataset.

### PHASE 8 — Dynamic obstacles with TraCI (advanced)
Write `scripts/traci_obstacle.py`: start sumo via traci, at a random time inject
a stopped vehicle ~40 m ahead of the ego on its current lane
(`traci.route.add` + `traci.vehicle.add` + `traci.vehicle.setStop`), log
everything, handle TraCIException. Randomize (time, gap, lane) across runs.
Explain the ego-centric observation window concept (20-step / 2 s history @ 10 Hz).

---

## 8. COMMON ERRORS — FIX IMMEDIATELY IF I HIT THEM
| Symptom | Cause | Fix |
|---|---|---|
| netconvert output empty/1 edge | bbox order wrong (must be W,S,E,N) | re-download with correct bbox |
| duarouter: "no connection between edges" | fragmented OSM network | `--keep-edges.components 1`, `--ignore-errors` |
| many "teleporting vehicle" warnings | gridlock / excess demand | raise `-p` headway, reduce obstacle jams |
| `sumo` not recognized | PATH / SUMO_HOME | fix env vars; pip puts binaries in Scripts/ |
| randomTrips.py not found | wrong tools path | use Phase 0 tools-locator command |
| fcd.xml empty | no vehicles departed | check depart errors in console |
| Excel row limit hit | full FCD too big | write CSV only |

## 9. DEFINITION OF DONE
- [ ] `sumo -c map.sumocfg` runs 3600 sim-seconds, no fatal errors
- [ ] sumo-gui shows mixed traffic on real Berhampur roads, obstacles working
- [ ] fcd_full.csv: 10 Hz spacing, lat/lon spot-check on Google Maps
- [ ] dataset_edge_ai.xlsx: ego_view sheet with populated TTC + labels
- [ ] ≥ 10 scenario runs merged with run_id
- [ ] Everything re-runs end-to-end from a clean folder

## 10. REFERENCES
https://sumo.dlr.de/docs/Import/OSM.html • /Simulation/Output/FCDOutput.html •
/Tools/Trip.html • /Simulation/Safety.html • /TraCI.html
— verify exact option names against MY installed version if anything errors.

**BEGIN NOW WITH PHASE 0 (after I confirm my system details from section 4).**
```

---

## What to do in the next 30 minutes (before even opening the AI)

1. Fill in **Section 4** of the file with your actual OS + Python version
2. Run these two commands yourself to see where you stand:
   ```
   sumo --version
   python --version
   ```
3. If SUMO isn't installed yet: `pip install eclipse-sumo pandas openpyxl matplotlib`
4. Then paste the file into your local AI and let it take over from Phase 0

**One tip:** don't let the AI rush — if it dumps all phases on you in one message, reply with *"stop, follow section 6 rule 1: one phase at a time."* That's the whole trick to getting good results from a local model on this.

Want me to also tell you which local AI setup works best for this (Cursor, Cline + Ollama, etc.), or are you already set on that?