# SUMO Visualization & Presentation Guidelines

This guide specifies visual rendering standards, satellite decal configurations, vehicle models, and color-coded V2X states for demonstrations and presentations in `sumo-gui`.

---

## 1. Fast Demonstration vs. Macro Satellite Visuals

The project supports two visual modes:

### Mode A: Lightweight 1km Corridor (Recommended for Presentations)
- **Path**: `simulation/corridor_1km/corridor.sumocfg`
- **Launcher**: Double-click `run_gui.bat` or run `python run_cooperative_v2x_sim.py`.
- **Performance**: High frame rate (**60+ FPS**) with instantaneous loading.
- **Visual Features**:
  - Two signalized intersections with live switching traffic lights.
  - Three distinct color-coded hazard zones placed on Lane 0 (Workzone, Pothole, Accident).
  - Cyan RSU Station POI marker at mid-corridor (`x=500, y=12`).
  - Real-time vehicle color state transitions:
    - **Orange `(255, 140, 0)`**: Detecting vehicle (onboard AI model flags hazard).
    - **Yellow `(255, 215, 0)`**: Warned vehicle (received V2X DENM advisory from RSU).
    - **Green `(0, 255, 120)`**: Safe vehicle (successfully executed anticipatory lane change).

---

### Mode B: Full Macro Berhampur OSM Map with Satellite Decals
- **Path**: `simulation/map.sumocfg`
- **When to Use**: When presenting large-scale urban routing over real-world road geometry in Odisha, India.
- **Rules for Satellite Backgrounds**:
  1. Do NOT use `polyconvert` to render flat colored polygons unless explicitly requested.
  2. Use `tileGet.py` to download high-resolution satellite imagery tiles:
     ```powershell
     python "C:\Program Files (x86)\Eclipse\Sumo\tools\tileGet.py" -n simulation/map.net.xml -t 10 -s simulation/satellite.xml -m satellite
     ```
  3. **Tile Overlap Rule**: In `satellite_decals.xml`, always inflate tile `width` and `height` by `+2.0 meters` to prevent white rendering gaps between tile boundaries.
  4. Launch `sumo-gui` with the settings file:
     ```powershell
     sumo-gui -c simulation/map.sumocfg -g simulation/satellite.xml
     ```

---

## 2. Vehicle Rendering Standards

To avoid generic flat triangle shapes in `sumo-gui`:
1. **Always Set `guiShape` in `<vType>`**:
   - Cars: `guiShape="passenger/sedan"`
   - Bikes: `guiShape="motorcycle"`
   - Auto-Rickshaws: `guiShape="passenger"`
   - Buses: `guiShape="bus"`
   - Trucks: `guiShape="truck"`
2. **Realistic Colors**:
   - Car: Red `(200, 0, 0)`
   - Bike: Blue `(0, 0, 200)`
   - Auto: Golden Yellow `(230, 180, 0)`
   - Bus: Green `(0, 160, 0)`
   - Truck: Dark Orange `(255, 140, 0)`
   - Ego / Connected: Clean White `(255, 255, 255)`

---

## 3. Dynamic Hazard Visual Elements

- **Hazard POI Definition**: Hazard zones should include visible POIs or poly markers on the closed lane so viewers instantly see why vehicles are braking or merging.
- **Camera Tracking**: In `sumo-gui`, right-click on the lead detecting vehicle and select **Track Vehicle** to smoothly follow the vehicle as it approaches the hazard and triggers the V2X warning beacon.
