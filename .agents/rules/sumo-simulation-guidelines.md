# SUMO Simulation Guidelines & Engineering Standards

These rules govern all SUMO network configurations, TraCI scripts, and vehicle routing files in this repository.

---

## 1. Vehicle Rendering Standards
- When defining any `<vType>` in SUMO, **ALWAYS** explicitly set `guiShape` (e.g., `passenger/sedan`, `motorcycle`, `bus`, `truck`) to ensure it renders as a realistic 3D model rather than a basic geometric triangle.
- Set realistic length, minimum gap, acceleration, and deceleration profiles matching real-world vehicular dynamics.

---

## 2. OpenStreetMap Routing Resilience
- When generating random traffic over an imported OSM network, always include:
  ```xml
  <ignore-route-errors value="true"/>
  ```
  inside the `<processing>` block of your `sumocfg` file. This prevents unexpected simulation crashes caused by isolated or disconnected OSM edge segments.
- Set `<time-to-teleport value="-1"/>` so vehicles do not teleport across hazards, which would corrupt trajectory and safety evaluations.

---

## 3. Satellite Tile Overlap & Decal Alignment
- When generating `satellite_decals.xml` programmatically, always inflate the calculated tile `width` and `height` by at least `2.0` meters to force a small visual overlap. This prevents white rendering gap artifacts from appearing between tiles in `sumo-gui`.

---

## 4. Corridor vs. Macro Network Selection
- For real-time demonstrations, presentations, and rapid debugging of V2X algorithms, prioritize the lightweight **1km Signalized Corridor** (`simulation/corridor_1km/corridor.sumocfg`).
- For macroscopic traffic flow and city-scale trajectory analysis, use the full **Berhampur Network** (`simulation/map.sumocfg`).

---

## 5. TraCI Closed-Loop Control & V2X Rules
- When injecting safety advisories via TraCI, issue both:
  1. A target lane change command (`traci.vehicle.changeLane(veh_id, target_lane, duration)`).
  2. A precautionary advisory speed reduction (`traci.vehicle.slowDown(veh_id, advisory_speed, duration)`).
- Always wrap TraCI connections in clean `try ... finally: traci.close()` blocks to prevent orphan simulation sockets from locking port numbers on Windows.
