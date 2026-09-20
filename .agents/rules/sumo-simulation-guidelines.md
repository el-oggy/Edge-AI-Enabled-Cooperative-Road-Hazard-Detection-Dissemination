# SUMO Simulation Guidelines

1. **Vehicle Rendering:** When defining a `<vType>` in SUMO, ALWAYS explicitly set `guiShape` (e.g., `passenger/sedan`, `motorcycle`, `bus`, `truck`) to ensure it renders as a realistic 3D/2D model rather than a basic geometric triangle.
2. **OpenStreetMap Routing Resilience:** When generating random traffic over an imported OSM network, always include `<ignore-route-errors value="true"/>` in the `<processing>` block of your `sumocfg` to prevent crashes from disconnected edge pairs.
3. **Satellite Tile Overlaps:** When generating `satellite_decals.xml` programmatically, always inflate the calculated tile `width` and `height` by at least `2.0` meters to force a small visual overlap. This prevents white rendering gap artifacts from appearing between tiles in `sumo-gui`.
