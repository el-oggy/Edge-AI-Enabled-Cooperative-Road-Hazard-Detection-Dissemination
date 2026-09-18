# SUMO Visual Preferences

When the user asks for realistic visuals, a satellite background, or a Google Maps style theme in SUMO:
1. Do NOT use `polyconvert` to extract colored polygons from OSM unless explicitly requested.
2. Instead, use `tileGet.py` to download satellite tiles.
3. First ensure `pyproj` and `requests` are installed via pip.
4. Run: `python "C:\Program Files (x86)\Eclipse\Sumo\tools\tileGet.py" -n map.net.xml -t 10 -s satellite.xml -m satellite`
5. Instruct the user to run `sumo-gui` with `-g satellite.xml` or `--gui-settings-file satellite.xml` instead of `-a polygons.add.xml`.
