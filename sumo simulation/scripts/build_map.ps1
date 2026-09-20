# Rebuild everything for the downloaded map_16_9_bbox.osm.xml

Write-Host "1/5: Building Network with ptstops and traffic lights..."
netconvert --osm-files maps/map_16_9_bbox.osm.xml -o maps/map_16_9.net.xml --geometry.remove --ramps.guess --tls.guess-signals --junctions.join --ptstop-output xml_configs/ptstops.xml

Write-Host "2/5: Extracting Polygons (Buildings, Water, etc)..."
polyconvert --net-file maps/map_16_9.net.xml --osm-files maps/map_16_9_bbox.osm.xml --type-file "C:\Program Files (x86)\Eclipse\Sumo\data\typemap\osmPolyconvert.typ.xml" -o xml_configs/polygons.add.xml

Write-Host "3/5: Rerouting Ego Vehicle for new network..."
duarouter -n maps/map_16_9.net.xml -r traffic/ego_trip.xml -o traffic/ego.rou.xml -a xml_configs/vtypes.add.xml --vtype-output dummy.xml --ignore-errors

Write-Host "4/5: Generating 2000 Background Traffic trips..."
python scripts/generate_traffic.py

Write-Host "5/5: Downloading High-Res Satellite Tiles..."
python scripts/download_tiles.py

Write-Host "Done!"
