# Generate Bike Traffic (frequent)
python "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n map.net.xml -r traffic_bike.rou.xml -a vtypes.add.xml -e 3600 -p 3 --prefix bike --fringe-factor 5 --trip-attributes "type='bike' departLane='best' departSpeed='max'" --validate --vtype-output dummy.xml

# Generate Car Traffic
python "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n map.net.xml -r traffic_car.rou.xml -a vtypes.add.xml -e 3600 -p 6 --prefix car --fringe-factor 5 --trip-attributes "type='car' departLane='best' departSpeed='max'" --validate --vtype-output dummy.xml

# Generate Auto Rickshaw Traffic
python "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n map.net.xml -r traffic_auto.rou.xml -a vtypes.add.xml -e 3600 -p 12 --prefix auto --fringe-factor 5 --trip-attributes "type='auto' departLane='best' departSpeed='max'" --validate --vtype-output dummy.xml

# Generate Truck Traffic (sparse)
python "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n map.net.xml -r traffic_truck.rou.xml -a vtypes.add.xml -e 3600 -p 45 --prefix truck --fringe-factor 5 --trip-attributes "type='truck' departLane='best' departSpeed='max'" --validate --vtype-output dummy.xml

# Generate Bus Traffic (very sparse)
python "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n map.net.xml -r traffic_bus.rou.xml -a vtypes.add.xml -e 3600 -p 60 --prefix bus --fringe-factor 5 --trip-attributes "type='bus' departLane='best' departSpeed='max'" --validate --vtype-output dummy.xml

Write-Host "Background traffic generation complete!"
