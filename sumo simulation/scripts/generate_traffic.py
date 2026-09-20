import os
import random
import sumolib

def generate_traffic():
    net_file = "maps/map_16_9.net.xml"
    out_file = "traffic/trips.rou.xml"
    
    print(f"Reading network {net_file}...")
    net = sumolib.net.readNet(net_file)
    edges = net.getEdges()
    
    # Filter for standard passenger edges (exclude pedestrian-only, railways, etc.)
    valid_edges = [e for e in edges if e.allows("passenger")]
    
    print(f"Found {len(valid_edges)} valid edges for routing.")
    
    vtypes = ["bike", "auto", "car", "bus", "truck"]
    # Probabilities for typical Indian mixed traffic
    weights = [0.3, 0.2, 0.35, 0.05, 0.1]
    
    num_vehicles = 4000
    start_time = 0
    end_time = 3600  # Spread across 1 hour
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    print(f"Generating {num_vehicles} random trips...")
    trips = []
    for i in range(num_vehicles):
        # Pick a random vtype
        vtype = random.choices(vtypes, weights=weights, k=1)[0]
        
        # Pick random origin and destination
        origin = random.choice(valid_edges)
        dest = random.choice(valid_edges)
        
        # Ensure origin != dest
        while origin == dest:
            dest = random.choice(valid_edges)
        
        # Random departure time
        depart = random.uniform(start_time, end_time)
        
        trips.append((depart, vtype, origin, dest, i))
        
    # Sort trips chronologically by depart time (required by SUMO)
    trips.sort(key=lambda x: x[0])
        
    with open(out_file, "w") as f:
        f.write('<routes>\n')
        
        for t in trips:
            depart, vtype, origin, dest, i = t
            f.write(f'    <trip id="veh_{i}" type="{vtype}" depart="{depart:.2f}" from="{origin.getID()}" to="{dest.getID()}"/>\n')
            
        f.write('</routes>\n')
        
    print(f"Successfully generated {num_vehicles} trips in {out_file}")

if __name__ == "__main__":
    generate_traffic()
