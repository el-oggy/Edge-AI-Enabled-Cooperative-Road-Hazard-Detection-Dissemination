import xml.etree.ElementTree as ET
import csv
import sys

fcd_file = "outputs/fcd.xml"
csv_file = "outputs/fcd_data.csv"
print(f"Parsing {fcd_file} using memory-efficient streaming parser...")

try:
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Write CSV header
        writer.writerow(['time', 'vehicle_id', 'x', 'y', 'angle', 'type', 'speed', 'pos', 'lane', 'slope'])
        
        count = 0
        unique_vehicles = set()
        
        context = ET.iterparse(fcd_file, events=('start', 'end'))
        current_time = 0.0
        
        for event, elem in context:
            if event == 'start' and elem.tag == 'timestep':
                current_time = float(elem.get('time', 0))
                
            elif event == 'end' and elem.tag == 'vehicle':
                vid = elem.get('id')
                unique_vehicles.add(vid)
                
                writer.writerow([
                    current_time,
                    vid,
                    elem.get('x', 0),
                    elem.get('y', 0),
                    elem.get('angle', 0),
                    elem.get('type', ''),
                    elem.get('speed', 0),
                    elem.get('pos', 0),
                    elem.get('lane', ''),
                    elem.get('slope', 0)
                ])
                count += 1
                
                # Memory management: clear the processed element from the tree
                elem.clear()
            
            elif event == 'end' and elem.tag == 'timestep':
                # Clear the timestep element after processing all its vehicles
                elem.clear()
                
        print(f"\n--- Simulation Data Summary ---")
        print(f"Total Unique Vehicles Spawned: {len(unique_vehicles)}")
        print(f"Total Trajectory Data Points: {count}")
        print("-------------------------------\n")
        print(f"Success! {count} rows have been streamed directly to {csv_file}")
        
except FileNotFoundError:
    print(f"Error: {fcd_file} not found. Please run the simulation first.")
    sys.exit(1)
except ET.ParseError:
    print(f"\n--- Simulation Data Summary ---")
    print(f"Total Unique Vehicles Spawned: {len(unique_vehicles)}")
    print(f"Total Trajectory Data Points: {count}")
    print("-------------------------------\n")
    print(f"Warning: The XML file was truncated, likely because the simulation was closed before finishing completely.")
    print(f"Success! However, all data generated up to that point ({count} rows) was successfully saved to {csv_file}")
