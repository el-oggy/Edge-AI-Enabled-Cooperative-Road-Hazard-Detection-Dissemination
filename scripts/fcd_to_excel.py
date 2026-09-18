import xml.etree.ElementTree as ET
import pandas as pd
import os

print("Parsing FCD XML file...")

fcd_file = "outputs/fcd.xml"
if not os.path.exists(fcd_file):
    print(f"Error: {fcd_file} not found. Please run the simulation headlessly first.")
    exit(1)

tree = ET.parse(fcd_file)
root = tree.getroot()

# List to hold all vehicle records
data = []
unique_vehicles = set()

# FCD structure is <timestep time="0.00"><vehicle id="ego" x="..." y="..." .../></timestep>
for timestep in root.findall('timestep'):
    time = timestep.get('time')
    for vehicle in timestep.findall('vehicle'):
        vid = vehicle.get('id')
        unique_vehicles.add(vid)
        
        # Extract vehicle attributes
        record = {
            'time': float(time),
            'id': vid,
            'x': float(vehicle.get('x', 0)),
            'y': float(vehicle.get('y', 0)),
            'angle': float(vehicle.get('angle', 0)),
            'type': vehicle.get('type', ''),
            'speed': float(vehicle.get('speed', 0)),
            'pos': float(vehicle.get('pos', 0)),
            'lane': vehicle.get('lane', ''),
            'slope': float(vehicle.get('slope', 0))
        }
        data.append(record)

print(f"\n--- Simulation Data Summary ---")
print(f"Total Unique Vehicles Spawned: {len(unique_vehicles)}")
print(f"Total Trajectory Data Points: {len(data)}")
print("-------------------------------\n")

print("Converting data to Pandas DataFrame...")
df = pd.DataFrame(data)

excel_file = "outputs/fcd_data.xlsx"
print(f"Saving to {excel_file} (this may take a moment)...")
df.to_excel(excel_file, index=False, engine='openpyxl')

print("Success! Data has been written to Excel.")
