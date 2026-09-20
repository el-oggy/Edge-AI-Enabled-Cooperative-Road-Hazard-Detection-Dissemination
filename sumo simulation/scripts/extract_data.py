import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_tripinfo(xml_file):
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found.")
        return None
        
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = []
    for tripinfo in root.findall('tripinfo'):
        info = tripinfo.attrib
        data.append({
            'Vehicle ID': info.get('id'),
            'Type': info.get('vType'),
            'Depart Time (s)': float(info.get('depart')),
            'Arrival Time (s)': float(info.get('arrival')),
            'Duration (s)': float(info.get('duration')),
            'Route Length (m)': float(info.get('routeLength')),
            'Waiting Time (s)': float(info.get('waitingTime')),
            'Waiting Count': int(info.get('waitingCount')),
            'Time Loss (s)': float(info.get('timeLoss')),
        })
        
    return pd.DataFrame(data)

def main():
    os.makedirs('asset', exist_ok=True)
    
    xml_file = 'outputs/tripinfo.xml'
    print(f"Parsing {xml_file}...")
    
    df = parse_tripinfo(xml_file)
    if df is not None and not df.empty:
        excel_path = 'asset/simulation_data.xlsx'
        df.to_excel(excel_path, index=False)
        print(f"Success! Data saved to {excel_path}")
    else:
        print("No trip data found. Simulation might not have completed any trips.")

if __name__ == "__main__":
    main()
