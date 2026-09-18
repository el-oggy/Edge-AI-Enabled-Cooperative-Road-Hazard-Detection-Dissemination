import sumolib
import argparse

def get_passenger_edges(net_file):
    # Parse the SUMO network
    net = sumolib.net.readNet(net_file)
    
    passenger_edges = []
    for edge in net.getEdges():
        # Check if any lane on this edge allows passenger vehicles
        allows_passenger = False
        for lane in edge.getLanes():
            if lane.allows("passenger"):
                allows_passenger = True
                break
        
        if allows_passenger:
            passenger_edges.append((edge.getID(), edge.getLength()))
            
    # Sort by length descending, so we can easily find long roads for our ego vehicle later
    passenger_edges.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(passenger_edges)} edges allowing passenger vehicles.")
    print("Top 10 longest edges (useful for spawning our ego vehicle later):")
    for edge_id, length in passenger_edges[:10]:
        print(f"  ID: {edge_id}, Length: {length:.2f}m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List passenger edges in a SUMO network")
    parser.add_argument("-n", "--net", default="map.net.xml", help="SUMO network file")
    args = parser.parse_args()
    
    get_passenger_edges(args.net)
