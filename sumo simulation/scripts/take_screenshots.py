import os
import sumolib
import traci
import random
import time

def take_screenshots():
    output_dir = os.path.join('asset', 'preppt')
    os.makedirs(output_dir, exist_ok=True)
    
    # Start sumo-gui with traci
    # Add geometry and gui settings for best visuals
    sumo_cmd = ["sumo-gui", "-c", "map.sumocfg", "--start", "-Q"]
    traci.start(sumo_cmd)
    
    net = sumolib.net.readNet('maps/map_16_9.net.xml')
    edges = net.getEdges()
    
    # Warm up simulation to get dense traffic (run to step 300)
    for _ in range(300):
        traci.simulationStep()
    
    screenshot_count = 0
    
    # 1. Macro view
    bbox = net.getBBoxXY()
    center_x = (bbox[0][0] + bbox[1][0]) / 2
    center_y = (bbox[0][1] + bbox[1][1]) / 2
    
    traci.gui.setOffset("View #0", center_x, center_y)
    traci.gui.setZoom("View #0", 1500)
    traci.gui.screenshot("View #0", os.path.join(output_dir, f"{screenshot_count:02d}_macro_view.png"))
    screenshot_count += 1
    
    # Advance time a bit to allow file to write
    for _ in range(10): traci.simulationStep()
    
    # Take various screenshots across busy edges over time
    for i in range(19):
        # find the busiest edges currently
        edge_counts = []
        for e in edges:
            c = traci.edge.getLastStepVehicleNumber(e.getID())
            if c > 0:
                edge_counts.append((c, e))
        
        edge_counts.sort(key=lambda x: x[0], reverse=True)
        
        if not edge_counts:
            # no vehicles, just pick a random edge
            target_edge = random.choice(edges)
        else:
            # pick one of the top 5 busiest edges
            top_edges = edge_counts[:5]
            target_edge = random.choice(top_edges)[1]
            
        shape = target_edge.getShape()
        ex, ey = shape[len(shape)//2] # middle of the edge
        
        zoom_level = random.choice([5000, 8000, 12000, 15000])
        
        traci.gui.setOffset("View #0", ex, ey)
        traci.gui.setZoom("View #0", zoom_level)
        
        # Advance simulation to create dynamic feel
        steps_to_advance = random.randint(10, 50)
        for _ in range(steps_to_advance):
            traci.simulationStep()
            
        filename = os.path.join(output_dir, f"{screenshot_count:02d}_traffic_detail.png")
        traci.gui.screenshot("View #0", filename)
        screenshot_count += 1
        
    traci.close()
    print(f"Successfully saved {screenshot_count} screenshots to {output_dir}")

if __name__ == "__main__":
    take_screenshots()
