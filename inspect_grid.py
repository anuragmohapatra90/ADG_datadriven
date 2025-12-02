import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from active_grid import config
from active_grid.envs import DistributionGridEnv

def inspect():
    print("--- Visualizing Moderate Grid Structure ---")
    env = DistributionGridEnv()
    net = env.net
    
    # 1. Initialize Coordinate Tables
    if not hasattr(net, "bus_geodata") or net.bus_geodata is None:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
        
    # 2. Create Layout
    plot.create_generic_coordinates(net, respect_switches=True)
    
    # 3. Force "V" Shape Separation
    # Shift Feeder A (2-8) Left, Feeder B (9-14) Right
    for i in range(2, 9):
        if i in net.bus_geodata.index:
            net.bus_geodata.at[i, 'x'] -= (i * 0.6) 
    for i in range(9, 15):
        if i in net.bus_geodata.index:
            net.bus_geodata.at[i, 'x'] += ((i-9) * 0.6)

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tiny Symbols
    plot.draw_collections([
        plot.create_line_collection(net, color='gray', linewidth=1.5),
        plot.create_trafo_collection(net, size=0.5, color='k'),
        plot.create_ext_grid_collection(net, size=0.5, orientation=0)
    ], ax=ax)
    
    # Highlight DER Buses
    der_buses = config.DER_BUSES
    load_buses = [b for b in net.bus.index if b not in der_buses and b > 1]
    
    # --- FIX APPLIED HERE ---
    # We use 'patch_type' instead of 'marker'
    plot.draw_collections([
        # Regular Loads: Circles
        plot.create_bus_collection(net, buses=load_buses, size=0.2, color='blue', patch_type="circle", zorder=10),
        # DERs: Squares (Rectangles)
        plot.create_bus_collection(net, buses=der_buses, size=0.3, color='green', patch_type="rect", zorder=11)
    ], ax=ax)

    # Labels
    for i in net.bus.index:
        if i in net.bus_geodata.index:
            x = net.bus_geodata.at[i, 'x']
            y = net.bus_geodata.at[i, 'y']
            ax.text(x, y + 0.4, f"{i}", ha='center', fontsize=9, color='black')
            
    plt.title("Grid Topology: 15-Bus Branched System\n(Green Squares = Controllable DERs)")
    plt.tight_layout()
    plt.show()

    # 5. Physics Validation
    print("\n--- CABLE CHECK ---")
    print(f"Cable Used: {net.line.std_type.iloc[0]}")
    r_val = net.line.iloc[0].length_km * net.line.iloc[0].r_ohm_per_km
    print(f"Sample Line Resistance: {r_val:.4f} Ohm")
    
if __name__ == "__main__":
    inspect()