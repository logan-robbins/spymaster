import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from pathlib import Path
import logging

def visualize_vacuum_heatmap_absolute():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    SCRIPT_DIR = Path(__file__).parent
    DATA_PATH = SCRIPT_DIR / "vacuum_heatmap_clean.json"
    PLOT_OUT = SCRIPT_DIR / "vacuum_density_heatmap_clean.png"
    
    if not DATA_PATH.exists():
        logger.error(f"Data not found: {DATA_PATH}")
        return

    logger.info("Loading Data...")
    with open(DATA_PATH) as f:
        data = json.load(f)

    timestamps = []
    prices_list = []
    vwaps_list = []
    
    # 1. Determine Global Price Range
    all_prices = []
    for frame in data:
         all_prices.extend(frame['prices'])
         
    global_min_p = min(all_prices)
    global_max_p = max(all_prices)
    
    # Pad global range
    global_min_p = np.floor(global_min_p) - 1.0
    global_max_p = np.ceil(global_max_p) + 1.0
    
    TICK = 0.25
    y_grid = np.arange(global_min_p, global_max_p + 0.001, TICK)
    y_to_idx = {round(p, 2): i for i, p in enumerate(y_grid)}
    
    logger.info(f"Global Price Grid: {global_min_p} to {global_max_p} ({len(y_grid)} levels)")
    
    # Z-Matrices
    z_matrix = np.zeros((len(y_grid), len(data)))
    
    for t_idx, frame in enumerate(data):
        ts = pd.Timestamp(frame['ts'], unit='ns', tz='UTC').tz_convert('America/New_York')
        timestamps.append(ts)
        
        current_price = frame.get('price', frame['vwap']) # Fallback
        prices_list.append(current_price)
        vwaps_list.append(frame['vwap'])
        
        frame_prices = frame['prices']
        vac_up = frame['vacuum_up']
        vac_down = frame['vacuum_down']
        
        for i, p in enumerate(frame_prices):
            p = round(p, 2)
            if p not in y_to_idx: continue
            y_idx = y_to_idx[p]
            
            # Logic:
            # If P < Current Price -> Vacuum Down (Bid Erosion)
            # If P > Current Price -> Vacuum Up (Ask Erosion)
            # If P == Current Price -> Average or Max?
            
            val = 0.0
            if p > current_price:
                val = vac_up[i]
            elif p < current_price:
                val = vac_down[i]
            else:
                val = max(vac_up[i], vac_down[i])
                
            z_matrix[y_idx, t_idx] = val

    # Plotting
    logger.info("Generating Absolute Plot...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(32, 24))
    
    # EDGE DEFINITIONS for 'flat' shading
    # Time (X): 0 to N+1 edges for N frames
    x_edges = np.arange(len(timestamps) + 1)
    
    # Price (Y): Edges around the centers (y_grid)
    # y_grid contains centers. Edge = center +/- TICK/2
    edge_offset = TICK / 2
    # Create edges from min-offset to max+offset
    # Since y_grid is regular, we can just linspace or construct
    y_edges = np.concatenate([y_grid - edge_offset, [y_grid[-1] + edge_offset]])
    
    # Heatmap with Flat Shading (Grid Aligned)
    vmax = np.percentile(z_matrix, 98)
    cmap = plt.cm.inferno
    
    # pcolormesh(X_edges, Y_edges, Z, shading='flat')
    c = ax.pcolormesh(x_edges, y_edges, z_matrix, cmap=cmap, shading='flat', vmin=0, vmax=vmax)
    
    # Overlays (Centered in the bin)
    x_centers = x_edges[:-1] + 0.5
    ax.plot(x_centers, prices_list, color='cyan', linewidth=2.5, label="Close Price")
    ax.plot(x_centers, vwaps_list, color='white', linestyle='--', linewidth=1.5, alpha=0.8, label="VWAP")
    
    # Colorbar
    cbar = plt.colorbar(c, ax=ax, pad=0.01)
    cbar.set_label("Vacuum Density (Velocity Potential)")
    
    # Formatting
    ax.set_title("Vacuum Physics Density Heatmap (Rigorous)\nBright Areas Below = Bid Erosion (Price Falls) | Bright Areas Above = Ask Erosion (Price Rises)", fontsize=20)
    ax.set_ylabel("Price (Tick Buckets)", fontsize=14)
    ax.set_xlabel("Time (ET)", fontsize=14)
    
    # Y-Axis Logic
    # Major Ticks at Centers (Labels)
    ax.set_yticks(y_grid)
    # Minor Ticks at Edges (Grid Lines)
    ax.set_yticks(y_edges, minor=True)
    
    # X-Axis Logic
    # Edges are 0, 1, 2...
    # Centers are 0.5, 1.5, 2.5...
    # We want Labels at Ceenters, Grid at Edges.
    
    # Grid Lines (Minor Ticks at Edge Indices)
    tick_step = 1
    ax.set_xticks(x_edges[::tick_step], minor=True)
    
    # Labels (Major Ticks at Center Indices)
    # We need timestamps for the centers. 
    # timestamp[i] corresponds to the box starting at x_edges[i]
    # So visually we label the center of that box with the start time (or center time).
    # "09:30:00" label in the middle of the 09:30:00-05 box.
    x_centers_ticks = x_edges[:-1] + 0.5
    ax.set_xticks(x_centers_ticks[::tick_step])
    
    # Generate Labels
    lbls = []
    for i in range(0, len(timestamps), tick_step):
        lbls.append(timestamps[i].strftime('%H:%M:%S'))
        
    ax.set_xticklabels(lbls, rotation=90, fontsize=8)
    
    ax.legend(loc='upper left', fontsize=14)
    
    # Grid Configuration
    # Show Grid for MINOR ticks (Lines between boxes)
    # Hide Grid for MAJOR ticks (Centers)
    ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(False, which='major')
    
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300)
    logger.info(f"Saved plot to {PLOT_OUT}")

if __name__ == "__main__":
    visualize_vacuum_heatmap_absolute()
