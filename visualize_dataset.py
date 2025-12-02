import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from active_grid import config

def visualize_comprehensive():
    print("--- Loading Dataset for Forensics ---")
    try:
        df = pd.read_csv("data/train.csv")
    except FileNotFoundError:
        print("Error: data/train.csv not found. Run generate_data.py first.")
        return

    # Filter columns by type
    p_cols = [c for c in df.columns if 'p_mw' in c]
    q_cols = [c for c in df.columns if 'q_mvar' in c]
    v_cols = [c for c in df.columns if 'vm_pu' in c]

    # Convert P/Q from MW to kW for easier reading in plots
    df_p_kw = df[p_cols] * 1000
    df_q_kvar = df[q_cols] * 1000
    
    # Extract Bus Indices from column names for labeling
    # Assumes format 'p_mw_0', 'p_mw_1', etc.
    bus_indices = [int(c.split('_')[-1]) for c in p_cols]

    # --- FIX: RENAME COLUMNS FOR ALIGNMENT ---
    # We rename columns to "Bus X" so both P and Q plots share the exact same X-axis labels.
    # This prevents 'sharex=True' from creating two separate sets of ticks.
    rename_map_p = {c: f"Bus {int(c.split('_')[-1])}" for c in p_cols}
    rename_map_q = {c: f"Bus {int(c.split('_')[-1])}" for c in q_cols}
    rename_map_v = {c: f"Bus {int(c.split('_')[-1])}" for c in v_cols}

    df_p_plot = df_p_kw.rename(columns=rename_map_p)
    df_q_plot = df_q_kvar.rename(columns=rename_map_q)
    df_v_plot = df[v_cols].rename(columns=rename_map_v)

    # --- FIGURE 1: INPUT DISTRIBUTIONS (P and Q Spread per Node) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot P Boxplots
    sns.boxplot(data=df_p_plot, ax=ax1, color='skyblue', showfliers=False)
    ax1.set_title("Distribution of Net Active Power (P) per Node across all Scenarios", fontsize=14)
    ax1.set_ylabel("Active Power (kW)\n(+ = Consumption, - = Generation)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=1)
    
    # Plot Q Boxplots
    sns.boxplot(data=df_q_plot, ax=ax2, color='lightgreen', showfliers=False)
    ax2.set_title("Distribution of Net Reactive Power (Q) per Node", fontsize=14)
    ax2.set_ylabel("Reactive Power (kVar)")
    
    # Rotate x-labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: VOLTAGE SPREAD (The Consequence) ---
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_v_plot, color='orange', showfliers=False)
    
    # Add Limits
    plt.axhline(config.V_MIN_PU, color='red', linestyle='--', linewidth=2, label='Min Safe (0.95)')
    plt.axhline(config.V_MAX_PU, color='red', linestyle='--', linewidth=2, label='Max Safe (1.05)')
    plt.axhline(1.0, color='black', linestyle='-', alpha=0.3, label='Nominal')
    
    plt.title("Voltage Spread per Node (Consequences of P/Q)", fontsize=14)
    plt.ylabel("Voltage (p.u.)")
    plt.xlabel("Grid Nodes")
    plt.xticks(rotation=45) # Rotate labels directly
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- FIGURE 3: PHYSICS VALIDATION (P-V Sensitivity) ---
    # We chose the two end-nodes (Bus 8 and Bus 14) as they are most sensitive
    critical_buses = config.DER_BUSES[-2:] # Take last two DER buses usually end of feeders
    if len(critical_buses) < 2: 
        critical_buses = [8, 14] # Fallback
    
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, bus_idx in enumerate(critical_buses):
        if i >= 2: break # Only plot 2
        ax = axes[i]
        p_col = f'p_mw_{bus_idx}'
        v_col = f'vm_pu_{bus_idx}'
        
        # Scatter Plot
        # We use 'alpha' to show density of points
        if p_col in df.columns and v_col in df.columns:
            ax.scatter(df[p_col] * 1000, df[v_col], alpha=0.2, s=10, c='purple')
            
            ax.set_title(f"Sensitivity at Bus {bus_idx}")
            ax.set_xlabel("Net Active Power (kW)")
            ax.set_ylabel("Voltage (p.u.)")
            ax.grid(True)
            
            # Add a trendline to show dV/dP
            z = np.polyfit(df[p_col] * 1000, df[v_col], 1)
            p = np.poly1d(z)
            ax.plot(df[p_col] * 1000, p(df[p_col] * 1000), "r--", linewidth=2, label=f"Slope: {z[0]*1000:.1e} pu/MW")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Bus data not found", ha='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_comprehensive()