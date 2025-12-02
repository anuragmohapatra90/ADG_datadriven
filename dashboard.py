import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandapower.plotting as plot
import pandapower as pp
from active_grid import config, utils
from active_grid.envs import DistributionGridEnv
from active_grid.controller import MLP_MPC_Controller

# --- PAGE SETUP ---
st.set_page_config(page_title="Active Grid Control", layout="wide")
st.title("⚡ AI-Driven Active Distribution Grid Control")

# --- INITIALIZE STATE ---
if 'env' not in st.session_state:
    st.session_state['env'] = DistributionGridEnv()
    try:
        net = st.session_state['env'].net
        if not hasattr(net, "bus_geodata") or net.bus_geodata is None:
            net.bus_geodata = pd.DataFrame(columns=["x", "y"])
        
        plot.create_generic_coordinates(net, respect_switches=True)
        
        # Apply V-shape adjustments
        for i in range(2, 9):
            if i in net.bus_geodata.index: net.bus_geodata.at[i, 'x'] -= (i * 0.6) 
        for i in range(9, 15):
            if i in net.bus_geodata.index: net.bus_geodata.at[i, 'x'] += ((i-9) * 0.6)
            
    except Exception as e:
        st.error(f"Grid Layout Init Failed: {e}")

    try:
        st.session_state['controller'] = MLP_MPC_Controller()
    except Exception as e:
        st.sidebar.error(f"Failed to load Controller: {e}")
        st.session_state['controller'] = None

if 'base_scenario' not in st.session_state:
    load_p, load_q, pv = utils.generate_scenarios(1, config.BUS_COUNT)
    st.session_state['base_scenario'] = {'load_p': load_p[0], 'load_q': load_q[0], 'pv': pv[0]}

# --- SIDEBAR ---
st.sidebar.header("1. Scenario Initialization")
if st.sidebar.button("🎲 Generate New Random Scenario"):
    load_p, load_q, pv = utils.generate_scenarios(1, config.BUS_COUNT)
    st.session_state['base_scenario'] = {'load_p': load_p[0], 'load_q': load_q[0], 'pv': pv[0]}
    st.sidebar.success("New scenario generated!")

st.sidebar.markdown("---")
st.sidebar.header("2. Operational Adjustments")
load_scale = st.sidebar.slider("Load Scaling Factor", 0.5, 2.0, 1.0)
pv_scale = st.sidebar.slider("PV Generation Factor", 0.0, 2.0, 1.0)
controller_active = st.sidebar.checkbox("Enable MPC Controller", value=True)

if controller_active:
    st.sidebar.subheader("MPC Gains")
    weight_p = st.sidebar.number_input("Cost P (Active)", value=1.0, step=0.1)
    weight_q = st.sidebar.number_input("Cost Q (Reactive)", value=0.1, step=0.1)
    weight_v = st.sidebar.number_input("Cost Voltage", value=1000.0, step=100.0)
else:
    weight_p, weight_q, weight_v = 1.0, 0.1, 1000.0

st.sidebar.markdown("---")
if st.sidebar.button("▶️ Simulate This Configuration"):
    st.session_state['run_sim'] = True

# --- PLOTTING HELPER (FIXED) ---
def plot_grid_heatmap(net, voltages, title):
    # 1. Setup Figure
    # Using specific figsize usually helps Streamlit render it consistently
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 2. Define Shared Color Properties
    # We define these once so both loads (circles) and DERs (squares) 
    # use the exact same color scale.
    vmin, vmax = 0.90, 1.10
    cmap_name = "coolwarm_r"
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 3. Draw Lines & Trafo (Background)
    # We draw these first so they are behind the buses
    collections = []
    collections.append(plot.create_line_collection(net, color='gray', linewidth=1.5))
    collections.append(plot.create_trafo_collection(net, size=0.5, color='k'))
    collections.append(plot.create_ext_grid_collection(net, size=0.5, orientation=0))
    
    # 4. Prepare Voltage Data
    der_buses = config.DER_BUSES
    load_buses = [b for b in net.bus.index if b not in der_buses and b > 1]
    
    v_loads = [voltages[b] for b in load_buses]
    v_ders = [voltages[b] for b in der_buses]
    
    # 5. Create Bus Collections
    # CRITICAL: We pass 'cbar_title=None' to STOP pandapower from making its own bars.
    # We pass 'norm=norm' and 'cmap=cmap' to ensure consistency.
    
    bc_load = plot.create_bus_collection(
        net, buses=load_buses, size=0.2, 
        z=v_loads, cmap=cmap, norm=norm,
        patch_type="circle", zorder=10,
        cbar_title=None 
    )
    
    bc_der = plot.create_bus_collection(
        net, buses=der_buses, size=0.3,
        z=v_ders, cmap=cmap, norm=norm,
        patch_type="rect", zorder=11,
        cbar_title=None
    )
    
    collections.append(bc_load)
    collections.append(bc_der)

    # Draw everything at once
    plot.draw_collections(collections, ax=ax)
    
    # 6. Add ONE Manual Colorbar using ScalarMappable
    # This creates a "fake" object (sm) that holds the color data, 
    # and we attach the colorbar to that, NOT to the plot collections.
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    
    # fraction and pad adjust the size/position of the bar
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Voltage (p.u.)")

    # 7. Labels
    for i in net.bus.index:
        if i in net.bus_geodata.index:
            x = net.bus_geodata.at[i, 'x']
            y = net.bus_geodata.at[i, 'y']
            # Offset the text slightly so it doesn't cover the node
            ax.text(x, y + 0.4, f"{i}", ha='center', fontsize=8, color='black')
            
    ax.set_title(title)
    ax.set_axis_off()
    
    return fig

# --- MAIN LOGIC ---
if st.session_state.get('run_sim', False):
    
    base = st.session_state['base_scenario']
    p_net_base = base['load_p'] * load_scale - base['pv'] * pv_scale
    q_net_base = base['load_q'] * load_scale
    
    # A. Uncontrolled
    v_uncontrolled = st.session_state['env'].step(p_net_base, q_net_base, {}, {})
    
    # B. Controlled
    v_controlled = None
    der_p_set = {}
    der_q_set = {}
    
    if controller_active and st.session_state['controller']:
        with st.spinner("MPC Optimizing..."):
            opt_p, opt_q = st.session_state['controller'].get_optimal_setpoints(
                p_net_base, q_net_base, 
                weight_p=weight_p, weight_q=weight_q, weight_v=weight_v
            )
            v_controlled = st.session_state['env'].step(p_net_base, q_net_base, opt_p, opt_q)
            der_p_set = opt_p
            der_q_set = opt_q
    
    # --- VISUALIZATION ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Voltage Profile")
        fig_profile, ax = plt.subplots(figsize=(10, 4))
        ax.axhline(config.V_MAX_PU, color='r', linestyle='--', label='Max')
        ax.axhline(config.V_MIN_PU, color='r', linestyle='--', label='Min')
        ax.axhline(1.0, color='gray', linewidth=0.5)
        
        if v_uncontrolled is not None:
            ax.plot(v_uncontrolled, 'o--', color='orange', label='Uncontrolled')
        if v_controlled is not None:
            ax.plot(v_controlled, 's-', color='green', linewidth=2, label='MPC Controlled')
            
        ax.set_ylim(0.88, 1.12) 
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_profile)
        plt.close(fig_profile) # Good practice to close explicit figures
        
    with col2:
        st.subheader("Actions")
        if der_p_set:
            df_ctrl = pd.DataFrame({
                "Bus": list(der_p_set.keys()),
                "P (kW)": [v * 1000 for v in der_p_set.values()],
                "Q (kVar)": [v * 1000 for v in der_q_set.values()]
            })
            st.dataframe(df_ctrl.style.format("{:.2f}"), height=150)
            
            st.markdown(rf"**Cost Function:** $J = {weight_p} \cdot \sum P^2 + {weight_q} \cdot \sum Q^2 + {weight_v} \cdot \sum V_{{viol}}^2$")
        else:
            if controller_active: st.info("Grid Safe.")
            else: st.warning("Controller OFF.")

    # HEATMAPS
    st.subheader("Voltage Heatmap")
    c1, c2 = st.columns(2)
    net_obj = st.session_state['env'].net
    
    with c1:
        st.caption("Uncontrolled")
        if v_uncontrolled is not None:
            # We pass the figure explicitly to st.pyplot
            fig1 = plot_grid_heatmap(net_obj, v_uncontrolled, "Baseline")
            st.pyplot(fig1)
            plt.close(fig1) # Clean up memory
            
    with c2:
        st.caption("Controlled")
        data_to_plot = v_controlled if v_controlled is not None else v_uncontrolled
        title = "MPC Controlled" if v_controlled is not None else "Uncontrolled"
        if data_to_plot is not None:
            fig2 = plot_grid_heatmap(net_obj, data_to_plot, title)
            st.pyplot(fig2)
            plt.close(fig2) # Clean up memory

    st.session_state['run_sim'] = False