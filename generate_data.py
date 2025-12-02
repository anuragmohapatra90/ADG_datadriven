import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from active_grid import config, utils
from active_grid.envs import DistributionGridEnv

def run_campaign(n_samples, name):
    env = DistributionGridEnv()
    
    # Get Standard Profiles
    load_p_profile, load_q_profile, pv_profile = utils.generate_scenarios(n_samples, config.BUS_COUNT)
    
    data = []
    failures = 0 
    
    for i in tqdm(range(n_samples), desc=f"Generating {name}"):
        # Net Load calculation
        p_net = load_p_profile[i] - pv_profile[i]
        q_net = load_q_profile[i]
        
        # DER actuation logic
        der_p_act = {}
        der_q_act = {}
        for bus in config.DER_BUSES:
            # FIXED: Use limits from config.py instead of hardcoded 0.02
            p_action = np.random.uniform(-config.MAX_P_MW, config.MAX_P_MW) 
            q_action = np.random.uniform(-config.MAX_Q_MVAR, config.MAX_Q_MVAR)
            der_p_act[bus] = p_action
            der_q_act[bus] = q_action

        # Run Step
        v_meas = env.step(p_net, q_net, der_p_act, der_q_act)
        
        # --- CONVERGENCE CHECK ---
        if v_meas is None:
            failures += 1
            continue  
            
        # Record Data
        row = {}
        for b in range(config.BUS_COUNT):
             # 1. Calculate the TOTAL Net Active Power (Load - Gen)
             # This is what the physics sees.
             
             p_injection_control = der_p_act.get(b, 0.0)
             q_injection_control = der_q_act.get(b, 0.0)
             
             # Total Net Load = Load - (PV_profile + Control_Action)
             total_p_net_load = load_p_profile[i][b] - (pv_profile[i][b] + p_injection_control)
             total_q_net_load = load_q_profile[i][b] - q_injection_control

             # 2. Save with explicit keys
             row[f'p_mw_{b}'] = total_p_net_load
             row[f'q_mvar_{b}'] = total_q_net_load
             row[f'vm_pu_{b}'] = v_meas[b]
        
        data.append(row)

    # --- REPORTING ---
    if len(data) == 0:
        print("CRITICAL ERROR: No data generated.")
        return pd.DataFrame()

    success_rate = 100 * (len(data) / n_samples)
    print(f"\n--- {name} Generation Report ---")
    print(f"Requested Samples: {n_samples}")
    print(f"Successful Samples: {len(data)}")
    print(f"Failed (Non-Converged): {failures}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # FIXED: Use counts from config.py
    print(f"--- 1. Generating Training Data ({config.N_SAMPLES_TRAIN} samples) ---")
    df_train = run_campaign(config.N_SAMPLES_TRAIN, "train")
    if not df_train.empty:
        df_train.to_csv("data/train.csv", index=False)
    
    print(f"--- 2. Generating Test Data ({config.N_SAMPLES_TEST} samples) ---")
    df_test = run_campaign(config.N_SAMPLES_TEST, "test")
    if not df_test.empty:
        df_test.to_csv("data/test.csv", index=False)
    
    print("Done! Data saved to data/ folder.")
    
    # --- DEBUGGING / VISUALIZATION ---
    if not df_train.empty:
        print(f"DEBUG: Columns found in dataframe: {df_train.columns[:5]} ...")
        
        try:
            plt.figure(figsize=(10,4))
            
            # Subplot 1: Histogram
            plt.subplot(1,2,1)
            plt.title("Voltage Histogram (Bus 14)")
            plt.hist(df_train['vm_pu_14'], bins=30, color='skyblue', edgecolor='black')
            plt.xlabel("Voltage (p.u.)")
            
            # Subplot 2: P vs V
            plt.subplot(1,2,2)
            plt.title("P vs Voltage (Bus 14)")
            plt.scatter(df_train['p_mw_14'], df_train['vm_pu_14'], s=5, alpha=0.5, c='orange')
            plt.xlabel("Net Load (MW)")
            plt.ylabel("Voltage (p.u.)")
            
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"\nVISUALIZATION ERROR: Could not find column {e}")