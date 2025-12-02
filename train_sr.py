import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import os
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the new SR model wrapper
from active_grid.sr_model import SRVoltageEstimator
from active_grid import config

# --- SETTINGS ---
EPOCHS = 2              # We will run 20 "chunks" of training
ITERATIONS_PER_EPOCH = 5 # 5 generations per chunk = 100 generations total
MODEL_PATH = "sr_model.pkl" 
SCALER_PATH = "scalers.pkl" 

def load_and_process_data(filepath):
    """Loads CSV and separates Features (P, Q) from Targets (V)."""
    df = pd.read_csv(filepath)
    p_cols = [c for c in df.columns if 'p_mw' in c]
    q_cols = [c for c in df.columns if 'q_mvar' in c]
    v_cols = [c for c in df.columns if 'vm_pu' in c]
    X = pd.concat([df[p_cols], df[q_cols]], axis=1).values.astype(np.float32)
    y = df[v_cols].values.astype(np.float32)
    return X, y

def print_all_equations(model, bus_count=15):
    """Cleanly prints the discovered equation for every bus."""
    print("\n" + "="*80)
    print(f"{'BUS':<5} | {'DISCOVERED VOLTAGE EQUATION (Simplified)':<70}")
    print("-" * 80)
    
    # get_best() returns a list of dictionaries (one per output/bus)
    best_eqs = model.get_best_equations()
    
    for i in range(bus_count):
        try:
            # Depending on PySR version, it might be a list of dicts or objects
            # We try to access the equation string safely
            if isinstance(best_eqs, list) and len(best_eqs) > i:
                eq_data = best_eqs[i]
                eq_str = eq_data["equation"] if "equation" in eq_data else str(eq_data)
            else:
                eq_str = "Equation not found"
            
            # Truncate if insanely long
            if len(eq_str) > 70:
                eq_str = eq_str[:67] + "..."
            
            print(f"{i:<5} | V_{i} = {eq_str}")
        except Exception as e:
            print(f"{i:<5} | [Error printing equation: {e}]")
    print("="*80 + "\n")

def train():
    print("--- 1. Data Preparation ---")
    X_full, y_full = load_and_process_data("data/train.csv")
    X_test_raw, y_test_raw = load_and_process_data("data/test.csv")
    
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    print("Fitting Scalers...")
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test_raw)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    joblib.dump({'X': scaler_X, 'y': scaler_y}, SCALER_PATH)

    print(f"\n--- 2. Model Training (Symbolic Regression) ---")
    print(f"Configuration: {EPOCHS} Epochs x {ITERATIONS_PER_EPOCH} Generations")
    print("Initializing Julia backend (this takes a moment)...")
    
    # Initialize with 'verbosity=0' to keep console clean
    sr_estimator = SRVoltageEstimator(niterations=ITERATIONS_PER_EPOCH, verbosity=0)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        # Fit for a short burst
        sr_estimator.fit(X_train_scaled, y_train_scaled)
        
        # 1. MSE on Scaled Data (Directly comparable to MLP Loss)
        pred_train = sr_estimator.predict(X_train_scaled)
        pred_val = sr_estimator.predict(X_val_scaled)
        
        mse_train = mean_squared_error(y_train_scaled, pred_train)
        mse_val = mean_squared_error(y_val_scaled, pred_val)
        
        # 2. MAE on Unscaled Data (Human Readable Voltage Error in p.u.)
        # We only do this for validation to check "True" performance
        pred_val_pu = scaler_y.inverse_transform(pred_val)
        y_val_pu = scaler_y.inverse_transform(y_val_scaled)
        mae_val_pu = mean_absolute_error(y_val_pu, pred_val_pu)
        
        train_losses.append(mse_train)
        val_losses.append(mse_val)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train MSE: {mse_train:.5f} | Val MSE: {mse_val:.5f} | Val MAE: {mae_val_pu:.5f} p.u.")

    # Save the model
    sr_estimator.save_pickle(MODEL_PATH)

    # --- 3. Final Output ---
    # Show Equations
    print_all_equations(sr_estimator, bus_count=y_train.shape[1])

    # Global MAE
    preds_scaled = sr_estimator.predict(X_test_scaled)
    preds_pu = scaler_y.inverse_transform(preds_scaled)
    global_mae = np.mean(np.abs(preds_pu - y_test_raw))
    print(f"FINAL TEST MAE (Global Average): {global_mae:.5f} p.u.")
    
    # Learning Curve
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Validation MSE')
    plt.xlabel("Epoch (Chunks)")
    plt.ylabel("MSE Loss (Scaled)")
    plt.title("Symbolic Regression Learning Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()