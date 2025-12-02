import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib 
import matplotlib.pyplot as plt
from active_grid.mlp_model import VoltageEstimator
from active_grid import config

# --- SETTINGS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
MODEL_PATH = "mlp_model.pth"
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

def plot_parity(model, X_test_scaled, y_test_raw, scaler_y):
    """
    Advanced Parity Plot showing the Worst-Case Bus performance.
    """
    print("\n--- Generating Parity Plot ---")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled)
        y_pred_scaled = model(X_tensor).numpy()
    
    y_pred_pu = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate Worst Bus (Highest Error) to show the critical case
    errors = np.abs(y_pred_pu - y_test_raw)
    mae_per_bus = np.mean(errors, axis=0)
    worst_bus = np.argmax(mae_per_bus)
    
    actual = y_test_raw[:, worst_bus]
    predicted = y_pred_pu[:, worst_bus]
    
    # Regression stats for the plot line
    reg = LinearRegression().fit(actual.reshape(-1, 1), predicted)
    r2 = r2_score(actual, predicted)
    line_x = np.linspace(actual.min(), actual.max(), 100)
    line_y = reg.predict(line_x.reshape(-1, 1))
    
    plt.figure(figsize=(8, 8))
    # Scatter points
    plt.scatter(actual, predicted, alpha=0.3, s=15, c='royalblue', edgecolors='none', label='Test Samples')
    # Ideal line
    plt.plot([0.9, 1.1], [0.9, 1.1], 'r--', linewidth=2, label='Ideal (y=x)')
    # Best fit line
    plt.plot(line_x, line_y, 'k-', linewidth=2, alpha=0.8, label=f'Best Fit (R²={r2:.4f})')
    
    plt.title(f"Parity Plot: Surrogate Accuracy (Bus {worst_bus} - Worst Case)", fontsize=14)
    plt.xlabel("Ground Truth Voltage (p.u.)", fontsize=12)
    plt.ylabel("Predicted Voltage (p.u.)", fontsize=12)
    plt.legend(loc='upper left')
    
    # Zoom to data range
    min_view = min(actual.min(), predicted.min()) - 0.01
    max_view = max(actual.max(), predicted.max()) + 0.01
    plt.xlim(min_view, max_view)
    plt.ylim(min_view, max_view)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Text box with MAE for THIS BUS specifically
    mae_val = mae_per_bus[worst_bus]
    plt.text(0.95, 0.05, f"Bus {worst_bus} MAE: {mae_val:.5f} p.u.", 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def train():
    print("--- 1. Data Preparation ---")
    X_full, y_full = load_and_process_data("data/train.csv")
    X_test_raw, y_test_raw = load_and_process_data("data/test.csv")
    
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test_raw)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    joblib.dump({'X': scaler_X, 'y': scaler_y}, SCALER_PATH)
    print(f"Scalers saved to {SCALER_PATH}")

    train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_scaled))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val_scaled))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("\n--- 2. Model Training (Original Simple Setup) ---")
    model = VoltageEstimator(X_train.shape[1], y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        batch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                preds = model(X_v)
                val_loss += criterion(preds, y_v).item()
        
        avg_train_loss = batch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # --- 3. Evaluation & Visualization ---
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Scaled)")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # Calculate Global Average MAE
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test_scaled)).numpy()
        preds_pu = scaler_y.inverse_transform(preds_scaled)
        global_mae = np.mean(np.abs(preds_pu - y_test_raw))
        print(f"\nFINAL TEST MAE (Global Average): {global_mae:.5f} p.u.")

    # Call the advanced plotting function
    plot_parity(model, X_test_scaled, y_test_raw, scaler_y)

if __name__ == "__main__":
    train()