import numpy as np
import torch
import joblib
from scipy.optimize import minimize
from . import config
from .mlp_model import VoltageEstimator

class MLP_MPC_Controller:
    def __init__(self, model_path="mlp_model.pth", scaler_path="scalers.pkl"):
        # 1. Load Scalers
        try:
            scalers = joblib.load(scaler_path)
            self.scaler_X = scalers['X']
            self.scaler_y = scalers['y']
        except FileNotFoundError:
            raise FileNotFoundError(f"Scalers not found at {scaler_path}. Run train_mlp.py first.")
        
        # 2. Load MLP Model
        try:
            self.model = VoltageEstimator(input_dim=2*config.BUS_COUNT, output_dim=config.BUS_COUNT)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except FileNotFoundError:
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_mlp.py first.")
        
        self.der_buses = config.DER_BUSES
        self.n_der = len(self.der_buses)
        
    def predict_voltages(self, load_p, load_q, der_p_setpoints, der_q_setpoints):
        p_vector = load_p.copy()
        q_vector = load_q.copy()
        
        for i, bus_idx in enumerate(self.der_buses):
            p_vector[bus_idx] -= der_p_setpoints[i]
            q_vector[bus_idx] -= der_q_setpoints[i]
            
        features = np.concatenate([p_vector, q_vector])
        features_scaled = self.scaler_X.transform(features.reshape(1, -1))
        
        with torch.no_grad():
            tensor_in = torch.tensor(features_scaled.astype(np.float32))
            pred_scaled = self.model(tensor_in).numpy()
            
        pred_pu = self.scaler_y.inverse_transform(pred_scaled)
        return pred_pu.flatten()

    def get_optimal_setpoints(self, current_load_p, current_load_q, weight_p=1.0, weight_q=0.1, weight_v=1000.0):
        """
        Solves MPC optimization with dynamic weights from dashboard.
        """
        x0 = np.zeros(2 * self.n_der)
        
        def objective(x):
            p_ctrl = x[:self.n_der]
            q_ctrl = x[self.n_der:]
            
            # Dynamic Cost Function
            cost_ctrl = weight_p * np.sum(p_ctrl**2) + weight_q * np.sum(q_ctrl**2)
            
            v_pred = self.predict_voltages(current_load_p, current_load_q, p_ctrl, q_ctrl)
            
            violation_low = np.maximum(0, config.V_MIN_PU - v_pred)
            violation_high = np.maximum(0, v_pred - config.V_MAX_PU)
            
            penalty_voltage = weight_v * np.sum(violation_low**2 + violation_high**2)
            
            return cost_ctrl + penalty_voltage

        bounds = []
        for _ in range(self.n_der): bounds.append((-config.MAX_P_MW, config.MAX_P_MW))
        for _ in range(self.n_der): bounds.append((-config.MAX_Q_MVAR, config.MAX_Q_MVAR))
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-4, 'disp': False})
        
        p_opt = res.x[:self.n_der]
        q_opt = res.x[self.n_der:]
        
        return (
            {bus: val for bus, val in zip(self.der_buses, p_opt)},
            {bus: val for bus, val in zip(self.der_buses, q_opt)}
        )