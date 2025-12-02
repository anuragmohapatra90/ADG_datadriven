import numpy as np
import torch
import joblib
from scipy.optimize import minimize
from . import config
# Import BOTH model wrappers
from .mlp_model import VoltageEstimator as MLP_Model
from .sr_model import SRVoltageEstimator as SR_Model

class MPC_Controller:
    def __init__(self, model_type="mlp", scaler_path="scalers.pkl"):
        """
        Initialize controller with a specific surrogate model type.
        model_type: "mlp" or "sr"
        """
        self.model_type = model_type.lower()
        
        # 1. Load Scalers (Shared)
        try:
            scalers = joblib.load(scaler_path)
            self.scaler_X = scalers['X']
            self.scaler_y = scalers['y']
        except FileNotFoundError:
            raise FileNotFoundError(f"Scalers not found at {scaler_path}. Run training first.")
        
        # 2. Load the Specific Model
        if self.model_type == "mlp":
            try:
                self.model = MLP_Model(input_dim=2*config.BUS_COUNT, output_dim=config.BUS_COUNT)
                # Assuming mlp_model.pth is in root
                self.model.load_state_dict(torch.load("mlp_model.pth"))
                self.model.eval()
            except FileNotFoundError:
                raise FileNotFoundError("mlp_model.pth not found. Run train_mlp.py.")
                
        elif self.model_type == "sr":
            try:
                # Assuming sr_model.pkl is in root
                self.model = SR_Model(model_path="sr_model.pkl")
            except Exception as e:
                raise RuntimeError(f"Failed to load SR model. Run train_sr.py. Error: {e}")
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp' or 'sr'.")
        
        self.der_buses = config.DER_BUSES
        self.n_der = len(self.der_buses)
        
    def predict_voltages(self, load_p, load_q, der_p_setpoints, der_q_setpoints):
        """
        Polymorphic prediction function handling both Torch (MLP) and Numpy (SR).
        """
        # Prepare Input Vector
        p_vector = load_p.copy()
        q_vector = load_q.copy()
        
        for i, bus_idx in enumerate(self.der_buses):
            p_vector[bus_idx] -= der_p_setpoints[i]
            q_vector[bus_idx] -= der_q_setpoints[i]
            
        features = np.concatenate([p_vector, q_vector])
        features_scaled = self.scaler_X.transform(features.reshape(1, -1))
        
        # Branch based on model type
        if self.model_type == "mlp":
            with torch.no_grad():
                tensor_in = torch.tensor(features_scaled.astype(np.float32))
                pred_scaled = self.model(tensor_in).numpy()
        else: # SR
            # SR model expects numpy, returns numpy
            pred_scaled = self.model.predict(features_scaled)
            
        pred_pu = self.scaler_y.inverse_transform(pred_scaled)
        return pred_pu.flatten()

    def get_optimal_setpoints(self, current_load_p, current_load_q, weight_p=1.0, weight_q=0.1, weight_v=1000.0):
        x0 = np.zeros(2 * self.n_der)
        
        def objective(x):
            p_ctrl = x[:self.n_der]
            q_ctrl = x[self.n_der:]
            
            # Cost 1: Actuation Effort
            cost_ctrl = weight_p * np.sum(p_ctrl**2) + weight_q * np.sum(q_ctrl**2)
            
            # Cost 2: Voltage Violation (Soft Constraint)
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