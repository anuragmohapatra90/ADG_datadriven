# active_grid/sr_model.py
import os
import joblib
from pysr import PySRRegressor
from . import config

class SRVoltageEstimator:
    """
    A Symbolic Regression Wrapper to act as a Surrogate Model.
    Wraps PySRRegressor to provide an interface compatible with the controller.
    """
    def __init__(self, model_path=None, **kwargs):
        self.model_path = model_path
        
        # 1. Define default parameters
        params = {
            "niterations": 5,          
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["inv(x) = 1/x", "square", "sqrt"],
            "extra_sympy_mappings": {"inv": lambda x: 1/x},
            "elementwise_loss": "loss(prediction, target) = (prediction - target)^2", 
            "model_selection": "best",
            "output_directory": "sr_logs",
            "verbosity": 0,            
            "warm_start": True         
        }
        
        # 2. Update defaults
        params.update(kwargs)

        # 3. Load or Initialize
        if model_path:
            if os.path.exists(model_path):
                print(f"Loading SR model from {model_path}...")
                try:
                    # UPDATED: Load directly with joblib (more robust)
                    self.model = joblib.load(model_path)
                except Exception as e:
                    # Fallback to PySR native loader if joblib fails (backward compatibility)
                    print(f"Joblib load failed, trying PySR native loader: {e}")
                    self.model = PySRRegressor.from_file(model_path)
            else:
                # Raise error if specifically asked for a model that doesn't exist
                # (unless we are just initializing a new one for training)
                if 'warm_start' not in kwargs: 
                    raise FileNotFoundError(f"SR Model file not found at '{model_path}'. Have you run 'train_sr.py'?")
                else:
                    self.model = PySRRegressor(**params)
        else:
            self.model = PySRRegressor(**params)

    def fit(self, X, y, **kwargs):
        """Fits the symbolic regression model."""
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        """Predicts voltages given inputs X (P, Q)."""
        return self.model.predict(X)

    def get_best_equations(self):
        """Returns a list of the best equation objects (one per output)."""
        return self.model.get_best()

    def save_pickle(self, destination_path):
        """
        UPDATED: Explicitly dumps the model object using joblib.
        This works even if PySR hasn't created its internal checkpoint file yet.
        """
        print(f"Saving SR model to {destination_path}...")
        joblib.dump(self.model, destination_path)
        print("Save complete.")