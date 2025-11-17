import os
import sys
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from agentlib_mpc.models.casadi_predictor import FunctionalWrapper
from agentlib.core.errors import OptionalDependencyError
import pandas as pd
import numpy as np
try:
    from physXAI import models  # Keep this import to ensure physXAI models are registered
except ImportError:
    raise OptionalDependencyError(dependency_name="physXAI", dependency_install="git+https://github.com/RWTH-EBC/physXAI.git", used_object="physXAI")


thresholds = {
    "3.9": {"rel_error": 0.25, "abs_diff": 5e-5},
    "3.10": {"rel_error": 0.25, "abs_diff": 5e-5},
    "3.11": {"rel_error": 0.25, "abs_diff": 5e-5},
    "3.12": {"rel_error": 0.25, "abs_diff": 5e-5},
}


def test_casadi_predictor(monkeypatch):
    monkeypatch.chdir(Path(__file__).parent)

    # Load models
    model_keras = keras.models.load_model('models/model.keras')
    model_casadi = FunctionalWrapper(model_keras)

    # Load sample data
    data = pd.read_csv('data/sample_data.csv', sep=';', index_col=0)

    # Separate features and target
    X = data[['x1', 'x2', 'x3']].values

    # Make predictions with both models
    y_keras = model_keras.predict(X, verbose=0).flatten()

    # Predict with casadi model
    y_casadi = np.array([float(model_casadi.forward(X[i, :])) for i in range(len(X))])

    # Calculate differences
    abs_diff = np.abs(y_keras - y_casadi)
    rel_error = (abs_diff / np.abs(y_keras)) * 100

    # Define thresholds per Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Use 3.12 thresholds as default for other versions
    threshold = thresholds.get(python_version, thresholds["3.12"])
    
    max_rel_error = max(rel_error)
    max_abs_diff = max(abs_diff)
    assert max_rel_error < threshold["rel_error"], (
        f"Relative error too high: expected < {threshold['rel_error']}%, "
        f"got {max_rel_error:.4f}% (Python {python_version})"
    )
    assert max_abs_diff < threshold["abs_diff"], (
        f"Absolute difference too high: expected < {threshold['abs_diff']}, "
        f"got {max_abs_diff:.6e} (Python {python_version})"
    )