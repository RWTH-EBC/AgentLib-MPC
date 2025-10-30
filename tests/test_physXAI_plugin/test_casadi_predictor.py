import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from agentlib_mpc.models.casadi_predictor import FunctionalWrapper
from physXAI import models
import pandas as pd
import numpy as np


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

    assert max(rel_error) < 0.1
    assert max(abs_diff) < 5e-5