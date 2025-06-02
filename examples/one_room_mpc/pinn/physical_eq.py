import pandas as pd
import tensorflow as tf

def T_pred_max_change(data):
    """
    Simple constraint on maximum temperature change rate

    Args:
        data: Either a dictionary with inputs/outputs or a direct tensor

    Returns:
        Constrained temperature change based on thermal inertia
    """
    # Handle different input types
    if isinstance(data, dict):
        dT = data['outputs']['T']
    elif isinstance(data, pd.DataFrame):
        dT = data['T']
    else:
        dT = data  # Tensor during training

    # Apply simple physics constraint - thermal inertia principle
    max_change_rate = 1  # K per time step

    # Use tanh to create soft constraint
    return max_change_rate * tf.tanh(dT / max_change_rate)