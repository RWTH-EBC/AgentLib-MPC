"""
Simple test to verify ADDMO to AgentLib-MPC config conversion.
"""

import json
from pathlib import Path

from agentlib_mpc.machine_learning_plugins.addmo import model_config_creation

addmo_2_agentlib_json = model_config_creation.addmo_2_agentlib_json



def main():
    """Test ADDMO to AgentLib-MPC config conversion."""
    # Your actual ADDMO JSON
    addmo_json = {
        "addmo_class": "SciKerasSequential",
        "addmo_commit_id": "8f888ad9",
        "library": "keras",
        "library_model_type": "Sequential",
        "library_version": "3.11.0",
        "target_name": "T",
        "features_ordered": ["u_hp", "T_amb", "rad_dir", "human_schedule", "T"],
        "preprocessing": ["Scaling as layer of the ANN."],
        "instructions": "Pass a single or multiple observations with features in the order listed above"
    }

    # Define lags (you need to provide these based on your training)
    feature_lags = {
        "u_hp": 1,
        "T_amb": 1,
        "rad_dir": 1,
        "human_schedule": 1,
        "T": 1
    }

    # Convert to AgentLib format (skip model validation since we don't have a real .keras file)
    agentlib_config = addmo_2_agentlib_json(
        keras_model_path="path/to/your/model.keras",
        addmo_json=addmo_json,
        dt=300,  # 5 minutes
        feature_lags=feature_lags,
        output_lag=1,
        output_type="absolute",
        validate_model=False  # Skip Keras validation for this test
    )

    # Print the result
    print("Generated AgentLib-MPC Config:")
    print(json.dumps(agentlib_config, indent=2))


if __name__ == "__main__":
    main()
