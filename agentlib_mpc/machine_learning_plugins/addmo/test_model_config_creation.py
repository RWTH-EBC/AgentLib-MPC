"""
Simple test to verify ADDMO to AgentLib-MPC config conversion.
"""

import json
import tempfile
from pathlib import Path

from agentlib_mpc.machine_learning_plugins.addmo import model_config_creation

addmo_2_agentlib_json = model_config_creation.addmo_2_agentlib_json



def main():
    """Test ADDMO to AgentLib-MPC config conversion."""
    # ADDMO JSON with lag notation (___lag0, ___lag1, etc.)
    # Features without ___lag suffix are treated as current time step only (lag=1)
    addmo_json = {
        "addmo_class": "SciKerasSequential",
        "addmo_commit_id": "8f888ad9",
        "library": "keras",
        "library_model_type": "Sequential",
        "library_version": "3.11.0",
        "target_name": "T",
        "features_ordered": [
            "u_hp___lag0",              # u_hp: only current time step
            "T_amb___lag0",             # T_amb: current + 1 lag
            "T_amb___lag1",
            "rad_dir",                  # rad_dir: no suffix = current time step only
            "human_schedule___lag0",    # human_schedule: only current time step
            "T___lag0"                  # T: recursive output, current time step
        ],
        "preprocessing": ["Scaling as layer of the ANN."],
        "instructions": "Pass a single or multiple observations with features in the order listed above"
    }

    # Create a temporary JSON file for the ADDMO config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(addmo_json, f)
        addmo_json_path = f.name

    try:
        # Lags are now AUTO-EXTRACTED from features_ordered! No need to specify manually.
        # Convert to AgentLib format (skip model validation since we don't have a real .keras file)
        agentlib_config = addmo_2_agentlib_json(
            keras_model_path="path/to/your/model.keras",
            addmo_json_path=addmo_json_path,
            dt=300,  # 5 minutes
            output_type="absolute",
            validate_model=False  # Skip Keras validation for this test
        )

        # Print the result
        print("Generated AgentLib-MPC Config:")
        print(json.dumps(agentlib_config, indent=2))
    finally:
        # Clean up temporary file
        Path(addmo_json_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
