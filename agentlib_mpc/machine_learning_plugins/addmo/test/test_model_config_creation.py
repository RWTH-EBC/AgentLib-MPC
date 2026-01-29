"""
Simple test to verify ADDMO to AgentLib-MPC config conversion.
"""

import json
import tempfile
from pathlib import Path

from agentlib_mpc.machine_learning_plugins.addmo import model_config_creation

addmo_2_agentlib_json = model_config_creation.addmo_2_agentlib_json


def test_synthetic():
    """Test with synthetic data."""
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
        # Convert to AgentLib format (skip model validation since we don't have a real .keras file)
        agentlib_config = addmo_2_agentlib_json(
            keras_model_path="path/to/your/model.keras",
            addmo_json_path=addmo_json_path,
            dt=300,  # 5 minutes
            output_type="absolute"
        )

        # Print the result
        print("Generated AgentLib-MPC Config:")
        print(json.dumps(agentlib_config, indent=2))
    finally:
        # Clean up temporary file
        Path(addmo_json_path).unlink(missing_ok=True)


def test_real_files(test_dir: Path):
    """Test with real files in the test directory."""
    keras_path = test_dir / "best_model_test.keras"
    json_path = test_dir / "best_model_metadata_test.json"

    if not keras_path.exists() or not json_path.exists():
        print(f"Skipping real file test: Files not found in {test_dir}")
        return

    print(f"\nTesting with real files from {test_dir}...")
    
    agentlib_config = addmo_2_agentlib_json(
        keras_model_path=keras_path,
        addmo_json_path=json_path,
        dt=300,
        output_type="absolute"
    )

    print("Generated AgentLib-MPC Config (from real files):")
    print(json.dumps(agentlib_config, indent=2))


def main():
    
    test_synthetic()

    test_dir = Path(__file__).parent
    test_real_files(test_dir)


if __name__ == "__main__":
    main()
