# ADDMO Plugin Workflow

This directory provides scripts to integrate ADDMO with AgentLib-MPC. The workflow involves formatting simulation data, training a model in ADDMO, and generating an AgentLib-MPC compatible configuration.

### 1. Data Conversion (`convert_csv.py`) *(Optional)*
Converts standard AgentLib `simulation_data.csv` into an ADDMO-compatible format. This step is optional; you can use any model trained in ADDMO directly in Step 3 as long as you know its sampling time (`dt`).

**Values to set before running (in the `__main__` block):**
- `input_file`: Path to your AgentLib `simulation_data.csv`.
- `target`: The variable to predict (e.g., `"T"`).
- `output_type`: Prediction type, either `"absolute"` (next value) or `"difference"` (delta).
- `wanted_sampling_rate`: Target sampling interval `dt` in seconds (e.g., `60`).
- `exclude_vars`: List of variable names to exclude from the dataset.
- `lags`: Dictionary specifying the total number of lags per feature (e.g., `{'mDot': 2, 'T': 5}`).

### 2. Train Model in ADDMO
Use the newly generated `*_addmo.csv` to train your model within the ADDMO framework.

### 3. Model Configuration Creation (`model_config_creation.py`)
Converts the ADDMO model artifacts into an AgentLib-MPC configuration.

**Values to set before running (in the `main()` function):**
- `source_folder`: Path containing the ADDMO output (must include `ml_model.keras` and `ml_model_metadata.json`).
- `target_folder`: Destination directory to save the AgentLib-MPC configuration and model.
- `dt`: The sampling rate matching your model and setup (e.g., `60`).

### 4. Integration with AgentLib-MPC
Use the resulting JSON configuration file by adding it to the `ml_model_sources` entry in your optimization backend configuration. This will plug the learned dynamics into the CasADi predictor.
