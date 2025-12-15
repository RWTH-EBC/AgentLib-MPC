"""
Converter module for ADDMO framework Keras models to AgentLib-MPC format.

This module provides functionality to convert Keras models trained with the ADDMO
framework into the AgentLib-MPC serialized format, similar to the physXAI plugin.
"""

import json
from pathlib import Path
from typing import Union, Optional


def validate_addmo_json(addmo_json: dict) -> None:
    """
    Validates that the ADDMO JSON contains all required fields.
    
    Args:
        addmo_json: The JSON metadata from ADDMO framework
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ["target_name", "features_ordered"]
    
    for field in required_fields:
        if field not in addmo_json:
            raise ValueError(
                f"ADDMO JSON is missing required field: '{field}'. "
                f"Available fields: {list(addmo_json.keys())}"
            )
    
    # Validate target_name is a string
    if not isinstance(addmo_json["target_name"], str):
        raise ValueError(
            f"'target_name' must be a string, got {type(addmo_json['target_name'])}"
        )
    
    # Validate features_ordered is a non-empty list
    if not isinstance(addmo_json["features_ordered"], list):
        raise ValueError(
            f"'features_ordered' must be a list, got {type(addmo_json['features_ordered'])}"
        )
    
    if len(addmo_json["features_ordered"]) == 0:
        raise ValueError("'features_ordered' cannot be empty")
    
    # Validate all features are strings
    for i, feature in enumerate(addmo_json["features_ordered"]):
        if not isinstance(feature, str):
            raise ValueError(
                f"All features in 'features_ordered' must be strings. "
                f"Feature at index {i} is {type(feature)}: {feature}"
            )


def infer_recursive_output(target_name: str, features_ordered: list[str]) -> bool:
    """
    Determines if the output feature is recursive (used as input).
    
    Args:
        target_name: Name of the output/target feature
        features_ordered: Ordered list of all features used in the model
        
    Returns:
        bool: True if target appears in features_ordered (recursive)
    """
    return target_name in features_ordered


def validate_keras_model_shape(
    keras_model_path: str,
    expected_features: list[str],
    feature_lags: dict[str, int]
) -> None:
    """
    Validates that the Keras model's input shape matches expected feature configuration.
    
    Args:
        keras_model_path: Path to the .keras model file
        expected_features: List of feature names in order
        feature_lags: Dictionary mapping feature names to their lag orders
        
    Raises:
        ValueError: If model input shape doesn't match lag configuration
        FileNotFoundError: If Keras model file doesn't exist
    """
    import keras
    
    keras_path = Path(keras_model_path)
    if not keras_path.exists():
        raise FileNotFoundError(
            f"Keras model file not found: {keras_model_path}"
        )
    
    # Load the model to inspect input shape
    model = keras.models.load_model(keras_model_path)
    
    # Get input shape (e.g., (None, 5) means batch_size, n_features)
    input_shape = model.input_shape
    
    if len(input_shape) != 2:
        raise ValueError(
            f"Expected Keras model with 2D input shape (batch_size, n_features), "
            f"but got shape with {len(input_shape)} dimensions: {input_shape}"
        )
    
    model_n_features = input_shape[1]
    
    # Calculate expected number of inputs based on lags
    # Each feature with lag=n contributes n inputs to the model
    expected_n_features = sum(feature_lags[feature] for feature in expected_features)
    
    if model_n_features != expected_n_features:
        raise ValueError(
            f"Keras model input shape mismatch!\n"
            f"  Model expects: {model_n_features} inputs\n"
            f"  Config expects: {expected_n_features} inputs\n"
            f"  Features: {expected_features}\n"
            f"  Feature lags: {feature_lags}\n"
            f"  Calculation: {' + '.join(f'{feature}({feature_lags[feature]})' for feature in expected_features)} = {expected_n_features}"
        )


def build_input_dict(
    features_ordered: list[str],
    target_name: str,
    feature_lags: dict[str, int],
    recursive: bool = False
) -> dict:
    """
    Builds the input dictionary for AgentLib-MPC format.
    
    Args:
        features_ordered: Ordered list of all features from ADDMO
        target_name: Name of the output/target feature
        feature_lags: Dict mapping feature names to lag orders (REQUIRED - must match training)
        recursive: Whether the output is recursive (used as input)
        
    Returns:
        dict: Input configuration in AgentLib format
              {"feature_name": {"name": "feature_name", "lag": 1}, ...}
        
    Raises:
        ValueError: If feature_lags is missing entries for features in features_ordered
    """
    input_dict = {}
    
    # Validate that all features have lag definitions
    for feature in features_ordered:
        if feature not in feature_lags:
            raise ValueError(
                f"Feature '{feature}' from features_ordered is missing in feature_lags. "
                f"All features must have lag definitions. "
                f"Provided lags: {list(feature_lags.keys())}"
            )
    
    for feature in features_ordered:
        # If recursive, the target appears in features_ordered but should NOT be in input dict
        # (AgentLib handles recursive outputs separately)
        if recursive and feature == target_name:
            continue
        
        input_dict[feature] = {
            "name": feature,
            "lag": feature_lags[feature]
        }
    
    return input_dict


def build_output_dict(
    target_name: str,
    output_lag: int = 1,
    output_type: str = "absolute",
    recursive: bool = True
) -> dict:
    """
    Builds the output dictionary for AgentLib-MPC format.
    
    Args:
        target_name: Name of the output/target feature
        output_lag: Lag order for the output feature
        output_type: Type of output ("absolute" or "difference")
        recursive: Whether output is recursive (used as input for next step)
        
    Returns:
        dict: Output configuration in AgentLib format
              {"target_name": {"name": "...", "lag": 1, "output_type": "...", "recursive": True}}
    """
    return {
        target_name: {
            "name": target_name,
            "lag": output_lag,
            "output_type": output_type,
            "recursive": recursive
        }
    }


def addmo_2_agentlib_json(
    keras_model_path: Union[str, Path],
    addmo_json: dict,
    dt: float,
    feature_lags: dict[str, int],
    output_lag: int = 1,
    output_type: str = "absolute",
    validate_model: bool = True
) -> dict:
    """
    Converts ADDMO Keras model configuration to AgentLib-MPC compatible JSON format.
    
    This is the main conversion function that orchestrates the transformation from
    ADDMO's minimal JSON format to AgentLib-MPC's comprehensive format required for
    model predictive control.
    
    Args:
        keras_model_path: Path to the .keras model file
        addmo_json: The JSON metadata from ADDMO framework containing:
                    - target_name: Output feature name
                    - features_ordered: Ordered list of input features
                    - (optional) preprocessing, instructions, etc.
        dt: Time step in seconds (e.g., 300 for 5 minutes)
        feature_lags: REQUIRED dict mapping feature names to their max lag order.
                      Must include ALL features from features_ordered.
                      Example: {"u_hp": 2, "T_amb": 1, "T": 1}
        output_lag: Lag order for the output feature (default: 1)
        output_type: Type of output - "absolute" or "difference" (default: "absolute")
        validate_model: Whether to validate Keras model shape against config (default: True)
        
    Returns:
        dict: The converted configuration in AgentLib-MPC JSON format with fields:
              - dt: Time step
              - model_type: "KerasANN"
              - model_path: Path to .keras file
              - input: Dict of input features with lags
              - output: Dict of output feature with metadata
              
    Raises:
        ValueError: If ADDMO JSON is invalid or model validation fails
        FileNotFoundError: If Keras model file doesn't exist
        
    Example:
        >>> addmo_json = {
        ...     "target_name": "T",
        ...     "features_ordered": ["u_hp", "T_amb", "rad_dir", "human_schedule", "T"]
        ... }
        >>> config = addmo_2_agentlib_json(
        ...     keras_model_path="models/model.keras",
        ...     addmo_json=addmo_json,
        ...     dt=300,
        ...     feature_lags={"u_hp": 1, "T_amb": 1, "rad_dir": 1, "human_schedule": 1, "T": 1}
        ... )
    """
    # Step 1: Validate ADDMO JSON structure
    validate_addmo_json(addmo_json)
    
    # Step 2: Extract key information
    target_name = addmo_json["target_name"]
    features_ordered = addmo_json["features_ordered"]
    
    # Step 3: Determine if output is recursive
    recursive = infer_recursive_output(target_name, features_ordered)
    
    # Step 4: Validate Keras model shape if requested
    if validate_model:
        validate_keras_model_shape(
            keras_model_path=str(keras_model_path),
            expected_features=features_ordered,
            feature_lags=feature_lags
        )
    
    # Step 5: Build input and output dictionaries
    input_dict = build_input_dict(
        features_ordered=features_ordered,
        target_name=target_name,
        feature_lags=feature_lags,
        recursive=recursive
    )
    
    output_dict = build_output_dict(
        target_name=target_name,
        output_lag=output_lag,
        output_type=output_type,
        recursive=recursive
    )
    
    # Step 6: Construct final AgentLib-MPC configuration
    agentlib_config = {
        "dt": dt,
        "model_type": "KerasANN",
        "model_path": str(keras_model_path),
        "input": input_dict,
        "output": output_dict
    }
    
    return agentlib_config


def load_addmo_model(
    keras_model_path: Union[str, Path],
    addmo_json_path: Union[str, Path],
    dt: float,
    feature_lags: dict[str, int],
    output_lag: int = 1,
    output_type: str = "absolute",
    validate_model: bool = True
) -> dict:
    """
    Convenience function to load both ADDMO JSON and convert to AgentLib format.
    
    Args:
        keras_model_path: Path to the .keras model file
        addmo_json_path: Path to the ADDMO JSON metadata file
        dt: Time step in seconds
        feature_lags: REQUIRED dict mapping feature names to lag orders
        output_lag: Lag order for output feature
        output_type: Type of output ("absolute" or "difference")
        validate_model: Whether to validate Keras model shape against config (default: True)
        
    Returns:
        dict: AgentLib-MPC compatible configuration
        
    Raises:
        FileNotFoundError: If paths don't exist
        ValueError: If JSON is invalid
    """
    # Load ADDMO JSON file
    json_path = Path(addmo_json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"ADDMO JSON file not found: {addmo_json_path}"
        )
    
    with open(json_path, 'r') as f:
        addmo_json = json.load(f)
    
    # Convert using main function
    return addmo_2_agentlib_json(
        keras_model_path=keras_model_path,
        addmo_json=addmo_json,
        dt=dt,
        feature_lags=feature_lags,
        output_lag=output_lag,
        output_type=output_type,
        validate_model=validate_model
    )