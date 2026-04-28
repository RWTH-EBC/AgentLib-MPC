import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional


# ADDMO uses "___lag{number}" pattern (three underscores)
ADDMO_LAG_PATTERN = r"___lag(\d+)$"
# AgentLib-MPC uses "_{number}" pattern (single underscore at end)
AGENTLIB_LAG_PATTERN = r"_(\d+)$"


def extract_lags_from_features(features_ordered: list[str]) -> dict[str, int]:
    # First pass: group features and collect all lag indices
    feature_groups = defaultdict(list)
    
    for idx, feature_name in enumerate(features_ordered):
        # Try ADDMO pattern first
        addmo_match = re.search(ADDMO_LAG_PATTERN, feature_name)
        if addmo_match:
            lag_num = int(addmo_match.group(1))
            base_name = feature_name[:addmo_match.start()]
            feature_groups[base_name].append((lag_num, idx, feature_name))
            continue
        
        # Try AgentLib pattern
        agentlib_match = re.search(AGENTLIB_LAG_PATTERN, feature_name)
        if agentlib_match:
            suffix_num = int(agentlib_match.group(1))
            base_name = feature_name[:agentlib_match.start()]
            # AgentLib convention: suffix directly represents lag index (feature_1 = lag_1, feature_2 = lag_2)
            lag_num = suffix_num
            feature_groups[base_name].append((lag_num, idx, feature_name))
            continue
        
        # No lag suffix means lag_0 (current time step)
        feature_groups[feature_name].append((0, idx, feature_name))

    print("Extracted feature groups with lags:")
    print(feature_groups)
    
    # Second pass: validate consecutive lags and compute lag counts
    feature_lags = {}
    
    for base_name, lag_list in feature_groups.items():
        # Sort by original position first to check ascending lag order
        lag_list_by_position = sorted(lag_list, key=lambda x: x[1])
        
        # Check lags appear in ascending order in the original features_ordered
        for i in range(len(lag_list_by_position) - 1):
            current_lag = lag_list_by_position[i][0]
            next_lag = lag_list_by_position[i + 1][0]
            if next_lag != current_lag + 1:
                raise ValueError(
                    f"Feature '{base_name}' lags must appear in ascending order (lag_0, lag_1, ...) in features_ordered. "
                    f"Found lag_{current_lag} followed by lag_{next_lag}. "
                    f"Features: {[fname for _, _, fname in lag_list_by_position]}"
                )
        
        # Now sort by lag number for remaining validations
        lag_list.sort(key=lambda x: x[0])
        lag_numbers = [lag_num for lag_num, _, _ in lag_list]
        
        # Validate consecutive lags starting from 0
        if lag_numbers[0] != 0:
            raise ValueError(
                f"Feature '{base_name}' lags must start from 0 (current time step). "
                f"Found lags: {lag_numbers}. "
                f"Features: {[fname for _, _, fname in lag_list]}"
            )
        
        # Check for consecutive lags (no gaps)
        for i in range(len(lag_numbers) - 1):
            if lag_numbers[i + 1] != lag_numbers[i] + 1:
                raise ValueError(
                    f"Feature '{base_name}' has non-consecutive lags. "
                    f"Found lags: {lag_numbers}. "
                    f"Missing lag_{lag_numbers[i] + 1}. "
                    f"Features: {[fname for _, _, fname in lag_list]}"
                )
        
        # Check that all lags are grouped consecutively in features_ordered
        original_indices = [idx for _, idx, _ in lag_list]
        for i in range(len(original_indices) - 1):
            if original_indices[i + 1] != original_indices[i] + 1:
                raise ValueError(
                    f"Feature '{base_name}' lags are not grouped consecutively in features_ordered. "
                    f"AgentLib requires all lags of a feature to appear together. "
                    f"Found at positions {original_indices}. "
                    f"Features: {[fname for _, _, fname in lag_list]}"
                )
        
        # Lag count is the number of lag instances
        feature_lags[base_name] = len(lag_numbers)
    
    return feature_lags


def validate_addmo_json(addmo_json: dict) -> None:

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


def build_input_dict(
    features_ordered: list[str],
    target_name: str,
    feature_lags: dict[str, int],
    recursive: bool = False
) -> dict:
    """
    Builds input dict preserving the order from features_ordered.
    
    Note: features_ordered may contain lag suffixes (e.g., "T_amb___lag0", "T_amb___lag1")
    but we need to extract base names and ensure they appear in input_dict in the same
    grouped order (all lags of T_amb, then all lags of next feature).
    """
    input_dict = {}
    seen_features = set()
    
    for feature_with_lag in features_ordered:
        # Extract base feature name (strip lag suffixes)
        base_name = feature_with_lag
        for pattern in [ADDMO_LAG_PATTERN, AGENTLIB_LAG_PATTERN]:
            match = re.search(pattern, feature_with_lag)
            if match:
                base_name = feature_with_lag[:match.start()]
                break
        
        # Skip if we've already added this base feature or if it's the recursive target
        if base_name in seen_features:
            continue
        if recursive and base_name == target_name:
            continue
        
        # Validate that base feature has lag definition
        if base_name not in feature_lags:
            raise ValueError(
                f"Feature '{base_name}' (from '{feature_with_lag}') is missing in feature_lags. "
                f"All features must have lag definitions. "
                f"Provided lags: {list(feature_lags.keys())}"
            )
        
        # Add to input_dict (preserves order via dict insertion order)
        input_dict[base_name] = {
            "name": base_name,
            "lag": feature_lags[base_name]
        }
        seen_features.add(base_name)
    
    return input_dict


def build_output_dict(
    target_name: str,
    output_lag: int = 1,
    output_type: str = "absolute",
    recursive: bool = True
) -> dict:
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
    addmo_json_path: Union[str, Path],
    dt: float,
    output_type: str = "absolute",
    target_folder: Optional[Union[str, Path]] = None
) -> dict:

    keras_model_path = Path(keras_model_path)
    json_path = Path(addmo_json_path)
    
    if target_folder:
        target_dir = Path(target_folder)
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = keras_model_path.parent
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"ADDMO JSON file not found: {addmo_json_path}"
        )
    
    import keras
    import numpy as np
    import shutil
    model = keras.models.load_model(keras_model_path)
    last_layer = model.layers[-1]
    
    if last_layer.__class__.__name__ == 'Normalization' and getattr(last_layer, 'invert', False):
        print(f"Replacing reverted Normalization layer {last_layer.name} with Rescaling layer.")
        mean = last_layer.mean.numpy()
        variance = last_layer.variance.numpy()
        scale = np.sqrt(variance)
        
        # Extract single values if they are stored as arrays
        if hasattr(scale, "shape") and len(scale.shape) > 0:
            scale = float(scale.item() if scale.size == 1 else scale.flatten()[-1])
            mean = float(mean.item() if mean.size == 1 else mean.flatten()[-1])

        rescaling_layer = keras.layers.Rescaling(scale=scale, offset=mean, name="fixed_rescaling")
        new_output = rescaling_layer(model.layers[-2].output)
        new_model = keras.Model(inputs=model.inputs, outputs=new_output)
        
        new_model_path = target_dir / f"{keras_model_path.stem}_fixed_norm{keras_model_path.suffix}"
        new_model.save(new_model_path)
        final_model_path = new_model_path
        print(f"Saved modified model to {new_model_path}")
    else:
        print(f"No reverted Normalization layer found at the end. Last layer is {last_layer.__class__.__name__}.")
        final_model_path = target_dir / keras_model_path.name
        if keras_model_path != final_model_path:
            shutil.copy2(keras_model_path, final_model_path)
            print(f"Copied original model to {final_model_path}")

    with open(json_path, 'r') as f:
        addmo_json = json.load(f)
    
    validate_addmo_json(addmo_json)
    
    target_name = addmo_json["target_name"]
    features_ordered = addmo_json["features_ordered"]
    
    feature_lags = extract_lags_from_features(features_ordered)
    
    # Detect if target has special suffix indicating state prediction
    # _absolute: absolute next-step prediction (recursive=True, output_type=absolute)
    # _difference: difference prediction (recursive=True, output_type=difference)
    base_target_name = target_name
    detected_output_type = output_type
    
    if target_name.endswith("_absolute"):
        base_target_name = target_name[:-9]
        detected_output_type = "absolute"

    elif target_name.endswith("_difference"):
        base_target_name = target_name[:-11]
        detected_output_type = "difference"

    # AgentLib-MPC limitation: recursive target features must be at the end
    if base_target_name in feature_lags:
        num_target_lags = feature_lags[base_target_name]
        expected_target_features = features_ordered[-num_target_lags:]
        for feat in expected_target_features:
            base_name = feat
            for pattern in [ADDMO_LAG_PATTERN, AGENTLIB_LAG_PATTERN]:
                match = re.search(pattern, feat)
                if match:
                    base_name = feat[:match.start()]
                    break
            if base_name != base_target_name:
                raise ValueError(
                    f"AgentLib-MPC currently requires the recursive target '{base_target_name}' "
                    f"to be the last feature(s) in 'features_ordered'. "
                    f"Please provide a model with reordered features."
                )
    
    # Check if base target (not the suffixed version) is in feature_lags to determine recursiveness
    recursive = base_target_name in feature_lags


    # Count how many lag instances of the base target exist
    output_lag = feature_lags.get(base_target_name, 1) if recursive else 1
    
    input_dict = build_input_dict(
        features_ordered=features_ordered,
        target_name=base_target_name,  # Use base name, not _next
        feature_lags=feature_lags,
        recursive=recursive
    )
    
    output_dict = build_output_dict(
        target_name=base_target_name,  # Use base name, not _absolute/_difference
        output_lag=output_lag,
        output_type=detected_output_type,  # Use detected output_type from suffix
        recursive=recursive
    )
    
    agentlib_config = {
        "dt": dt,
        "model_type": "KerasANN",
        "model_path": str(final_model_path.absolute()), # Absolute path avoids execution directory issues
        "input": input_dict,
        "output": output_dict
    }
    
    return agentlib_config


def main():
    # Fill out/check these parameters before running the script
    source_folder = Path(r"C:\Users\sle-fmu\Desktop\Git\ADDMo\tmp\agentlib_keras_train") # Addmo path with files ml_model.keras and ml_model_metadata.json
    target_folder = Path(r"C:\Users\sle-fmu\Desktop\Git\AgentLib-MPC\examples\one_room_mpc\addmo_plugin\keras") # Agentlib-MPC path
    dt = 60
    
    keras_path = source_folder / "ml_model.keras"
    json_path = source_folder / "ml_model_metadata.json"

    if not keras_path.exists() or not json_path.exists():
        print(f"Files not found in {source_folder}")
        return
    
    print(f"Creating AgentLib-MPC config from ADDMO files in {source_folder}...")

    agentlib_config = addmo_2_agentlib_json(
        keras_model_path=keras_path,
        addmo_json_path=json_path,
        dt=dt,
        target_folder=target_folder
    )

    with open(target_folder / "keras_ann.json", 'w') as f:
        json.dump(agentlib_config, f, indent=2)

    print("Generated AgentLib-MPC Config:")
    print(agentlib_config)

if __name__ == "__main__":
    main()