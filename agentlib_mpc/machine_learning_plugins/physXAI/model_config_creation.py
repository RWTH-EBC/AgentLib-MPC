import re
from collections import defaultdict
import joblib


output_type_pattern = r"Change\((.*)\)"  # Pattern to differentiate between 'absolute' and 'difference' outputs
lag_pattern = r"_lag(\d+)$"  # Pattern to extract lag information from feature names
preprocessing_training_info = ["test_size", "val_size", "random_state"]  # Define relevant preprocessing info keys


def model_path_generation(run_id: str, output_name: str) -> str:
    """
    Generates the relative model path based on run_id and output_name.
    Args:
        run_id (str): The unique identifier for the mpc run.
        output_name (str): The name of the output feature.
    Returns:
        str: The relative path to the model file.
    """

    return f"models/{run_id}/{output_name}"


def physXAI_2_agentlib_json(run_id: str, preprocessing_dict: dict, model_dict: dict = None, training_dict: dict = None,
                           model_type: str = 'ANN') -> dict:
    """
    Converts physXAI model configurations to an AgentLib-MPC compatible JSON format.
    Args:
        run_id (str): The unique identifier for the mpc run.
        preprocessing_dict (dict): The preprocessing configuration from physXAI.
        model_dict (dict, optional): The model configuration from physXAI. Defaults to None.
        training_dict (dict, optional): The training configuration from physXAI. Defaults to None.
        model_type (str, optional): The type of model ('ANN' or 'LinReg'). Defaults to 'ANN'.
    Returns:
        dict: The converted configuration in AgentLib-MPC JSON format.
    """

    # Initialize target dictionary
    target_dict = {
        "dt": preprocessing_dict["time_step"],
        "input": {},
        "output": {},
        "agentlib_mpc_hash": f"physXAI",
        "training_info": {"preprocessing": {}, "model": {}, "training": {}},
    }

    # Preprocessing Info
    for info in preprocessing_training_info:
        if info in preprocessing_dict:
            target_dict["training_info"]["preprocessing"][info] = preprocessing_dict[info]

    # Model Info
    if model_dict is not None:
            target_dict["training_info"]["model"] = model_dict

    # Training Info
    if training_dict is not None:
            target_dict["training_info"]["training"] = training_dict

    # Validate and extract shift and output
    default_shift = preprocessing_dict.get("shift", 1)
    if default_shift != 1:
        raise ValueError(f"Config Translation Error: Shift should be 1 to be used in AgentLib, but was {default_shift}")
    if not isinstance(preprocessing_dict.get("output"), list) or len(preprocessing_dict["output"]) != 1:
        raise ValueError("Config Translation Error: Output should be a list with 1 element")

    # Extract output type
    output_str = preprocessing_dict["output"][0]
    output_type = "absolute"
    output_full_feature_name = output_str
    change_match = re.match(output_type_pattern, output_str)
    if change_match:
        output_type = "difference"
        output_full_feature_name = change_match.group(1).strip()
    output_key_name = output_full_feature_name

    # Extract inputs
    grouped_inputs = defaultdict(list)
    for i, input_str in enumerate(preprocessing_dict["inputs"]):
        lag = default_shift
        feature_base_str = input_str

        lag_match = re.search(lag_pattern, input_str)
        if lag_match:
            lag_value = int(lag_match.group(1))
            lag = default_shift + lag_value
            feature_base_str = input_str[:lag_match.start()]
        base_name = feature_base_str

        grouped_inputs[base_name].append({'original_index': i, 'lag': lag, 'full_name': input_str})

    # Validate and extract lags
    for base_name, items in grouped_inputs.items():
        if len(items) > 1:
            items.sort(key=lambda x: x['original_index'])
            for j in range(len(items) - 1):
                current_item = items[j]
                next_item = items[j + 1]
                if next_item['original_index'] != current_item['original_index'] + 1:
                    raise ValueError(
                        f"Config Translation Error: Features for '{base_name}' are not grouped consecutively. "
                        f"Found '{current_item['full_name']}' at Index {current_item['original_index']} and "
                        f"'{next_item['full_name']}' at Index {next_item['original_index']}."
                    )

                if next_item['lag'] != current_item['lag'] + 1:
                    raise ValueError(
                        f"Config Translation Error: Lags for '{base_name}' are not in ascending order. "
                        f"Lag {current_item['lag'] - 1} is followed by {next_item['lag'] - 1}."
                    )

    # Generate final inputs
    for base_name, items in grouped_inputs.items():
        max_lag = max(item['lag'] for item in items)
        target_dict["input"][base_name] = {
            "name": base_name,
            "lag": max_lag
        }

    # Validate and extract output
    recursive = False
    num_recursive_inputs = 1
    if output_key_name in target_dict["input"]:
        recursive = True

        recursive_inputs = grouped_inputs[output_key_name]
        num_recursive_inputs = len(recursive_inputs)
        total_inputs = len(preprocessing_dict["inputs"])
        expected_indices = list(range(total_inputs - num_recursive_inputs, total_inputs))
        actual_indices = [item['original_index'] for item in recursive_inputs]
        if expected_indices != actual_indices:
            raise ValueError(
                f"Config Translation Error: Recursive Feature '{output_key_name}' and its Lags must be at the end of the "
                f"'inputs'-List. Expected Indices: {expected_indices}, "
                f"Actual Indices: {actual_indices}."
            )
        target_dict["input"].pop(output_key_name)

    # Generate final output
    target_dict["output"][output_key_name] = {
        "name": output_key_name,
        "lag": num_recursive_inputs,
        "output_type": output_type,
        "recursive": recursive
    }

    # LinReg is saved as normal AgentLib json with parameters, ANN as keras model path
    if model_type == 'LinReg' or (model_dict is not None and model_dict['__class_name__'] == 'LinearRegressionModel'):
        target_dict["model_type"] = "LinReg"

        load_path = model_path_generation(run_id, output_key_name) + '.joblib'
        model = joblib.load(load_path)
        target_dict["parameters"] = {
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "n_features_in": model.n_features_in_,
            "rank": model.rank_,
            "singular": model.singular_.tolist(),
        }

    else:
        target_dict["model_type"] = "KerasANN"
        target_dict["model_path"] = model_path_generation(run_id, output_key_name) + '.keras'

    return target_dict