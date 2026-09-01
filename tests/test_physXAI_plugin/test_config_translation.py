from pathlib import Path
import pytest
from agentlib_mpc.machine_learning_plugins.physXAI.model_config_creation import physXAI_2_agentlib_json
from agentlib_mpc.models.serialized_ml_model import SerializedKerasRNN, SerializedMLModel


def test_physXAI_2_agentlib_json(monkeypatch):
    monkeypatch.chdir(Path(__file__).parent)

    source_1 = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "QTabs_set",
            "T_ahu_set",
            "TDryBul",
            "HDirNor",
            "T_room"
        ],
        "output": [
            "Change(T_room)"
        ],
        "time_step": 1,
        "shift": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    }

    source_2 = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "QTabs_set",
            "TDryBul",
            "TDryBul_lag1",
            "TDryBul_lag2",
            "HDirNor"
        ],
        "output": [
            "T_room"
        ],
        "time_step": 1,
        "shift": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    }

    source_3_error_order = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "TDryBul",
            "QTabs_set",
            "TDryBul_lag1"
        ],
        "time_step": 1,
        "output": ["T_room"],
        "shift": 1
    }

    source_4_error_recursive_pos = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "T_room",
            "QTabs_set",
            "T_ahu_set"
        ],
        "output": ["Change(T_room)"],
        "time_step": 1,
        "shift": 1
    }

    source_5_error_shift = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "QTabs_set",
            "T_ahu_set",
            "TDryBul",
            "HDirNor",
            "T_room"
        ],
        "output": [
            "Change(T_room)"
        ],
        "time_step": 1,
        "shift": 2,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    }

    source_6_error_output_list = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "QTabs_set",
            "T_ahu_set",
            "TDryBul",
            "HDirNor",
            "T_room"
        ],
        "output": "Change(T_room)",
        "time_step": 1,
        "shift": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    }

    source_7_error_output_list_len = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "QTabs_set",
            "T_ahu_set",
            "TDryBul",
            "HDirNor",
            "T_room"
        ],
        "output": ["Change(T_room)", "T_room"] ,
        "time_step": 1,
        "shift": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    }

    source_8_error_lag_order = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "QTabs_set",
            "TDryBul",
            "TDryBul_lag2",
            "TDryBul_lag1",
            "HDirNor"
        ],
        "output": [
            "T_room"
        ],
        "time_step": 1,
        "shift": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    }

    source_1_linreg = {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": [
            "QTabs_set",
            "T_ahu_set",
            "TDryBul",
            "HDirNor",
            "T_room"
        ],
        "output": [
            "T_room"
        ],
        "time_step": 1,
        "shift": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    }

    model_dict =  {
        "__class_name__": "ClassicalANNModel",
        "batch_size": 32,
    }

    training_dict = {
        "metrics": {
            "train_kpis": {
                "MSE Train": 3457.159518090897,
                "RMSE Train": 58.79761490137927,
                "R2 Train": 0.9981096371519428
            },
        }
    }

    physXAI_2_agentlib_json('01', source_1, model_dict=model_dict, training_dict=training_dict)

    physXAI_2_agentlib_json('02', source_2)

    with pytest.raises(ValueError):
        physXAI_2_agentlib_json('03', source_3_error_order)

    with pytest.raises(ValueError):
        physXAI_2_agentlib_json('04', source_4_error_recursive_pos)

    with pytest.raises(ValueError):
        physXAI_2_agentlib_json('05', source_5_error_shift)

    with pytest.raises(ValueError):
        physXAI_2_agentlib_json('06', source_6_error_output_list)

    with pytest.raises(ValueError):
        physXAI_2_agentlib_json('07', source_7_error_output_list_len)

    with pytest.raises(ValueError):
        physXAI_2_agentlib_json('08', source_8_error_lag_order)

    physXAI_2_agentlib_json('01', source_1_linreg, model_type='LinReg')


def multi_step_source(**overrides) -> dict:
    """Preprocessing config physXAI writes for a multi step (recurrent) model."""
    source = {
        "__class_name__": "PreprocessingMultiStep",
        "inputs": [
            "TDryBul",
            "HDirNor",
            "QTabs_set",
        ],
        "output": ["T_room"],
        "label_width": 48,
        "warmup_width": 24,
        "init_features": ["T_room"],
        "warmup_columns_input": [],
        "warmup_columns_labels": ["T_room"],
        "overlapping_sequences": True,
        "batch_size": 32,
        "time_step": 900,
        "shift": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42,
    }
    source.update(overrides)
    return source


def test_physXAI_multi_step_2_agentlib_json():
    """A multi step config becomes a 'KerasRNN' config without lags."""
    config = physXAI_2_agentlib_json(
        "01",
        multi_step_source(),
        model_dict={"__class_name__": "RNNModel", "rnn_units": 32, "rnn_layer": "LSTM"},
    )

    assert config["model_type"] == "KerasRNN"
    assert config["dt"] == 900
    assert config["model_path"].endswith(".keras")
    # the warmup replaces the lags, the model itself only sees the current time step
    assert config["warmup_steps"] == 24
    assert config["rnn_inputs"] == ["TDryBul", "HDirNor", "QTabs_set"]
    assert all(feature["lag"] == 1 for feature in config["input"].values())
    assert config["output"] == {
        "T_room": {
            "name": "T_room",
            "lag": 1,
            "output_type": "absolute",
            "recursive": True,
        }
    }

    # the result has to be loadable by agentlib_mpc
    serialized = SerializedMLModel.load_serialized_model(config)
    assert isinstance(serialized, SerializedKerasRNN)
    assert serialized.rnn_inputs == config["rnn_inputs"]


def test_physXAI_multi_step_options():
    """Difference outputs, autoregressive inputs and an explicit warmup."""
    config = physXAI_2_agentlib_json(
        "01",
        multi_step_source(
            inputs=["TDryBul", "QTabs_set", "T_room"], output=["Change(T_room)"]
        ),
        warmup_steps=96,
    )

    assert config["warmup_steps"] == 96
    # a feature that is also the output stays in the input order of the model, but is
    # only declared once, as an output
    assert config["rnn_inputs"] == ["TDryBul", "QTabs_set", "T_room"]
    assert list(config["input"]) == ["TDryBul", "QTabs_set"]
    assert config["output"]["T_room"]["output_type"] == "difference"

    serialized = SerializedMLModel.load_serialized_model(config)
    assert serialized.rnn_inputs == ["TDryBul", "QTabs_set", "T_room"]


def test_physXAI_multi_step_errors():
    """Lagged inputs cannot be expressed by a recurrent model."""
    with pytest.raises(ValueError, match="lagged inputs"):
        physXAI_2_agentlib_json(
            "01", multi_step_source(inputs=["TDryBul", "TDryBul_lag1", "QTabs_set"])
        )

    with pytest.raises(ValueError, match="Shift"):
        physXAI_2_agentlib_json("01", multi_step_source(shift=2))

    with pytest.raises(ValueError, match="non empty list"):
        physXAI_2_agentlib_json("01", multi_step_source(output=[]))
