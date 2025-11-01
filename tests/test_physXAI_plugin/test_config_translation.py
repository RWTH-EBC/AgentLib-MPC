from pathlib import Path
import pytest
from agentlib_mpc.machine_learning_plugins.physXAI.model_config_creation import physXAI_2_agentlib_json


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