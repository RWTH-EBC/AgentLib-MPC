import importlib
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import MagicMock


def test_model_creation(monkeypatch):
    monkeypatch.chdir(Path(__file__).parent)

    dir_to_remove = 'models/03'
    if Path(dir_to_remove).exists() and Path(dir_to_remove).is_dir():
        shutil.rmtree(dir_to_remove)

    test_args_01 = {
        "models": "02",
        "physXAI_scripts_path": "physXAI_scripts",
        "training_data_path": "training_data.csv",
        "run_id": "03",
    }
    test_return_value_01 = ["models\\03\\output_model.json", "models\\03\\output_preprocessing.json", "models\\03\\output_training_data.json"]
    ##############################################################################
    test_args_02 = {
        "models": ["output"],
        "physXAI_scripts_path": "physXAI_scripts",
        "training_data_path": "training_data.csv",
        "run_id": "03",
    }
    test_return_value_02 = ["models\\03\\output.json"]
    test_return_value_02 = [os.path.abspath(p) for p in test_return_value_02]
    ###############################################################################

    mock_models = MagicMock(name="MockedModelsClass")
    mock_physXAI_module = MagicMock(name="MockedPhysXAIModule")
    mock_physXAI_module.models = mock_models
    monkeypatch.setitem(sys.modules, "physXAI", mock_physXAI_module)

    from agentlib_mpc.machine_learning_plugins.physXAI.model_generation import generate_physxai_model

    mock_train_model_func = MagicMock(return_value="output")

    def mock_exec_module(module_obj):
        module_obj.train_model = mock_train_model_func

    mock_loader = MagicMock()
    mock_loader.exec_module.side_effect = mock_exec_module
    mock_spec = MagicMock()
    mock_spec.loader = mock_loader

    monkeypatch.setattr(
        importlib.util, 
        "spec_from_file_location", 
        lambda name, path: mock_spec
    )

    files_01 = generate_physxai_model(**test_args_01)
    assert files_01 == test_return_value_01

    files_02 = generate_physxai_model(**test_args_02)
    assert files_02 == test_return_value_02

    dir_to_remove = 'models/03'
    if Path(dir_to_remove).exists() and Path(dir_to_remove).is_dir():
        shutil.rmtree(dir_to_remove)

