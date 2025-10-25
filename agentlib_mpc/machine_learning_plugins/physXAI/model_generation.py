import importlib
import json
import os
import pathlib
import shutil
from typing import Union
from agentlib_mpc.machine_learning_plugins.physXAI.model_config_creation import config_physXAI_2_ddmpc


model_save_path_rel: str = 'models'


def use_existing_models(old_id: str, new_id: str, model_save_path: str):
    new_path = pathlib.Path(os.path.join(model_save_path, new_id))
    os.makedirs(new_path, exist_ok=True)

    old_path = pathlib.Path(os.path.join(model_save_path, old_id))
    if not old_path.is_dir():
        raise ValueError(f"Error: Models should be a list or dict of new models or should contain a path to an existing "
                         f"model folder. {str(old_path)} is not a valid directory.")

    try:
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
    except Exception as e:
        print(f"An error occurred: {e}")

    file_names = [str(p) for p in new_path.glob('*.json') if p.is_file()]
    return file_names


def generate_physxai_model(models: Union[list, dict, str], physXAI_config_base_path: str, scenario_name: str,
                           training_data_path: str, run_id: str, time_step: int = 900):

    if isinstance(models, str):
        return use_existing_models(models, run_id, model_save_path_rel)

    model_save_path =  os.path.abspath(model_save_path_rel)
    model_names = list()
    if isinstance(models, list):
        for model in models:
            if not model.endswith('.py'):
                model += '.py'
            spec = importlib.util.spec_from_file_location("train_model", os.path.join(physXAI_config_base_path,
                                                                                      'executables', scenario_name,
                                                                                      model))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            name = module.train_model(base_path=model_save_path, folder_name=run_id, training_data_path=training_data_path,
                         time_step=time_step)
            model_names.append(name)

    else:
        for model_name, model_path in models.items():
            if not model_path.endswith('.py'):
                model_path += '.py'
            spec = importlib.util.spec_from_file_location("train_model", os.path.join(physXAI_config_base_path,
                                                                                      'executables', scenario_name,
                                                                                      model_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.train_model(base_path=model_save_path, folder_name=run_id, training_data_path=training_data_path,
                         time_step=time_step, output_name=model_name)
            model_names.append(model_name)

    files = list()
    for name in model_names:
        pathes = {
            "preprocessing": os.path.join(model_save_path, run_id, f"{name}_preprocessing.json"),
            "constructed": os.path.join(model_save_path, run_id, f"{name}_constructed.json"),
            "model": os.path.join(model_save_path, run_id, f"{name}_model.json"),
            "training_data": os.path.join(model_save_path, run_id, f"{name}_training_data.json"),
            "training_data_pkl": os.path.join(model_save_path, run_id, f"{name}_training_data.pkl"),
        }
        with open(pathes["preprocessing"], "r") as f:
            preprocessing = json.load(f)
        with open(pathes["model"], "r") as f:
            model = json.load(f)
        with open(pathes["training_data"], "r") as f:
            training_data = json.load(f)
        for path in pathes.values():
            os.remove(path)

        model_config = config_physXAI_2_ddmpc(run_id, preprocessing, model, training_data)
        os.makedirs(os.path.join(model_save_path, run_id), exist_ok=True)
        file = os.path.join(model_save_path, run_id, f"{name}.json")
        with open(file, 'w') as f:
            json.dump(model_config, f)
        files.append(str(file))

    return files
