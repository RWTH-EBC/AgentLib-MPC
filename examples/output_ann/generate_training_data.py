from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from agentlib import Agent, Environment

from agentlib_mpc.models.casadi_model import (
    CasadiInput,
    CasadiModel,
    CasadiModelConfig,
    CasadiOutput,
)
from agentlib_mpc.models.casadi_ml_model import CasadiMLModelConfig, CasadiMLModel
from agentlib_mpc.models.serialized_ann import SerializedANN
from agentlib_mpc.modules.ml_model_training.ann_trainer import ANNTrainer


def generate_test_data() -> pd.DataFrame:
    inputs = list(np.random.random((10000, 1)) * 100 - 50)
    func = lambda x: 2 * x
    outputs = [func(x) for x in inputs]
    func2 = lambda x: x + 10
    outputs2 = [func2(x) for x in inputs]
    df = pd.DataFrame({"inputs": inputs, "outputs": outputs, "outputs2": outputs2})
    return df

    # df.to_csv("test_data.csv")


def create_trainer():
    ag_config = {
        "modules": [],
        "id": "test",
    }
    trainer_config = {
        "step_size": 1,
        "module_id": "trainer",
        "type": "agentlib_mpc.ann_trainer",
        "agent_id": "test",
        "epochs": 50,
        "batch_size": 64,
        "inputs": [{"name": "inputs"}],
        "outputs": [{"name": "outputs"}, {"name": "outputs2"}],
        "output_types": {"outputs": "absolute", "outputs2": "absolute"},
        "recursive_outputs": {"outputs": False, "outputs2": False},
        "layers": [{32, "sigmoid"}, {32, "sigmoid"}, {32, "sigmoid"}],
        "train_share": 0.6,
        "validation_share": 0.2,
        "test_share": 0.2,
        "retrain_delay": 2,
        "save_directory": "anns",
        "use_values_for_incomplete_data": True,
        "save_data": True,
        "save_ann": True,
        "save_plots": True,
        "early_stopping": {"activate": "True", "patience": 500},
    }

    trainer = ANNTrainer(
        agent=Agent(env=Environment(), config=ag_config), config=trainer_config
    )
    test_data = generate_test_data()
    trainer.history_dict["outputs"] = list(test_data.index), list(test_data["outputs"])
    trainer.history_dict["outputs2"] = list(test_data.index), list(
        test_data["outputs2"]
    )
    trainer.history_dict["inputs"] = list(test_data.index), list(test_data["inputs"])
    trainer._update_time_series_data()

    sampled = trainer.resample()
    inputs, outputs = trainer.create_inputs_and_outputs(sampled)
    training_data = trainer.divide_in_tvt(inputs, outputs)
    trainer.fit_ann(training_data)
    serialized_ann = trainer.serialize_ann()
    trainer.save_all(serialized_ann, training_data)


def create_model():

    str_path = "anns//test_trainer_0//ann.json"
    ann = SerializedANN.load_serialized_ann(Path(str_path))
    inputs_ = [CasadiInput(name=name) for name in ann.input]
    outputs_ = [CasadiOutput(name=name) for name in ann.output]

    class TestModelConfig(CasadiMLModelConfig):
        inputs: List[CasadiInput] = inputs_
        outputs: List[CasadiOutput] = outputs_

    class TestModel(CasadiMLModel):
        config: TestModelConfig

        def setup_system(self):
            pass

    model = TestModel(ml_model_sources=[str_path])
    res = model.do_step(t_start=0)
    print(res)


if __name__ == "__main__":
    # create_trainer()
    create_model()
