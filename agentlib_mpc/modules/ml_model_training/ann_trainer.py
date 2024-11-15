from typing import Literal

import keras.api.callbacks
import pydantic
from agentlib import Agent
from agentlib.core.errors import OptionalDependencyError
from keras import Sequential

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.serialized_ml_model.serialized_ann import SerializedANN
from agentlib_mpc.modules.ml_model_training.ml_model_trainer import (
    MLModelTrainerConfig,
    MLModelTrainer,
)


class ANNTrainerConfig(MLModelTrainerConfig):
    """
    Pydantic data model for ANNTrainer configuration parser
    """

    epochs: int = 100
    batch_size: int = 100
    layers: list[tuple[int, ml_model_datatypes.Activation]] = pydantic.Field(
        default=[(16, "sigmoid")],
        description="Hidden layers which should be created for the ANN. An ANN always "
        "has a BatchNormalization Layer, and an Output Layer the size of "
        "the output dimensions. Additional hidden layers can be specified "
        "here as a list of tuples: "
        "(#neurons of layer, activation function).",
    )
    early_stopping: EarlyStoppingCallback = pydantic.Field(
        default=EarlyStoppingCallback(),
        description="Specification of the EarlyStopping Callback for training",
    )


class ANNTrainer(MLModelTrainer):
    """
    Module that generates ANNs based on received data.
    """

    config: ANNTrainerConfig
    model_type = SerializedANN

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config, agent)

    def build_ml_model(self) -> "Sequential":
        """Build an ANN with a one layer structure, can only create one ANN"""
        try:
            from keras import layers, Sequential
        except ImportError as err:
            raise OptionalDependencyError(
                dependency_install="keras",
                used_object="Neural Networks",
            ) from err

        ann = Sequential()
        ann.add(layers.BatchNormalization(axis=1))
        for units, activation in self.config.layers:
            ann.add(layers.Dense(units=units, activation=activation))
        ann.add(layers.Dense(units=len(self.config.outputs), activation="linear"))
        ann.compile(loss="mse", optimizer="adam")
        return ann

    def fit_ml_model(self, training_data: ml_model_datatypes.TrainingData):
        callbacks = []
        if self.config.early_stopping.activate:
            callbacks.append(self.config.early_stopping.callback())

        self.ml_model.fit(
            x=training_data.training_inputs,
            y=training_data.training_outputs,
            validation_data=(
                training_data.validation_inputs,
                training_data.validation_outputs,
            ),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
        )


class EarlyStoppingCallback(pydantic.BaseModel):
    patience: int = (1000,)
    verbose: Literal[0, 1] = 0
    restore_best_weights: bool = True
    activate: bool = False

    def callback(self):
        import keras.callbacks

        return keras.callbacks.EarlyStopping(
            patience=self.patience,
            verbose=self.verbose,
            restore_best_weights=self.restore_best_weights,
        )
