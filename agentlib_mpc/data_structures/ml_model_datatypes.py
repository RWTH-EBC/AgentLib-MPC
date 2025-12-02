import dataclasses
from pathlib import Path
from typing import Literal

import keras.callbacks
import pandas as pd
import pydantic
from enum import Enum
from agentlib.core.errors import ConfigurationError
from pydantic import BaseModel
from pydantic_core.core_schema import FieldValidationInfo

ML_MODEL_TO_MPC = "__mlmodel_to_mpc"

class OutputType(str, Enum):
    absolute = "absolute"
    difference = "difference"


class Feature(BaseModel):
    name: str
    lag: int = 1


class OutputFeature(Feature):
    output_type: OutputType = pydantic.Field(
        description="What kind of output this is. If 'absolute' an forward pass of the"
        " MLModel will yield the absolute value of the featuer at the next time"
        " step. If it is 'difference', the difference to the last time step"
        " will be generated, so it has to be added again."
    )
    recursive: bool = pydantic.Field(
        default=True,
        description="If the output feature is recursive, it will also be used as an "
        "input for the model. This is useful for mpc, where we want to "
        "model the evolution of states based on their previous value. If "
        "false, can be used to model algebraic relationships. Default is "
        "true.",
    )

    @pydantic.field_validator("recursive")
    @classmethod
    def non_recursive_features_have_to_be_absolute(
        cls, recursive, info: FieldValidationInfo
    ):
        output_type = info.data["output_type"]
        if not recursive and output_type == "difference":
            raise ConfigurationError(
                f"Output Feature {info.data['name']} was specified as a non-recursive feature"
                f" for which the differenc in output should be learned. This "
                f"combination is not allowed. Please set 'output_type' to "
                f"'absolute' for non-recursive features."
            )
        return recursive


@dataclasses.dataclass
class TrainingData:
    """Stores the data which is used to train a model."""

    training_inputs: pd.DataFrame
    training_outputs: pd.DataFrame
    validation_inputs: pd.DataFrame
    validation_outputs: pd.DataFrame
    test_inputs: pd.DataFrame
    test_outputs: pd.DataFrame

    def save(self, path: Path):
        """Saves three csv files in the path location. The csv files contain the test,
        training and validation data"""
        training = pd.concat(
            [self.training_inputs, self.training_outputs],
            keys=["inputs", "outputs"],
            axis=1,
        )
        validation = pd.concat(
            [self.validation_inputs, self.validation_outputs],
            keys=["inputs", "outputs"],
            axis=1,
        )
        test = pd.concat(
            [self.test_inputs, self.test_outputs],
            keys=["inputs", "outputs"],
            axis=1,
        )
        full = pd.concat(
            [training, validation, test], keys=["training", "validation", "test"]
        )
        full.sort_index(inplace=True)
        path.mkdir(parents=True, exist_ok=True)
        full.to_csv(Path(path, "train_test_val_data.csv"))

    @classmethod
    def load(cls, path: Path):
        full = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
        return cls(
            training_inputs=full.loc["training"]["inputs"],
            test_inputs=full.loc["test"]["inputs"],
            validation_inputs=full.loc["validation"]["inputs"],
            training_outputs=full.loc["training"]["outputs"],
            test_outputs=full.loc["test"]["outputs"],
            validation_outputs=full.loc["validation"]["outputs"],
        )


Activation = Literal[
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh",
    "selu",
    "elu",
    "exponential",
]


def column_order(
    inputs: dict[str, Feature], outputs: dict[str, OutputFeature]
) -> list[str]:
    """Defines the order of the columns in which Training data should be passed to
    keras, and saved for the Serialization."""
    ordered: list[str] = []
    for name, feat in inputs.items():
        for i in range(feat.lag):
            ordered.append(name_with_lag(name, i))
    for name, feat in outputs.items():
        if not feat.recursive:
            continue
        for i in range(feat.lag):
            ordered.append(name_with_lag(name, i))
    return ordered


def name_with_lag(name: str, lag: int) -> str:
    if lag == 0:
        return name
    return f"{name}_{lag}"


class EarlyStoppingCallback(pydantic.BaseModel):
    patience: int = (1000,)
    verbose: Literal[0, 1] = 0
    restore_best_weights: bool = True
    activate: bool = False

    def callback(self):
        return keras.callbacks.EarlyStopping(
            patience=self.patience,
            verbose=self.verbose,
            restore_best_weights=self.restore_best_weights,
        )
