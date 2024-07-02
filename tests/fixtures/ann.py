import keras
import numpy as np
import pytest
from keras import layers

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.serialized_ml_model import SerializedANN


class ANNTrainer:
    """
    Trains Ann with Keras.
    """

    def __init__(self):
        self.test_ann = self.build_test_ann()

    def build_test_ann(self):
        """
        Builds ANN and defines the architecture of the ANN and returns it.
        """
        test_model = keras.Sequential()
        test_model.add(layers.Dense(12, input_dim=2, activation="relu"))
        test_model.add(layers.Dense(8, activation="relu"))
        test_model.add(layers.Dense(1, activation="sigmoid"))

        test_model.compile(loss="mean_squared_error", optimizer="adam")

        return test_model

    def fit_test_ann(self, data: dict):
        """
        Optimizes weights and biases of ANN's and uses train data for this.

        Args:
            data: training data as a dict
        """
        self.test_ann.fit(x=data.get("x"), y=data.get("y"), epochs=3, batch_size=5)


@pytest.fixture
def example_ann(training_data):
    testanntrainer = ANNTrainer()
    testanntrainer.fit_test_ann(training_data)
    return testanntrainer.test_ann


@pytest.fixture
def example_serialized_ann(example_ann):
    inputs = {"x": ml_model_datatypes.Feature()}
    outputs = {
        "y": ml_model_datatypes.OutputFeature(
            output_type=ml_model_datatypes.OutputType.absolute
        )
    }
    return SerializedANN.serialize(example_ann, dt=1, input=inputs, output=outputs)
