"Module for tests of SerializedKerasANN"

import os
import numpy as np
import unittest

from keras import Sequential
from pathlib import Path

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.casadi_predictor import CasadiANN
from agentlib_mpc.models.serialized_ml_model import SerializedKerasANN
from fixtures.ann import ANNTrainer
from fixtures.data_generator import DataGenerator


class TestSerializedKerasANN(unittest.TestCase):
    """Class to test the SerializedKerasANN class."""

    def setUp(self) -> None:
        """Setup the ANN for Testing the functions."""
        testdatagenerator = DataGenerator()
        testanntrainer = ANNTrainer()
        self.test_data = testdatagenerator.test_data
        testanntrainer.fit_test_ann(self.test_data)
        self.test_ann = testanntrainer.test_ann
        self.inputs = {"x": ml_model_datatypes.Feature(name="test_feature", lag=1)}
        self.outputs = {
            "y": ml_model_datatypes.OutputFeature(
                name="test_output_feature", lag=1, output_type="absolute"
            )
        }

    def test_serialize(self):
        """Test serialize function."""
        # setup
        serialized_ann = SerializedKerasANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        # tests
        custom_path = Path("stored_models/model.keras")
        self.assertIsInstance(serialized_ann.model_path, Path)
        assert serialized_ann.model_path == custom_path

        if custom_path.exists():
            os.remove(custom_path)
            custom_path.parent.rmdir()

        # setup
        test_dir = Path(__file__).parent
        custom_path = Path(test_dir, "custom_path", "model.keras")
        self.test_ann.save_path = str(custom_path)
        serialized_ann = SerializedKerasANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        # tests
        assert serialized_ann.model_path == custom_path

        if custom_path.exists():
            os.remove(custom_path)
            custom_path.parent.rmdir()
        

    def test_deserialize_function(self):
        """Test deserialize function."""
        # setup
        serialized_ann = SerializedKerasANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        deserialized_ann = serialized_ann.deserialize()
        # tests
        self.assertIsInstance(deserialized_ann, Sequential)
        self.assertEqual(
            self.test_ann.predict(np.ones((1, 2))),
            deserialized_ann.predict(np.ones((1, 2))),
        )

    def test_save_and_load_serialized_ann(self):
        """Test save_serialized_ann function and load_serialized_ann function."""
        # setup
        serialized_ann = SerializedKerasANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        path = Path("test_ann.json")
        serialized_ann.save_serialized_model(path=path)
        loaded_ann = SerializedKerasANN.load_serialized_model_from_file(path=path)
        deserialized_ann = loaded_ann.deserialize()
        # tests
        self.assertIsInstance(deserialized_ann, Sequential)
        self.assertEqual(
            self.test_ann.predict(np.ones((1, 2))),
            deserialized_ann.predict(np.ones((1, 2))),
        )
        path_abs = path.resolve()
        os.remove(path_abs)

    def test_casadi_ann(self):
        """Tests if the casadi ann can be instantiated"""
        serialized_ann = SerializedKerasANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        casadi_ann = CasadiANN(serialized_model=serialized_ann)
        self.assertIsInstance(casadi_ann, CasadiANN)
        self.assertEqual(
            round(self.test_ann.predict(np.ones((1, 2)))[0].item(), 5),
            round(float(casadi_ann.predict(np.ones((1, 2)))[0]), 5),
        )
