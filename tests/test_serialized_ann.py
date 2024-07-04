"Module for tests of serialized_ann.py"
import os
import numpy as np
import unittest

import pytest
from keras.models import Sequential
from pathlib import Path
from typing import List

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.casadi_predictor import CasadiANN
from agentlib_mpc.models.serialized_ml_model import SerializedANN
from fixtures.ann import ANNTrainer
from fixtures.data_generator import DataGenerator


class TestSerializedANN(unittest.TestCase):
    """Class to test the SerializedANN class."""

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

        serialized_ann = SerializedANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        # tests
        self.assertIsInstance(serialized_ann.structure, str)
        self.assertIsInstance(serialized_ann.weights, list)
        self.assertIsInstance(serialized_ann.weights[0][0], list)
        self.assertIsInstance(serialized_ann.weights[2][1], list)

    def test_deserialize_function(self):
        """Test deserialize function."""
        # setup
        serialized_ann = SerializedANN.serialize(
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
        serialized_ann = SerializedANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        path = Path("test_ann.json")
        serialized_ann.save_serialized_model(path=path)
        loaded_ann = SerializedANN.load_serialized_model_from_file(path=path)
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
        serialized_ann = SerializedANN.serialize(
            self.test_ann, dt=1, input=self.inputs, output=self.outputs
        )
        casadi_ann = CasadiANN(serialized_model=serialized_ann)
        self.assertIsInstance(casadi_ann, CasadiANN)
        self.assertEqual(
            round(float(self.test_ann.predict(np.ones((1, 2)))[0]), 5),
            round(float(casadi_ann.predict(np.ones((1, 2)))[0]), 5),
        )
