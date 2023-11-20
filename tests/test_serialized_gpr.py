import numpy as np
import os
import unittest

from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.casadi_predictor import CasadiGPR
from agentlib_mpc.models.serialized_ml_model import SerializedGPR, CustomGPR, MLModels
from fixtures.gpr import GPRTrainer
from fixtures.data_generator import DataGenerator


class TestSerializedGPR(unittest.TestCase):
    """Class to test the SerializedGPR class."""

    def setUp(self) -> None:
        """Setup the GPR for Testing the functions."""
        testdatagenerator = DataGenerator()
        testgprtrainer = GPRTrainer()
        self.test_data = testdatagenerator.test_data
        testgprtrainer.fit_test_gpr(self.test_data)
        self.test_gpr = testgprtrainer.test_gpr
        self.inputs = {"x": ml_model_datatypes.Feature(name="test_feature", lag=1)}
        self.outputs = {
            "y": ml_model_datatypes.OutputFeature(
                name="test_output_feature", lag=1, output_type="absolute"
            )
        }

    def test_deserialize_function(self):
        """Test deserialize function."""
        # setup
        serialized_gpr = SerializedGPR.serialize(
            self.test_gpr, dt=1, input=self.inputs, output=self.outputs
        )
        deserialized_gpr = serialized_gpr.deserialize()
        # tests
        self.assertIsInstance(deserialized_gpr, GaussianProcessRegressor)
        self.assertEqual(
            self.test_gpr.predict(np.ones((1, 2))),
            deserialized_gpr.predict(np.ones((1, 2))),
        )

    def test_save_and_load_serialized_gpr(self):
        """Test save_serialized_gpr function and load_serialized_gpr function."""
        # setup
        serialized_gpr = SerializedGPR.serialize(
            self.test_gpr, dt=1, input=self.inputs, output=self.outputs
        )
        path = Path("test_gpr.json")
        serialized_gpr.save_serialized_model(path=path)
        loaded_gpr = SerializedGPR.load_serialized_model_from_file(path=path)
        deserialized_gpr = loaded_gpr.deserialize()
        # tests
        self.assertIsInstance(deserialized_gpr, CustomGPR)
        self.assertEqual(
            self.test_gpr.predict(np.ones((1, 2))),
            deserialized_gpr.predict(np.ones((1, 2))),
        )
        path_abs = path.resolve()
        os.remove(path_abs)

    def test_casadi_gpr(self):
        """Tests if the casadi gpr can be instantiated"""
        serialized_gpr = SerializedGPR.serialize(
            self.test_gpr, dt=1, input=self.inputs, output=self.outputs
        )
        casadi_gpr = CasadiGPR(serialized_model=serialized_gpr)
        self.assertIsInstance(casadi_gpr, CasadiGPR)
        self.assertEqual(
            round(float(self.test_gpr.predict(np.ones((1, 2)))), 6),
            round(float(casadi_gpr.predict(np.ones((1, 2)))), 6),
        )
