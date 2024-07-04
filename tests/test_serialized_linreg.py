import numpy as np
import os
import unittest

from pathlib import Path
from sklearn.linear_model import LinearRegression

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.casadi_predictor import CasadiLinReg
from agentlib_mpc.models.serialized_ml_model import SerializedLinReg
from fixtures.linreg import LinRegTrainer
from fixtures.data_generator import DataGenerator


class TestSerializedLinReg(unittest.TestCase):
    """Class to test the SerializedLinReg class."""

    def setUp(self) -> None:
        """Setup the LinReg for Testing the functions."""
        testdatagenerator = DataGenerator()
        testlinregtrainer = LinRegTrainer()
        self.test_data = testdatagenerator.test_data
        testlinregtrainer.fit_test_linreg(self.test_data)
        self.test_linreg = testlinregtrainer.test_linreg
        self.inputs = {"x": ml_model_datatypes.Feature(name="test_feature", lag=1)}
        self.outputs = {
            "y": ml_model_datatypes.OutputFeature(
                name="test_output_feature", lag=1, output_type="absolute"
            )
        }

    def test_deserialize_function(self):
        """Test deserialize function."""
        # setup
        serialized_linreg = SerializedLinReg.serialize(
            self.test_linreg, dt=1, input=self.inputs, output=self.outputs
        )
        deserialized_linreg = serialized_linreg.deserialize()
        # tests
        self.assertIsInstance(deserialized_linreg, LinearRegression)
        self.assertEqual(
            self.test_linreg.predict(np.ones((1, 2))),
            deserialized_linreg.predict(np.ones((1, 2))),
        )

    def test_save_and_load_serialized_linreg(self):
        """Test save_serialized_linreg function and load_serialized_linreg function."""
        # setup
        serialized_linreg = SerializedLinReg.serialize(
            self.test_linreg, dt=1, input=self.inputs, output=self.outputs
        )
        path = Path("test_linreg.json")
        serialized_linreg.save_serialized_model(path=path)
        loaded_linreg = SerializedLinReg.load_serialized_model_from_file(path=path)
        deserialized_linreg = loaded_linreg.deserialize()
        # tests
        self.assertIsInstance(deserialized_linreg, LinearRegression)
        self.assertEqual(
            self.test_linreg.predict(np.ones((1, 2))),
            deserialized_linreg.predict(np.ones((1, 2))),
        )
        path_abs = path.resolve()
        os.remove(path_abs)

    def test_casadi_linreg(self):
        """Tests if the casadi linreg can be instantiated"""
        serialized_linreg = SerializedLinReg.serialize(
            self.test_linreg, dt=1, input=self.inputs, output=self.outputs
        )
        casadi_linreg = CasadiLinReg(serialized_model=serialized_linreg)
        self.assertIsInstance(casadi_linreg, CasadiLinReg)
        self.assertEqual(
            round(float(self.test_linreg.predict(np.ones((1, 2)))[0]), 5),
            round(float(casadi_linreg.predict(np.ones((1, 2)))[0]), 5),
        )
