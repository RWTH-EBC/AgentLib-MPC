
import pytest

import numpy as np


class DataGenerator:
    """Generates test data for ANN training."""

    def __init__(self):
        self.test_data = self.generate_test_data()

    def generate_test_data(self):
        """Generates and returns test data as np.ndarray of shape [10, 2]"""
        np.random.seed(1)
        x = np.random.uniform(-5, 5, size=(10, 2))
        y = self.evaluate_rosenbrock_function(x)
        return {"x": x, "y": y}

    def evaluate_rosenbrock_function(self, inp: np.ndarray):
        """
        Takes np.ndarray of shape [*, 2] and evaluates the Rosenbrock function
        for every row. The results are returned as np.ndarray of shape [*, ].

        Args:
            inp:    input data (np.ndarray of shape [*, 2]) for x and y of
                    Rosenbrock function.
        Returns:
            res:    results (np.ndarray of shape [*, ]) of Rosenbrock function.
        """
        a = 1
        b = 100
        res = np.zeros((inp.shape[0], 1))
        for i in range(0, inp.shape[0]):
            x = inp[i, 0]
            y = inp[i, 1]
            res[i] = (a - x) ** 2 + b * (y - x**2) ** 2
        return res



@pytest.fixture
def training_data():
    testdatagenerator = DataGenerator()
    return testdatagenerator.test_data