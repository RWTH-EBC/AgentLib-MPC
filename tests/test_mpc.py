import unittest
import datetime

import numpy as np
import pandas as pd
import pathlib

from agentlib.core.environment import Environment
from agentlib.core.agent import Agent
from agentlib.core.errors import ConfigurationError

from agentlib_mpc.data_structures.interpolation import InterpolationMethods
from agentlib_mpc.optimization_backends.backend import OptimizationBackend
from agentlib_mpc.utils.sampling import sample

model_file = pathlib.Path(__file__).parent.joinpath("fixtures//casadi_test_model.py")

a = 1
# pylint: disable=missing-module-docstring,missing-class-docstring


class TestSampling(unittest.TestCase):
    @unittest.skip
    def test_sample_datetime(self):
        my_index = [datetime.datetime(2020, 2, 11, i) for i in range(10)]
        my_values = range(10)
        traj = pd.Series(my_values, index=my_index)

        # request trajectory within available data
        res1 = sample(
            traj,
            grid=np.linspace(0, 20 * 60, 5),
            current=datetime.datetime(2020, 2, 11, 3, 15),
        )
        res1_true = [
            3.25,
            3.5833333333333335,
            3.9166666666666665,
            4.25,
            4.583333333333333,
        ]
        self.assertTrue(np.allclose(np.array(res1), np.array(res1_true)))

        # request trajectory longer than available data
        res2 = sample(
            traj,
            grid=np.linspace(0, 50 * 60, 10),
            current=datetime.datetime(2020, 2, 11, 3, 15),
        )
        res2_true = [
            3.25,
            3.5833333333333335,
            3.9166666666666665,
            4.25,
            4.583333333333333,
        ]
        self.assertTrue(np.allclose(np.array(res2), np.array(res2_true)))

        # request trajectory beginning after end of available data
        res3 = sample(
            traj,
            grid=np.linspace(0, 50 * 60, 4),
            current=datetime.datetime(2020, 2, 12, 3, 15),
        )
        res3_true = [
            3.25,
            3.5833333333333335,
            3.9166666666666665,
            4.25,
            4.583333333333333,
        ]
        self.assertTrue(np.allclose(np.array(res3), np.array(res3_true)))

    def test_get_scalar(self):
        get1 = sample(
            trajectory=np.inf, current=200, grid=np.linspace(0, 50 * 60, 4)
        )
        self.assertEqual(get1, [np.inf] * 4)

    def test_series(self):
        sr = pd.Series([10, 12, 10, 12, 11], index=[0, 10, 20, 30, 40])

        # trajectory within
        res = sample(
            trajectory=sr, grid=[5, 12, 20, 28, 35], current=0
        )
        res_true = [11.0, 11.6, 10.0, 11.6, 11.5]
        self.assertTrue(np.allclose(np.array(res), np.array(res_true)))

        # some extrapolation
        res = sample(
            trajectory=sr, grid=[5, 12, 20, 28, 35], current=30
        )
        res_true = [11.5, 11, 11, 11, 11]
        self.assertTrue(np.allclose(np.array(res), np.array(res_true)))

    def test_list(self):
        traj = [10, 12, 10, 12, 11]

        # cannot process lists with length != grid length
        with self.assertRaises(ValueError):
            _ = sample(
                trajectory=traj, grid=[5, 12, 28, 35], current=0
            )

        # return lists of same length
        res = sample(
            trajectory=traj, grid=[5, 12, 20, 28, 35], current=0
        )
        self.assertEqual(traj, res)

    def test_previous_interpolation_method(self):
        values = [10, 20, 10, 20, 10]
        old_index = [0, 40, 50, 80, 200]
        traj = pd.Series(values, index=old_index)

        res = sample(
            trajectory=traj,
            grid=[0, 15, 30, 45, 60, 75, 90, 105, 120],
            current=0,
            method=InterpolationMethods.previous,
        )
        true_result = [10.0, 10.0, 10.0, 20.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        self.assertEqual(true_result, res)


env_config = {"rt": False, "strict": True}


class TestCasadiMPC(unittest.TestCase):
    """Tests of the mpc."""

    def setUp(self) -> None:
        self.agent_config = {
            "id": "myMPCAgent",
            "modules": [
                {
                    "module_id": "myMPC",
                    "type": "agentlib_mpc.mpc",
                    "optimization_backend": {
                        "type": "casadi",
                        "model": {
                            "type": {"file": model_file, "class_name": "MyCasadiModel"}
                        },
                        "discretization_options": {},
                    },
                    "time_step": 900,
                    "prediction_horizon": 5,
                    "inputs": [
                        {"name": "disturbance", "value": 270},
                    ],
                    "controls": [{"name": "myctrl", "value": 0.02, "ub": 1, "lb": 0}],
                    # Owned by MPC
                    "states": [{"name": "state", "value": 298.16}],  # Owned by MPC
                },
            ],
        }

    def test_CasadiMPC(self):
        """Creates an agent with a casadi based mpc. Runs one optimization and
        accesses output."""

        env = Environment(config=env_config)
        mpc_agent = Agent(config=self.agent_config, env=env)
        mpc = mpc_agent._modules["myMPC"]

        updated_vars = mpc.collect_variables_for_optimization()

        # solve optimization problem with up to date values from data_broker
        result = mpc.optimization_backend.solve(mpc.env.time, updated_vars)
        self.assertEqual(len(result["myctrl"]), 5)

    def test_incomplete_config(self):
        env = Environment(config=env_config)
        _agent_config = self.agent_config
        _agent_config["modules"][0]["states"].pop(0)
        with self.assertRaises(ConfigurationError):
            _ = Agent(config=_agent_config, env=env)


if __name__ == "__main__":
    unittest.main()
