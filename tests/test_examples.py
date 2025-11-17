
import logging
import os
import pandas as pd
import pathlib
import unittest

from agentlib.utils import custom_injection
from agentlib.utils.local_broadcast_broker import LocalBroadcastBroker


class TestExamples(unittest.TestCase):
    """Test all examples inside the agentlib"""

    def setUp(self) -> None:
        self.timeout = 15  # Seconds which the script is allowed to run
        self.main_cwd = os.getcwd()

    def tearDown(self) -> None:
        broker = LocalBroadcastBroker()
        broker.delete_all_clients()
        # Change back cwd:
        os.chdir(self.main_cwd)

    def _run_example_with_return(
        self, file: str, func_name: str, **kwargs
    ) -> dict[str, dict[str, pd.DataFrame]]:
        file = pathlib.Path(__file__).absolute().parents[1].joinpath("examples", file)

        # Custom file import
        test_func = custom_injection({"file": file, "class_name": func_name})
        results = test_func(**kwargs)
        self.assertIsInstance(results, dict)
        agent_name, agent = results.popitem()
        self.assertIsInstance(agent, dict)
        module_name, module_res = agent.popitem()
        self.assertIsInstance(module_res, pd.DataFrame)
        agent_results = results.setdefault(agent_name, {})
        agent_results[module_name] = module_res
        return results

    def test_ml_model_simulator_mpc(self):
        """Test the data-driven MPC examples"""

        # Linear regression examples
        self._run_example_with_return(
            file="one_room_mpc/ml_simulator/simple_mpc_ml_sim.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    def test_physxai_mpc(self):
        """Test physXAI plugin for one room mpc"""
        self._run_example_with_return(
            file="one_room_mpc//physXAI_plugin//simple_mpc_nn_physXAI.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
            until=3600,
            testing=True,
        )