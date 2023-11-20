"""This will be the example runner eventually."""

import unittest
import os
import subprocess
import logging
import pathlib
import pandas as pd
import pytest

from agentlib.utils import custom_injection
from agentlib.utils.local_broadcast_broker import LocalBroadcastBroker


class TestExamples(unittest.TestCase):
    """Test all examples inside the agentlib"""

    def setUp(self) -> None:
        self.timeout = 10  # Seconds which the script is allowed to run
        self.main_cwd = os.getcwd()

    def tearDown(self) -> None:
        broker = LocalBroadcastBroker()
        broker.delete_all_clients()
        # Change back cwd:
        os.chdir(self.main_cwd)

    def _run_example(self, example, timeout=None):
        if timeout is None:
            timeout = self.timeout
        ex_py = (
            pathlib.Path(__file__).absolute().parents[1].joinpath("examples", example)
        )
        try:
            subprocess.check_output(
                ["python", ex_py], stderr=subprocess.STDOUT, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            pass
        except subprocess.CalledProcessError as proc_err:
            raise Exception(proc_err.output.decode("utf-8")) from proc_err

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

    def test_mpc(self):
        """Test the mpc agent example"""
        self._run_example_with_return(
            file="one_room_mpc//physical//simple_mpc.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )
        self._run_example_with_return(
            file="one_room_mpc//physical//mpc_trajectories.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )
        self._run_example_with_return(
            file="one_room_mpc//physical//mixed_integer_mpc.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )
        """Test the mpc agent example"""
        self._run_example_with_return(
            file="one_room_mpc//physical//with_change_control_penalty.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    def test_admm_local(self):
        self._run_example_with_return(
            file="admm//admm_example_local.py",
            func_name="run_example",
            with_plots=False,
            until=1000,
            log_level=logging.FATAL,
            testing=True,
        )

    def test_admm_coordinated(self):
        self._run_example_with_return(
            file="admm//admm_example_coordinator.py",
            func_name="run_example",
            with_plots=False,
            until=1000,
            log_level=logging.FATAL,
        )

    @pytest.mark.slow
    def test_admm_realtime(self):
        self._run_example_with_return(
            file="admm//admm_example_main.py",
            func_name="run_example",
            with_plots=False,
            until=1000,
            log_level=logging.FATAL,
            testing=True,
        )

    def test_exchange_admm(self):
        self._run_example_with_return(
            file="exchange_admm//admm_4rooms_main.py",
            func_name="run_example",
            with_plots=False,
            until=1000,
            log_level=logging.FATAL,
        )
        self._run_example_with_return(
            file="exchange_admm//admm_4rooms_main_coord.py",
            func_name="run_example",
            with_plots=False,
            until=1000,
            log_level=logging.FATAL,
        )

    def test_admm_mp_broadcast(self):
        self._run_example_with_return(
            file="admm//admm_example_multiprocessing.py",
            func_name="run_example",
            with_plots=False,
            until=1000,
            log_level=logging.FATAL,
            TESTING=True,
        )

    def test_mpc_nn(self):
        """Test the mpc agent example"""
        self._run_example_with_return(
            file="one_room_mpc//ann//simple_mpc_nn.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    def test_mpc_gpr(self):
        """Test the mpc agent example"""
        self._run_example_with_return(
            file="one_room_mpc//gpr//simple_mpc_gpr.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    def test_mpc_linreg(self):
        """Test the mpc agent example"""
        self._run_example_with_return(
            file="one_room_mpc//linreg//simple_mpc_linreg.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    @pytest.mark.skip
    def test_mpc_nn_learned_output(self):
        """Test the mpc agent example"""
        self._run_example_with_return(
            file="one_room_mpc//ann//with_learned_cost//simple_mpc_nn.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    @pytest.mark.slow
    def test_admm_three_zone(self):
        results = self._run_example_with_return(
            file="three_zone_datadriven_admm//admm_3zone_sim.py",
            func_name="run_example",
            with_plots=False,
            until=3600,
            log_level=logging.FATAL,
        )
        residuals = results["Coordinator"]["admm_coordinator"]
        # if it takes too few iterations something went wrong
        assert residuals.shape[0] > 60
        assert residuals.loc[(0.001, 45)]["primal_residual"] < 0.1
        assert residuals.loc[(0.001, 45)]["dual_residual"] < 1
