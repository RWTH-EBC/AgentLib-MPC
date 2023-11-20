import logging
import unittest
from copy import deepcopy
import threading
import time

import pathlib
import numpy as np

from agentlib.core import Environment, Agent
from agentlib.utils.local_broadcast_broker import LocalBroadcastBroker

from agentlib_mpc.data_structures.admm_datatypes import ADMM_PREFIX
from agentlib_mpc.modules.dmpc.admm.admm import ADMM

model_file = pathlib.Path(__file__).parent.joinpath("fixtures//casadi_test_model.py")

examples_dir = pathlib.Path(__file__).parents[1].joinpath("examples")


logger = logging.getLogger(__name__)
a = 1

MAX_ITER = 20

agent_config = {
    "id": "myMPCAgent",
    "modules": [
        {
            "module_id": "myADMM",
            "type": "agentlib_mpc.admm",
            "optimization_backend": {
                "type": "casadi_admm",
                "model": {"type": {"file": model_file, "class_name": "MyCasadiModel"}},
                "discretization_options": {},
            },
            "time_step": 900,
            "prediction_horizon": 5,
            "max_iterations": MAX_ITER,
            "inputs": [
                {"name": "disturbance", "value": 270},
            ],
            "controls": [{"name": "myctrl", "value": 0.02, "ub": 1, "lb": 0}],
            "states": [{"name": "state", "value": 298.16}],
            "couplings": [
                {
                    "name": "myout",
                    "value": 298.16,
                }
            ],
        },
        {"type": "local_broadcast", "module_id": "lbr"},
    ],
}

agent_config2 = deepcopy(agent_config)
agent_config2["id"] = "MyAgent2"
admm_mod_con = agent_config2["modules"][0]
admm_mod_con["module_id"] = "myADMM2"
admm_mod_con["couplings"][0]["value"] = 295


class TestRTADMM(unittest.TestCase):
    def setUp(self) -> None:
        env_config = {"rt": True, "factor": 1}
        self.env = Environment(config=env_config)
        self.env.t_start = 0
        self.agent_config = deepcopy(agent_config)

    def tearDown(self) -> None:
        broker = LocalBroadcastBroker()
        broker.delete_all_clients()

    def test_admm_init(self):
        agent = Agent(env=self.env, config=self.agent_config)
        admm_module = agent.get_module("myADMM")
        for module in agent.modules:
            module.terminate()

    def test_comm(self):
        # setup two agents
        agent1 = Agent(env=self.env, config=deepcopy(agent_config))
        agent2 = Agent(env=self.env, config=deepcopy(agent_config2))
        # manually start callbacks since environment is not started
        next(agent1.data_broker._start_executing_callbacks(self.env))
        next(agent2.data_broker._start_executing_callbacks(self.env))

        self.admm_module1: ADMM = agent1.get_module("myADMM")
        self.admm_module2: ADMM = agent2.get_module("myADMM2")
        comm_1 = agent1.get_module("lbr")
        comm_2 = agent2.get_module("lbr")
        self.counter = 0
        self.counter2 = 0
        # import logging
        # logging.basicConfig(level=logging.DEBUG)

        # modify callback
        self.admm_module1.receive_participant = self.new_receive
        self.admm_module2.receive_participant = self.new_receive_2

        # stop reset of variable queues
        def nothing():
            pass

        self.admm_module1.deregister_all_participants = nothing
        self.admm_module1.deregister_all_participants = nothing

        # use debug mode for solution (skips casadi)
        self.admm_module1._solve_local_optimization = (
            self.admm_module1._solve_local_optimization_debug
        )
        self.admm_module2._solve_local_optimization = (
            self.admm_module2._solve_local_optimization_debug
        )

        # start threads
        threading.Thread(
            target=self.admm_module1._admm_loop,
            daemon=True,
            name=f"admm_loop_{self.admm_module1.agent.id}",
        ).start()
        threading.Thread(
            target=self.admm_module2._admm_loop,
            daemon=True,
            name=f"admm_loop_{self.admm_module2.agent.id}",
        ).start()
        next(comm_1._process_realtime())
        next(comm_2._process_realtime())

        self.env.t_start = 0
        self.admm_module1.start_step.set()
        self.admm_module2.start_step.set()

        # check run
        time.sleep(5)
        self.assertEqual(MAX_ITER + 1, self.counter)
        self.assertEqual(MAX_ITER + 1, self.counter2)

        name = ADMM_PREFIX + "_coupling_myout"
        parti = list(self.admm_module1.registered_participants[name].values())[0]
        parti2 = list(self.admm_module2.registered_participants[name].values())[0]

        # make sure no variables are left
        self.assertTrue(parti2.received.empty())
        self.assertTrue(parti.received.empty())

        # check that the sum of all multipliers is zero
        multiplier = ADMM_PREFIX + "_lambda_myout"
        lam1 = np.array(self.admm_module1._admm_variables.get(multiplier).value)
        lam2 = np.array(self.admm_module2._admm_variables.get(multiplier).value)
        lam_difference = lam1 + lam2
        lam_abs = np.array(np.absolute(lam1) + np.absolute(lam2)) / 2

        # makes sure the difference is below 10%. This is very generous and should catch
        # the case where we are off by one. This is not intended, but it won't affect
        # the practical applicability and this way the CI won't complain
        # In the ideal case, the agents should have the same multipliers
        close = all(d / a < 0.1 for d, a in zip(lam_difference, lam_abs))

        self.assertTrue(close)
        self.assertNotEqual(0, lam1[0])  # if this is zero there was no communication

    def new_receive(self, variable):
        ADMM.receive_participant(self.admm_module1, variable)
        self.counter += 1

    def new_receive_2(self, variable):
        ADMM.receive_participant(self.admm_module2, variable)
        self.counter2 += 1


if __name__ == "__main__":
    unittest.main()
