"""
Example for running a multi-agent-system performing a distributed MPC with
ADMM. Creates three agents, one for the AHU, one for a supplied room and one
for simulating the system.

All agents are started within the same Environment. To schedule execution of
the algorithm within one Environment, the admm_local module has to be used.
Since no time is required for scheduling, this can and should be run in a
regular Environment (not RealTime).

With this file belong the models
    - ca_cooler_model.py
    - ca_room_model.py
    - simulation_model.py

And the config files
    - cooled_room.json
    - cooler.json
    - simulator.json
    - local_broadcast.json

"""

import os
import json
import logging
from agentlib.utils.multi_agent_system import MultiProcessingMAS
from agentlib.utils import MultiProcessingBroker

agent_configs = [
    "configs//cooler_with_coordinator.json",
    "configs//cooled_room_with_coordinator.json",
    "configs//coordinator.json",
    "configs//simulator.json",
]


def plot(results, start_pred: float):
    import matplotlib.pyplot as plt
    from agentlib_mpc.utils.analysis import admm_at_time_step

    res_sim = results["Simulation"]["simulator"]
    start_time = res_sim.index[0]
    res_sim.index = res_sim.index - start_time
    mpc_room_results = results["CooledRoom"]["admm_module"]

    room_res = admm_at_time_step(
        data=mpc_room_results,
        time_step=start_pred,
        iteration=-1,
        index_offset=start_time,
    )

    fig, ax = plt.subplots(2, 1)
    ax[0].axhline(294.55, label="reference value")
    ax[0].plot(res_sim["T_0_out"], label="temperature")
    ax[0].plot(room_res["variable"]["T_0"], label="temperature prediction")
    ax[1].plot(res_sim["mDot_0"], label="air mass flow")
    ax[1].plot(room_res["variable"]["mDot_0"].dropna(), label="air flow prediction")
    ax[1].legend()
    ax[0].legend()
    plt.show()


def run_example(
    until=3000,
    with_plots=True,
    start_pred=0,
    log_level: int = logging.INFO,
    cleanup=True,
):
    # Set the log-level
    logging.basicConfig(level=log_level)

    # Change the working directly so that relative paths work
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    conf_dicts = []
    for conf in agent_configs:
        with open(conf) as f:
            conf_dict = json.load(f)
        modules = conf_dict["modules"]
        for i, mod in enumerate(modules):
            if isinstance(mod, str):
                with open(mod) as f:
                    mod = json.load(f)
            if mod["type"] in ["mqtt", "local_broadcast"]:
                conf_dict["modules"][
                    i
                ] = "configs//communicators//multiprocessing_broadcast.json"

        conf_dicts.append(conf_dict)

    env_config = {"rt": True, "factor": 0.08, "t_sample": 60}
    broker = MultiProcessingBroker(config={"port": 32300})
    mas = MultiProcessingMAS(
        agent_configs=conf_dicts,
        env=env_config,
        variable_logging=False,
        cleanup=cleanup,
        log_level=log_level,
    )
    mas.run(until=until)
    results = mas.get_results()

    if with_plots:
        plot(results, start_pred=start_pred)
    return results


if __name__ == "__main__":
    run_example(
        with_plots=True,
        until=1200,
        start_pred=600,
        cleanup=True,
        log_level=logging.INFO,
    )
