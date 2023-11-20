"""
Example for running a multi-agent-system performing a data-driven distributed MPC with
ADMM. Creates five agents, one for the AHU control, one for the BKA control and three thermal zones

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
import logging
from pathlib import Path

from agentlib.utils.multi_agent_system import LocalMASAgency


agent_configs = [
    "configs//mpc//cca_controller.json",
    "configs//mpc//ahu_controller.json",
    "configs//mpc//Room_1.json",
    "configs//mpc//Room_2.json",
    "configs//mpc//Room_3.json",
    "configs//predictions.json",
    "configs//disturbances.json",
    "configs//coordinator.json",
    "configs//simulation//simulator_agent.json",
]


def run_example(
    until=3000, with_plots=True, start_pred=0, log_level=logging.INFO, cleanup=True
):
    # Set the log-level
    logging.basicConfig(level=log_level)

    # Change the working directly so that relative paths work
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    ann_paths = Path("anns/ann_t_air_0"), Path("anns/ann_t_cca_0")
    for path in ann_paths:
        if not path.exists():
            import training_direct

            training_direct.main()

    env_config = {"rt": False, "factor": 0.08, "t_sample": 60}
    mas = LocalMASAgency(
        agent_configs=agent_configs, env=env_config, variable_logging=False
    )
    mas.run(until=until)
    results = mas.get_results(cleanup=cleanup)

    if with_plots:
        import plot

        plot.main(results)
    return results


if __name__ == "__main__":
    import time

    """
    Currently simulates with DAE model
    To simulate this in a new directory, first use 'trainmodel.py' to generate
    serialized ANNs. The Serialized ANNs are not uploaded to not pollute git
    """
    start = time.perf_counter()
    run_example(
        with_plots=True,
        until=3600 * 24 * 3,
        start_pred=2 * 3600,
        cleanup=False,
        log_level=logging.INFO,
    )
    end = time.perf_counter()
    print(end - start)
