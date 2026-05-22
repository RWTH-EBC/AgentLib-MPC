"""
This is part three of the tutorial. 
In this part we learn about some advanced settings for the MPC agent, in particular:
    - Changing/setting the solver settings for the MPC agent.
    - Changing/setting the discretization options for the MPC agent.

All these settings can be specified in the respective configuration. 
For this part, refer to the configuration in mpc/part3_config.py.


"""


import logging
import matplotlib.pyplot as plt
from agentlib.utils.multi_agent_system import LocalMASAgency

from mpc.part3_config import get_config as get_part3_config



env_config = {"rt": False}



agent_configs = [
    "fmu//config.json",
    get_part3_config(),
    "predictor//config.json",
]


def run_example():
    logging.basicConfig(level=logging.INFO)

    mas = LocalMASAgency(
        agent_configs=agent_configs,
        env=env_config,
        variable_logging=True,
    )
    until = 86400
    mas.run(until=until)
    results = mas.get_results(cleanup=True)


    plot_results(results, until)




# Plotting function for the results of this example.
# Will produce a helpful plot, but can be ignored for this tutorial.

def plot_results(results, until):
    sim = results["SimAgent"]["SimRoom"].copy()
    if sim.index.nlevels > 1:
        sim = sim[sim.index.get_level_values(1) == 0].reset_index(level=1, drop=True)
    sim.index = sim.index - sim.index[0]

    mpc = results["myMPCAgent"]["myMPC"].copy()
    q_in = mpc["variable"]["Q_in"]
    if q_in.index.nlevels > 1:
        q_in = q_in[q_in.index.get_level_values(1) == 0].reset_index(level=1, drop=True)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(sim["T_zone"], label="T_zone")
    ax[0].set_ylabel("T_zone in K")
    ax[0].legend()

    ax[1].plot(q_in, label="Q_in")
    ax[1].set_ylabel("Q_in")
    ax[1].set_xlabel("Time")
    ax[1].legend()

    plt.xlim([0, until])
    plt.show()


if __name__ == "__main__":
    run_example()