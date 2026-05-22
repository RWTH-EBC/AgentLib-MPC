"""
This is part one of the tutorial. 
It shows how to set up a simple multi-agent system with an FMU-based simulator and an MPC agent.

In this tutorial we learn about:
- Setting up a multi-agent system with three agents: a simulator agent, predictor agent, and an MPC agent.
- Running the multi-agent system and retrieving the results.

Run this file to see the multi-agent system in action.
To understand the structure of the system that is modelled in this tutorial, we recommend to check out the simple_predictor.py file before proceeding.
Then, you can refer to the part1_simple_mpc_model.py for the first details of the MPC agent.
"""


import logging
import matplotlib.pyplot as plt
from agentlib.utils.multi_agent_system import LocalMASAgency



env_config = {"rt": False}

# Agents are specified in the same way as it is known from the agentlib.
# For better structure and readability, we chose to define agents in seperate json files.

agent_configs = [
    "fmu//config.json",
    "mpc//part1_config.json",
    "predictor//config.json",
]


# The defined agents are run in the same way as usual in the agentlib. 
# After the run, the results are retrieved and plotted.

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