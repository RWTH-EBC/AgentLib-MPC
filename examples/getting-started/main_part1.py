"""
This is part one of the getting-started.
It shows how to set up a simple multi-agent system with an FMU-based simulator and an MPC agent.

In this tutorial we learn about:
- Setting up a multi-agent system with three agents: a simulator agent, predictor agent, and an MPC agent.
- Running the multi-agent system and retrieving the results.

In this part:
1. Run this file to see the multi-agent system in action.
2. Check out the predictor/simple_predictor.py file to understand the structure of the system that is modelled in this tutorial.
3. Go to mpc/part1_simple_mpc_model.py for the first details of the MPC agent and find out why the simulation is not (yet) correct.
"""


import logging
import matplotlib.pyplot as plt
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.plotting.mpc import plot_mpc



env_config = {"rt": False}

# Agents are specified in the same way as it is known from the agentlib.
# For better structure and readability, we chose to define agents in separate json files.

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

    # The MPC results have a MultiIndex (time_step, inner_time) and columns
    # (value_type, variable). value_type can be 'variable', 'parameter',
    # 'lower' or 'upper'.
    mpc.index = mpc.index.set_levels(
        mpc.index.levels[0] - mpc.index.levels[0][0], level=0
    )

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1) Simulated temperature and its bounds
    ax[0].plot(sim["T_zone"], label="T_zone")
    ax[0].plot(mpc["parameter"]["T_upper"].groupby(level=0).first(), label="T_upper")
    ax[0].plot(mpc["parameter"]["T_lower"].groupby(level=0).first(), label="T_lower")
    ax[0].set_ylabel("T_zone in K")
    ax[0].set_title("Simulated temperature and bounds")
    ax[0].legend()

    # 2) Control input
    ax[1].plot(q_in, label="Q_in")
    ax[1].set_ylabel("Q_in")
    ax[1].set_title("Control input")
    ax[1].legend()

    # 3) MPC estimated/predicted temperature trajectory over the horizon.
    #    prediction_step reduces clutter by only plotting every nth trajectory.
    plot_mpc(
        series=mpc["variable"]["T_zone"],
        ax=ax[2],
        plot_actual_values=True,
        plot_predictions=True,
        prediction_step=8,
    )
    ax[2].set_ylabel("T_zone in K")
    ax[2].set_title("MPC estimated temperature trajectory")

    ax[2].set_xlabel("Time")
    plt.xlim([0, until])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_example()