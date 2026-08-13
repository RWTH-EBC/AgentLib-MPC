"""
This is part three of the getting-started. In this part we learn about some advanced settings for the MPC agent, in particular:
    - Changing/setting the solver settings for the MPC agent.
    - Changing/setting the discretization options for the MPC agent.

All these settings can be specified in the respective configuration.
For this part, refer to the configuration in mpc/part3_config.py. Its implemented as a function that
returns the configuration dictionary. This is done to allow comments in the configuration file,
which is not possible in a json file. 

In this part:
1. Run this file to see how the MPC behaves with the advanced solver and discretization settings.
2. Open mpc/part3_config.py to inspect the configuration options that control the MPC agent.
3. Compare the result with part 2 if you want to see the effect of changing the solver/discretization settings.


"""


import logging
import matplotlib.pyplot as plt
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.plotting.mpc import plot_mpc
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