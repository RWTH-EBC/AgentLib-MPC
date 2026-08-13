"""
This is part two of the getting-started. In this part we learn about different types of objectives.
Since objective functions are defined in the MPC model, we refer to the mpc model module (examples/getting-started/mpc/part2_simple_mpc_model.py) for this part of the tutorial.
In this part:
1. Run this file to see how the different objective terms affect the MPC behavior. Note how this time dashboard is shown,
   giving deeper insights into the MPC behavior and solver performance. This is a basic agentlib-mpc feature. 
   You can find the example code down here at the end of the plotting function marked by "DASHBOARD".
2. Open mpc/part2_simple_mpc_model.py to inspect how the objective is defined in the MPC model.
3. Compare the result with part 1 if you want to see how the objective setup changed.

"""


import logging
import matplotlib.pyplot as plt
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.plotting.mpc import plot_mpc


env_config = {"rt": False}



agent_configs = [
    "fmu//config.json",
    "mpc//part2_config.json",
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

    # DASHBOARD
    # The interactive dashboard is a further support to test out the behavior
    # of the different objective terms. It shows the MPC results and, if
    # available, the solver statistics.
    from agentlib_mpc.utils.analysis import load_mpc_stats
    from agentlib_mpc.utils.plotting.interactive import show_dashboard

    try:
        stats = load_mpc_stats(results)
    except Exception:
        stats = None

    mpc_results = results["myMPCAgent"]["myMPC"]
    show_dashboard(mpc_results, stats=stats)


if __name__ == "__main__":
    run_example()