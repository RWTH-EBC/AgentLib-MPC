from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentlib_mpc.data_structures.admm_datatypes import (
    MEAN_PREFIX,
    MULTIPLIER_PREFIX,
)
from agentlib_mpc.utils.analysis import load_sim, load_admm, admm_at_time_step
import agentlib_mpc.utils.plotting.basic as mpcplot

ROOM_COL1 = mpcplot.EBCColors.red
ROOM_COL2 = mpcplot.EBCColors.blue
ROOM_COL3 = mpcplot.EBCColors.green
WINDOW_AREA = 6.6  # m^2
ResultsT = dict[str, dict[str, pd.DataFrame]]

res_path = "results"


def load_results() -> ResultsT:
    results = {
        "Simulation": {
            "roomsimulator1": load_sim(Path(res_path, "room_1_sim.csv")),
            "roomsimulator2": load_sim(Path(res_path, "room_2_sim.csv")),
            "roomsimulator3": load_sim(Path(res_path, "room_3_sim.csv")),
        },
        "Temp_Controller": {
            "admm_module": load_admm(Path(res_path, "tempcontroller_res.csv"))
        },
        "aircooler": {"admm_module": load_admm(Path(res_path, "aircooler_res.csv"))},
        "CooledRoom_nn": {"admm_module": load_admm(Path(res_path, "room_1.csv"))},
        "CooledRoom_nn2": {"admm_module": load_admm(Path(res_path, "room_2.csv"))},
        "CooledRoom_nn3": {"admm_module": load_admm(Path(res_path, "room_3.csv"))},
    }
    return results


def to_celsius(srs: pd.Series) -> pd.Series:
    return srs - 273.15


def plot_disturbance(results: dict[str, dict[str, pd.DataFrame]]):
    res_sim1 = results["Simulation"]["roomsimulator1"].iloc[1:]
    res_sim2 = results["Simulation"]["roomsimulator2"].iloc[1:]
    res_sim3 = results["Simulation"]["roomsimulator3"].iloc[1:]

    # fig, (ax_r1, ax_r2, ax_r3) = make_fig(style=Style(use_tex=False), rows=3)
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=4)
    ax_r1, ax_r2, ax_r3, ax_t_amb = axs

    # room 1
    ax_r1.set_ylabel("$\dot{Q}_{Room,1}$ / W")
    res_sim1["d"].plot(ax=ax_r1, color="red", linestyle="-", label="Internal gains")
    q_rad = res_sim1["Q_rad"] * WINDOW_AREA
    q_rad.plot(ax=ax_r1, color="orange", linestyle="-", label="Solar Irradiation")

    # room 2
    ax_r2.set_ylabel("$\dot{Q}_{Room,2}$ / W")
    res_sim2["d"].plot(ax=ax_r2, color="red", linestyle="-")
    q_rad = res_sim2["Q_rad"] * WINDOW_AREA
    q_rad.plot(ax=ax_r2, color="orange", linestyle="-")

    # room 3
    ax_r3.set_ylabel("$\dot{Q}_{Room,3}$ / W")
    res_sim3["d"].plot(ax=ax_r3, color="red", linestyle="-", label="Internal gains")
    q_rad = res_sim3["Q_rad"] * WINDOW_AREA
    q_rad.plot(ax=ax_r3, color="orange", linestyle="-", label="Solar Irradiation")

    # t amb
    ax_t_amb.set_ylabel("$T_{amb}$ / °C")
    to_celsius(res_sim1["T_amb"]).plot(ax=ax_t_amb, color="0")

    x_ticks = np.arange(0, 3600 * 24 + 1, 3600 * 4)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax_t_amb.set_xticks(x_ticks)
    ax_t_amb.set_xticklabels(x_tick_labels)
    ax_t_amb.set_xlabel("Time / hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        # ax.set_yticks([15, 20, 25, 30])
        # ax.set_ylim(15, 30)
        ax.set_xlim(0, 3600 * 24)

    ax_r1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    fig.subplots_adjust(bottom=0.2)


def plot_room_temperatures(results: dict[str, dict[str, pd.DataFrame]]):
    res_sim1 = results["Simulation"]["roomsimulator1"].iloc[1:]
    res_sim2 = results["Simulation"]["roomsimulator2"].iloc[1:]
    res_sim3 = results["Simulation"]["roomsimulator3"].iloc[1:]

    # fig, (ax_r1, ax_r2, ax_r3) = make_fig(style=Style(use_tex=False), rows=3)
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=4)
    ax_r1, ax_r2, ax_r3, ax_cca = axs

    # room 1
    ax_r1.set_ylabel("$T_{room, 1}$ / °C")
    to_celsius(res_sim1["T_lower"]).plot(
        ax=ax_r1, color="0.5", linestyle="--", drawstyle="steps-post"
    )
    to_celsius(res_sim1["T_0_out"]).plot(
        ax=ax_r1,
        color=ROOM_COL1,
    )
    to_celsius(res_sim1["T_ahu"]).plot(ax=ax_r1, color=ROOM_COL1, linestyle=":")
    to_celsius(res_sim1["T_v"]).plot(ax=ax_r1, color="0", linestyle="--")

    # room 2
    ax_r2.set_ylabel("$T_{room, 2}$ / °C")
    to_celsius(res_sim2["T_upper"]).plot(
        ax=ax_r2, color="0.5", linestyle="--", drawstyle="steps-post"
    )
    to_celsius(res_sim2["T_lower"]).plot(
        ax=ax_r2, color="0.5", linestyle="--", drawstyle="steps-post"
    )
    to_celsius(res_sim2["T_0_out"]).plot(
        ax=ax_r2,
        color=ROOM_COL2,
    )
    to_celsius(res_sim2["T_ahu"]).plot(ax=ax_r2, color=ROOM_COL2, linestyle=":")
    to_celsius(res_sim1["T_v"]).plot(ax=ax_r2, color="0", linestyle="--")

    # room 3
    ax_r3.set_ylabel("$T_{room, 3}$ / °C")
    to_celsius(res_sim3["T_upper"]).plot(
        ax=ax_r3, color="0.5", linestyle="--", drawstyle="steps-post"
    )
    to_celsius(res_sim3["T_lower"]).plot(
        ax=ax_r3, color="0.5", linestyle="--", drawstyle="steps-post"
    )
    to_celsius(res_sim3["T_0_out"]).plot(
        ax=ax_r3,
        color=ROOM_COL3,
    )
    to_celsius(res_sim3["T_ahu"]).plot(ax=ax_r3, color=ROOM_COL3, linestyle=":")
    to_celsius(res_sim1["T_v"]).plot(ax=ax_r3, color="0", linestyle="--")

    # room 3
    ax_cca.set_ylabel("$T_{cca}$ / °C")
    to_celsius(res_sim1["T_v"]).plot(ax=ax_cca, color="0")

    x_ticks = np.arange(0, 3600 * 24 + 1, 3600 * 4)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax_cca.set_xticks(x_ticks)
    ax_cca.set_xticklabels(x_tick_labels)
    ax_cca.set_xlabel("Time / hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_yticks([15, 20, 25, 30])
        ax.set_ylim(12, 33)
        ax.set_xlim(0, 3600 * 24)
    # ax_cca.set_ylim(10, 25)


def calculate_Energy(
    air_controller,
    res_controller,
):
    powers = {
        "dQ_cca_1": [],
        "dQ_cca_2": [],
        "dQ_cca_3": [],
        "dQ_ahu_1": [],
        "dQ_ahu_2": [],
        "dQ_ahu_3": [],
    }
    time_steps = air_controller.index.unique(0)

    for t in time_steps:
        outer_index = air_controller.index.get_level_values(0)
        idx = np.searchsorted(outer_index, t, side="left")
        simtime = outer_index.values[idx]
        ahu_input = admm_at_time_step(data=air_controller, time_step=t, iteration=-1)
        bka_input = admm_at_time_step(data=res_controller, time_step=t, iteration=-1)

        powers["dQ_cca_1"].append(bka_input["variable"]["W1"][simtime])
        powers["dQ_cca_2"].append(bka_input["variable"]["W2"][simtime])
        powers["dQ_cca_3"].append(bka_input["variable"]["W3"][simtime])
        powers["dQ_ahu_1"].append(ahu_input["variable"]["W1"][simtime])
        powers["dQ_ahu_2"].append(ahu_input["variable"]["W2"][simtime])
        powers["dQ_ahu_3"].append(ahu_input["variable"]["W3"][simtime])

    for label, values in powers.items():
        powers[label] = pd.Series(values, index=time_steps)
    return tuple(p for p in powers.values())


def plot_energy_use(results: dict[str, dict[str, pd.DataFrame]]):
    res_controller = results["Temp_Controller"]["admm_module"]
    air_controller = results["aircooler"]["admm_module"]
    dQ_cca_1, dQ_cca_2, dQ_cca_3, dQ_ahu_1, dQ_ahu_2, dQ_ahu_3 = calculate_Energy(
        air_controller,
        res_controller,
    )

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    ax_cca, ax_ahu = axs

    # cca power
    ax_cca.set_ylabel("$\dot{Q}_{CCA}$ / W")
    dQ_cca_1.plot(
        ax=ax_cca,
        color=ROOM_COL1,
        linestyle="-",
        drawstyle="steps-post",
        label="Room 1",
    )
    dQ_cca_2.plot(
        ax=ax_cca,
        color=ROOM_COL2,
        linestyle="-",
        drawstyle="steps-post",
        label="Room 2",
    )
    dQ_cca_3.plot(
        ax=ax_cca,
        color=ROOM_COL3,
        linestyle="-",
        drawstyle="steps-post",
        label="Room 3",
    )

    # ahu power
    ax_ahu.set_ylabel("$\dot{Q}_{AHU}$ / W")
    dQ_ahu_1.plot(
        ax=ax_ahu,
        color=ROOM_COL1,
        linestyle="-",
        drawstyle="steps-post",
        label="Room 1",
    )
    dQ_ahu_2.plot(
        ax=ax_ahu,
        color=ROOM_COL2,
        linestyle="-",
        drawstyle="steps-post",
        label="Room 2",
    )
    dQ_ahu_3.plot(
        ax=ax_ahu,
        color=ROOM_COL3,
        linestyle="-",
        drawstyle="steps-post",
        label="Room 3",
    )

    x_ticks = np.arange(0, 3600 * 24 + 1, 3600 * 4)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax_ahu.set_xticks(x_ticks)
    ax_ahu.set_xticklabels(x_tick_labels)
    ax_ahu.set_xlabel("Time / hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        # ax.set_yticks([15, 20, 25, 30])
        # ax.set_ylim(15, 30)
        ax.set_xlim(0, 3600 * 24)

    ax_ahu.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    fig.subplots_adjust(left=0.15, bottom=0.2)


def plot_cca(results: ResultsT, start_pred: float = 4 * 3600):
    it = -1
    res_sim = results["Simulation"]["roomsimulator1"]
    res_sim2 = results["Simulation"]["roomsimulator2"]
    res_sim3 = results["Simulation"]["roomsimulator3"]
    res_controller = results["Temp_Controller"]["admm_module"]
    air_controller = results["aircooler"]["admm_module"]
    room_res_1 = results["CooledRoom_nn"]["admm_module"]
    room_res_2 = results["CooledRoom_nn2"]["admm_module"]
    room_res_3 = results["CooledRoom_nn3"]["admm_module"]
    room_res1 = admm_at_time_step(data=room_res_1, time_step=start_pred, iteration=it)
    room_res2 = admm_at_time_step(data=room_res_2, time_step=start_pred, iteration=it)
    room_res3 = admm_at_time_step(data=room_res_3, time_step=start_pred, iteration=it)
    cca_res = admm_at_time_step(data=res_controller, time_step=start_pred, iteration=it)

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    ax_cca, ax_lmbda = axs

    # cca temperature
    ax_cca.set_ylabel("$\dot{Q}_{CCA}$ / W")
    # res_sim["T_v"].plot(ax=ax_cca, color=ROOM_COL1, linestyle="-", label="actual")
    to_celsius(room_res1["parameter"][MEAN_PREFIX + "_" + "T_v"]).plot(
        ax=ax_cca, label="mean"
    )

    # multipliers
    ax_lmbda.set_ylabel("Multipliers / -")
    room_res1["parameter"][MULTIPLIER_PREFIX + "_" + "T_v"].plot(
        ax=ax_lmbda, label="Room 1"
    )
    room_res2["parameter"][MULTIPLIER_PREFIX + "_" + "T_v"].plot(
        ax=ax_lmbda, label="Room 2"
    )
    room_res3["parameter"][MULTIPLIER_PREFIX + "_" + "T_v"].plot(
        ax=ax_lmbda, label="Room 3"
    )
    cca_lam = (
        cca_res["parameter"][MULTIPLIER_PREFIX + "_" + "T_v_out"]
        + cca_res["parameter"][MULTIPLIER_PREFIX + "_" + "T_v_out2"]
        + cca_res["parameter"][MULTIPLIER_PREFIX + "_" + "T_v_out3"]
    )
    cca_lam.plot(ax=ax_lmbda, label="CCA Controller")
    # check = (
    #     room_res1["parameter"][MULTIPLIER_PREFIX + "_" + "T_v"]
    #     + room_res2["parameter"][MULTIPLIER_PREFIX + "_" + "T_v"]
    #     + room_res3["parameter"][MULTIPLIER_PREFIX + "_" + "T_v"]
    #     + cca_lam
    # )
    # check.plot(ax=ax_lmbda, label="check")

    x_ticks = np.arange(start_pred, start_pred + 3600 * 12 + 1, 3600 * 2)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax_lmbda.set_xticks(x_ticks)
    ax_lmbda.set_xticklabels(x_tick_labels)
    ax_lmbda.set_xlabel("Time / hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        # ax.set_yticks([15, 20, 25, 30])
        # ax.set_ylim(15, 30)
        ax.set_xlim(start_pred, start_pred + 3600 * 12 + 1)

    ax_lmbda.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        fancybox=True,
        shadow=True,
        ncol=4,
    )
    fig.subplots_adjust(bottom=0.2)


def main(results=None):
    if results is None:
        results = load_results()
    plot_energy_use(results)
    plot_disturbance(results)  # , start_pred=0, hours=24)
    plot_room_temperatures(results)
    # plot_consensus_shades(results)
    plot_cca(results, start_pred=4 * 3600)
    plt.show()


if __name__ == "__main__":
    main()
