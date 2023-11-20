from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentlib_mpc.models.casadi_neural_network import NeuralNetwork

from agentlib_mpc.models.serialized_ann import SerializedANN
from agentlib_mpc.utils.analysis import load_sim


def plot(sim_results_file: str, ann_file: str = "anns//room_ann.json"):
    df = load_sim(Path("results//simulation_data_14days.csv"))
    df = df.loc[: 3600 * 24 * 2]

    s_ann = SerializedANN.load_serialized_ann(Path(ann_file))
    ann = NeuralNetwork(s_ann.deserialize())
    dt = s_ann.dt
    ann_output = [np.nan, df.iloc[0]["T"]]
    ann_deltas = []
    sim_deltas = []
    for time in df.index[1:-1]:
        inp = []
        all_inputs = s_ann.input | s_ann.output
        for var_name, feature in all_inputs.items():
            for i in range(feature.lag):
                inp.append(df.loc[(time - i * dt, var_name)])
        ann_delta = float(ann.predict(inp))
        ann_output.append(ann_delta + ann_output[-1])

        try:
            sim_deltas.append(df.loc[(time + dt, "T")] - df.loc[(time, "T")])
            ann_deltas.append(ann_delta)
        except KeyError:
            pass

    df["T_ann"] = ann_output
    # df = df.sort_values("T").dropna()

    fig, (ax_T_out, ax_mDot, ax_load, ax_T_in) = plt.subplots(4, 1)

    ax_T_out: plt.Axes
    df["T"].plot(ax=ax_T_out, label="Physical")
    # df["T_ann"].plot(
    #     ax=ax_T_out,
    #     marker="o",
    #     markeredgecolor="blue",
    #     linestyle="None",
    #     label="ANN",
    #     markerfacecolor="None",
    # )
    df["mDot"].plot(ax=ax_mDot, label="mDot")
    df["load"].plot(ax=ax_load, label="load")
    df["T_in"].plot(ax=ax_T_in, label="T_in")
    for x in (ax_T_out, ax_mDot, ax_load, ax_T_in):
        x.legend()
    plt.show()

    delta_df = pd.DataFrame({"simulation": sim_deltas, "black_box": ann_deltas})
    delta_df.sort_values(by="simulation", inplace=True)

    fig, ax = plt.subplots()
    ax.plot(delta_df["simulation"].values, color="red", label="sim")
    ax.plot(
        delta_df["black_box"].values,
        label="ann",
        marker="o",
        markeredgecolor="blue",
        linestyle="None",
        markerfacecolor="None",
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot("results//simulation_data_14days.csv")
