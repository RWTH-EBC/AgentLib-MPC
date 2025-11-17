from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from agentlib_mpc.utils import analysis
from agentlib_mpc.utils.plotting import admm_animation
from agentlib_mpc.utils.plotting import basic


def customize_fig(fig: plt.Figure, ax: plt.Axes) -> (plt.Figure, plt.Axes):
    # grids
    basic.make_grid(ax)

    # auxiliary
    ax.set_ylim(0, 0.11)
    ax.set_xlim(0, 3000)
    ax.legend()
    ax.set_ylabel("mass flow / $frac{kg}{s}$")

    ax.set_xlabel("Time / s")
    return fig, ax


def main():
    try:
        rooms = [analysis.load_admm(Path(f"admm_opt_{i}.csv")) for i in range(1, 5)]
        rlt = analysis.load_admm(Path("admm_opt_rlt.csv"))["variable"]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Results files do not exist. Make sure you run 'admm_4rooms_coord_main.py'"
            " with 'cleanup=False' to generate result files."
        ) from e
    sum_of_mass_flows_rooms = sum(room["variable"]["mDot"] for room in rooms).dropna()
    sum_of_mass_flows_rlt = rlt[[f"mDot_{i}" for i in range(1, 5)]].dropna().sum(axis=1)
    data_ = {
        admm_animation.Label("rooms mass flow"): sum_of_mass_flows_rooms,
        admm_animation.Label("rlt mass flow"): sum_of_mass_flows_rlt,
    }

    iter_dict = analysis.get_number_of_iterations(sum_of_mass_flows_rooms)
    iters = pd.Series(iter_dict).iloc[0]

    admm_animation.make_image(
        file_name="four_rooms_mDot_0.png",
        data=data_,
        customize=customize_fig,
        time_step=0,
        iteration=0,
    )
    admm_animation.make_animation(
        file_name="four_rooms_mDot.gif",
        data=data_,
        customize=customize_fig,
        time_step=0,
        iteration=iters,
    )


if __name__ == "__main__":
    main()
