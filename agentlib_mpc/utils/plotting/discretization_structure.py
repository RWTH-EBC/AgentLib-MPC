from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from agentlib_mpc.utils.analysis import load_admm, admm_at_time_step
from agentlib_mpc.utils.plotting import basic
from agentlib_mpc.utils.plotting.basic import Customizer, Style, EBCColors


def spy_structure(df: pd.DataFrame, customizer: Customizer = None, file: Path = ""):
    with basic.Style() as style:
        style.font_dict["fontsize"] = 12
        fig, ax = basic.make_fig(style, customizer=customizer)

        columns = df.columns
        index = df.index
        ax.spy(
            df.notnull(),
            markersize=10,
            markerfacecolor=EBCColors.red,
            markeredgecolor=EBCColors.red,
        )
        ax.set_xticklabels(columns, fontsize=style.font_dict["fontsize"])
        ax.set_xticks([i for i, _ in enumerate(columns)])
        ax.set_yticklabels(
            [f"{int(time)} s" for i, time in enumerate(index)],
            fontsize=style.font_dict["fontsize"],
        )
        ax.set_yticks([i for i, _ in enumerate(index)])

    if file:
        fig.savefig(file)
    else:
        fig.show()


if __name__ == "__main__":

    def customize(fig: plt.Figure, ax: plt.Axes, style: Style):
        cm = 1 / 2.54
        fig.set_size_inches(8 * cm, 9 * cm)
        return fig, ax

    base_path = Path(
        r"C:\Users\ses\Dokumente\Konferenzen\MED2022\Figures\Simulations"
        r"\iteration analysis\tracking_constantrho1"
    )

    cons1_path = Path(base_path, "admm_consumer1.csv")
    cons2_path = Path(base_path, "admm_consumer2.csv")
    prod_path = Path(base_path, "admm_prod_opt.csv")

    cons1 = load_admm(cons1_path)
    room_0_0 = admm_at_time_step(cons1, time_step=0, iteration=0)
    room_0_0_vars = room_0_0["variable"].iloc[:8]
    # room_0_0_vars = room_0_0_vars[['room_T', 'wall_T', 'room_T_slack', 'Heat_flow_in', 'heating_T']]
    room_0_0_vars.drop("room_T_slack", axis=1, inplace=True)
    column_labels = ["$T_z$", "$T_w$", r"$\dot{Q}_{in}$", "$T_h$"]
    room_0_0_vars.columns = column_labels
    spy_structure(
        room_0_0_vars, customizer=customize, file=Path("cons_structure_short.png")
    )

    def customize(fig: plt.Figure, ax: plt.Axes, style: Style):
        cm = 1 / 2.54
        fig.set_size_inches(8 * cm, 9 * cm)
        ax.yaxis.tick_right()
        return fig, ax

    prod = load_admm(prod_path)
    prod_0_0 = admm_at_time_step(prod, time_step=0, iteration=0)
    prod_0_0_vars = prod_0_0["variable"].iloc[:8]
    prod_0_0_vars = prod_0_0_vars[["heating_1_T", "Heat_flow_out_1", "heating_1_speed"]]
    column_labels = ["$T_h$", r"$\dot{Q}_{in}$", "$u_1$"]
    prod_0_0_vars.columns = column_labels
    spy_structure(
        prod_0_0_vars, customizer=customize, file=Path("prod_structure_short.png")
    )
