from ast import literal_eval
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from agentlib_mpc.utils.plotting.basic import Style, make_fig, make_grid, EBCColors


def load_residuals(file: Union[str, Path]) -> pd.DataFrame:
    """Loads a residuals csv file in the correct format."""
    df = pd.read_csv(file, index_col=0)
    new_ind = [literal_eval(i) for i in df.index]
    df.index = pd.MultiIndex.from_tuples(new_ind)
    return df


def plot_single_time_step(
    residuals: pd.DataFrame,
    time_step: float = None,
    primal_tol: float = None,
    dual_tol: float = None,
) -> (plt.Figure, plt.Axes):
    """Plots the decrease of the residuals over iterations for a time step"""

    if time_step is None:
        residuals_time = residuals.index.get_level_values(0)[0]
        first_opt = residuals.loc[residuals_time]
    else:
        first_opt = residuals.loc[time_step]

    with Style() as style:
        fig, ax = make_fig(style)
        make_grid(ax)
        ax.set_ylabel("Residuals")

        first_opt["primal_residual"].plot(
            ax=ax, label="$r^k$", color=EBCColors.blue, linewidth=0.7
        )
        first_opt["dual_residual"].plot(
            ax=ax, label="$s^k$", color=EBCColors.red, linewidth=0.7
        )
        ax.set_yscale("log")

        if primal_tol:
            ax.axhline(
                primal_tol,
                label="$r_0$",
                color=EBCColors.blue,
                linewidth=0.7,
                linestyle="--",
            )
        if dual_tol:
            ax.axhline(
                dual_tol,
                label="$s_0$",
                color=EBCColors.red,
                linewidth=0.7,
                linestyle="--",
            )
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, handlelength=1
        )
    return fig, ax


def plot_over_time(
    residuals: pd.DataFrame,
    primal_tol: float = None,
    dual_tol: float = None,
    max_iters: int = None,
) -> (plt.Figure, (plt.Axes, plt.Axes)):
    """Plots the final residuals over time."""
    res_over_time = residuals_over_time(residuals)

    with Style() as style:
        fig, (ax_res, ax_iter) = plt.subplots(2, 1)
        ax_res: plt.Axes
        ax_iter: plt.Axes
        ax_res.tick_params(
            axis="both",
            which="major",
            labelsize=style.font_dict["fontsize"],
            left=False,
        )
        ax_iter.tick_params(
            axis="both",
            which="major",
            labelsize=style.font_dict["fontsize"],
            left=False,
        )
        make_grid(ax_res)
        make_grid(ax_iter)

        res_over_time["primal_residual"].plot(
            ax=ax_res, label="$r_t$", color=EBCColors.blue, linewidth=0.7
        )
        res_over_time["dual_residual"].plot(
            ax=ax_res, label="$s_t$", color=EBCColors.red, linewidth=0.7
        )
        if primal_tol:
            ax_res.axhline(
                primal_tol,
                label="$r_0$",
                color=EBCColors.blue,
                linewidth=0.7,
                linestyle="--",
            )
        if dual_tol:
            ax_res.axhline(
                dual_tol,
                label="$s_0$",
                color=EBCColors.red,
                linewidth=0.7,
                linestyle="--",
            )
        ax_res.set_ylabel("Residuals")
        ax_res.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, handlelength=1
        )
        # ax_res.set_yscale("log")

        ax_iter.set_ylabel("Iterations")
        res_over_time["iters"].plot(
            ax=ax_iter, label="iterations", color="black", linewidth=0.7
        )
        if max_iters:
            ax_iter.axhline(
                max_iters,
                label="Iteration limit",
                color="black",
                linewidth=0.7,
                linestyle="--",
            )
        ax_iter.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, handlelength=1
        )
    return fig, (ax_res, ax_iter)


def residuals_over_time(residuals: pd.DataFrame) -> pd.DataFrame:
    """Evaluates the residuals over time. Takes a raw residuals DataFrame and returns a
    Dataframe, which has for each time step the number of iterations and the final primal and dual residuals.

    Returns:
        DataFrame with float index (time in seconds) and the columns
        ("primal_residual", "dual_residual", "iters")
    """
    time_vals = set(residuals.index.get_level_values(0))
    iters = {t: residuals.loc[t].shape[0] for t in time_vals}
    prim_res = {t: residuals.loc[t].iloc[-1]["primal_residual"] for t in time_vals}
    dual_res = {t: residuals.loc[t].iloc[-1]["dual_residual"] for t in time_vals}

    df = pd.DataFrame(
        {"primal_residual": prim_res, "dual_residual": dual_res, "iters": iters}
    ).sort_index()
    return df
