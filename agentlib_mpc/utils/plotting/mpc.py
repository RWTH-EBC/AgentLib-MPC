# from agentlib_mpc.utils.plotting.basic import ColorTuple
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from agentlib_mpc.utils import TIME_CONVERSION
from agentlib_mpc.utils.analysis import load_mpc
from agentlib_mpc.utils.plotting.basic import (
    ColorTuple,
    EBCColors,
    Float0to1,
    make_fig,
    Style,
)


def interpolate_colors(progress: Float0to1, colors: list[ColorTuple]) -> ColorTuple:
    """
    Interpolates colors based on a sample number. To be used when plotting many mpc
    predictions in one plot, so a fade from old to new predictions can be seen.

    Original credit to Max Berktold.

    Args:
        progress:
        colors:

    Returns:

    """
    if progress <= 0:
        return colors[0]
    elif progress >= 1:
        return colors[-1]

    num_colors = len(colors)
    interval = 1 / (num_colors - 1)
    color_index = int(progress / interval)
    t = (progress - interval * color_index) / interval
    color1 = colors[color_index]
    color2 = colors[color_index + 1]
    return (
        (1 - t) * color1[0] + t * color2[0],
        (1 - t) * color1[1] + t * color2[1],
        (1 - t) * color1[2] + t * color2[2],
    )


def plot_mpc(
    series: pd.Series,
    ax: plt.Axes,
    plot_actual_values: bool = True,
    plot_predictions: bool = False,
    step: bool = False,
    convert_to: Literal["seconds", "minutes", "hours", "days"] = "seconds",
):
    """

    Args:
        series: A column of the MPC results Dataframe
        ax: which Axes to plot on
        plot_actual_values: whether the closed loop actual values at the start of each
         optimization should be plotted (default True)
        plot_predictions: whether all predicted trajectories should be plotted
        step:
        convert_to: Will convert the index of the returned series to the specified unit
         (seconds, minutes, hours, days)

    Returns:

    """
    number_of_predictions: int = series.index.unique(level=0).shape[0]

    # stores the first value of each prediction. In the case of a control_variable,
    # this will give the optimal control output the mpc determined this step, or in
    # the case of a state, this will give the measurement it worked with
    actual_values: dict[float, float] = {}

    for i, (time_seconds, prediction) in enumerate(series.groupby(level=0)):
        prediction: pd.Series = prediction.dropna().droplevel(0)

        time_converted = time_seconds / TIME_CONVERSION[convert_to]
        if plot_actual_values:
            actual_values[time_converted] = prediction.at[0]

        prediction.index = (prediction.index + time_seconds) / TIME_CONVERSION[
            convert_to
        ]

        if plot_predictions:
            progress = i / number_of_predictions
            prediction_color = interpolate_colors(
                progress=progress,
                colors=[EBCColors.red, EBCColors.dark_grey, EBCColors.light_grey],
            )
            if not step:
                prediction.plot(
                    ax=ax, color=prediction_color, linewidth=0.7, label="_nolegend_"
                )
            else:
                prediction.plot(
                    ax=ax,
                    color=prediction_color,
                    drawstyle="steps-post",
                    linewidth=0.7,
                    label="_nolegend_",
                )

    if plot_actual_values:
        actual_series = pd.Series(actual_values)
        if not step:
            actual_series.plot(ax=ax, color="black", linewidth=1.5)
        else:
            actual_series.plot(
                ax=ax, color="black", linewidth=1.5, drawstyle="steps-post"
            )


def plot_admm(
    series: pd.Series,
    ax: plt.Axes,
    plot_actual_values: bool = True,
    plot_predictions: bool = False,
    step: bool = False,
    convert_to: Literal["seconds", "minutes", "hours", "days"] = "seconds",
):
    """

    Args:
        series: A column of the MPC results Dataframe
        ax: which Axes to plot on
        plot_actual_values: whether the closed loop actual values at the start of each
         optimization should be plotted (default True)
        plot_predictions: whether all predicted trajectories should be plotted
        step:
        convert_to: Will convert the index of the returned series to the specified unit
         (seconds, minutes, hours, days)

    Returns:

    """
    grid = series.index.get_level_values(2).unique()
    tail_length = len(grid[grid >= 0])
    series_final_predictions = series.groupby(level=0).tail(tail_length).droplevel(1)
    return plot_mpc(
        series=series_final_predictions,
        ax=ax,
        plot_actual_values=plot_actual_values,
        plot_predictions=plot_predictions,
        step=step,
        convert_to=convert_to,
    )
