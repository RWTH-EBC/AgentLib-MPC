import pandas as pd

import agentlib_mpc.data_structures.admm_datatypes as adt
from agentlib_mpc.utils.analysis import admm_at_time_step
from agentlib_mpc.utils.plotting.basic import EBCColors
from agentlib_mpc.utils.plotting.mpc import interpolate_colors


def plot_consensus_shades(
    results: dict[str, dict[str, pd.DataFrame]],
    data: dict[str, pd.DataFrame],
    time_step: float,
    # series: pd.Series,
    # ax: plt.Axes,
    # plot_actual_values: bool = False,
    # step: bool = False,
):
    """

    Args:
        series:

    Returns:

    """
    data = {}

    def mean(df: pd.DataFrame, name: str) -> pd.Series:
        return df["parameter"][adt.MEAN_PREFIX + name]

    def local(df: pd.DataFrame, name: str) -> pd.Series:
        return df["variable"][adt.LOCAL_PREFIX + name]

    def lmbda(df: pd.DataFrame, name: str) -> pd.Series:
        return df["parameter"][adt.MULTIPLIER_PREFIX + name]

    room_2 = results["CooledRoom_nn2"]["admm_module"]
    trajectories = {label: admm_at_time_step(srs) for label, srs in data.items()}

    # check the number of iterations on a random trajectory
    a_trajectory = next(iter(trajectories.values()))
    number_of_iterations: int = room_2.index.unique(level=0).shape[0]

    # series = room_2[]
    number_of_predictions: int = room_2.index.unique(level=0).shape[0]

    # stores the first value of each prediction. In the case of a control_variable,
    # this will give the optimal control output the mpc determined this step, or in
    # the case of a state, this will give the measurement it worked with
    actual_values: dict[float, float] = {}

    for i, (time, prediction) in enumerate(series.groupby(level=0)):
        prediction: pd.Series = prediction.dropna()
        actual_values[time] = prediction.iloc[0]

        progress = i / number_of_predictions
        prediction_color = interpolate_colors(
            progress=progress,
            colors=[EBCColors.red, EBCColors.dark_grey, EBCColors.light_grey],
        )
        prediction.index = prediction.index.droplevel(0) + time
        print(prediction)
        # if not step:
        #     prediction.plot(ax=ax, color=prediction_color)
        # else:
        #     prediction.plot(ax=ax, color=prediction_color, drawstyle="steps-post")

    # if plot_actual_values:
    #     actual_series = pd.Series(actual_values)
    #     if not step:
    #         actual_series.plot(ax=ax, color="black")
    #     else:
    #         actual_series.plot(ax=ax, color=EBCColors.dark_red, drawstyle="steps-post")

    # last_index = prediction.index[-1]
    # num_iters = last_index[1]
