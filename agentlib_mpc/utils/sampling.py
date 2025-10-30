import itertools
import logging
from typing import Union, Iterable, Sequence, List
from numbers import Real

import numpy as np
import pandas as pd

from agentlib_mpc.data_structures.interpolation import InterpolationMethods

logger = logging.getLogger(__name__)


def sample_values_to_target_grid(
    values: Iterable[float],
    original_grid: Iterable[float],
    target_grid: Iterable[float],
    method: Union[str, InterpolationMethods],
) -> list[float]:
    if method == InterpolationMethods.linear:
        return np.interp(target_grid, original_grid, values).tolist()
    elif method == InterpolationMethods.spline3:
        raise NotImplementedError("Spline interpolation is currently not supported")
    elif method == InterpolationMethods.previous:
        return interpolate_to_previous(target_grid, original_grid, values)
    elif method == InterpolationMethods.mean_over_interval:
        values = np.array(values)
        original_grid = np.array(original_grid)
        result = []
        for i, j in pairwise(target_grid):
            slicer = np.logical_and(original_grid >= i, original_grid < j)
            result.append(values[slicer].mean())
        # take last value twice, so the length is consistent with the other resampling
        # methods
        result.append(result[-1])
        return result
    else:
        raise ValueError(
            f"Chosen 'method' {method} is not a valid method. "
            f"Currently supported: linear, spline, previous"
        )


def sample(
    trajectory: Union[Real, pd.Series, list[Real], dict[Real, Real]],
    grid: Union[list, np.ndarray],
    current: float = 0,
    method: str = "linear",
) -> list:
    """
    Obtain the specified portion of the trajectory.

    Args:
        trajectory:  The trajectory to be sampled. Scalars will be
            expanded onto the grid. Lists need to exactly match the provided
            grid. Otherwise, a pandas Series is accepted with the timestamp as index. A
             dict with the keys as time stamps is also accepted.
        current: start time of requested trajectory
        grid: target interpolation grid in seconds in relative terms (i.e.
            starting from 0 usually)
        method: interpolation method, currently accepted: 'linear',
            'spline', 'previous'

    Returns:
        Sampled list of values.

    Takes a slice of the trajectory from the current time step with the
    specified length and interpolates it to match the requested sampling.
    If the requested horizon is longer than the available data, the last
    available value will be used for the remainder.

    Raises:
        ValueError
        TypeError
    """
    target_grid_length = len(grid)
    if isinstance(trajectory, (float, int)):
        # return constant trajectory for scalars
        return [trajectory] * target_grid_length
    if isinstance(trajectory, list):
        # return lists of matching length without timestamps
        if len(trajectory) == target_grid_length:
            return trajectory
        raise ValueError(
            f"Passed list with length {len(trajectory)} "
            f"does not match target ({target_grid_length})."
        )
    if isinstance(trajectory, pd.Series):
        trajectory = trajectory.dropna()
        source_grid = np.array(trajectory.index)
        values = trajectory.values
    elif isinstance(trajectory, dict):
        source_grid = np.array(list(trajectory))
        values = np.array(list(trajectory.values()))
    else:
        raise TypeError(
            f"Passed trajectory of type '{type(trajectory)}' " f"cannot be sampled."
        )
    target_grid = np.array(grid) + current

    # expand scalar values
    if len(source_grid) == 1:
        if isinstance(trajectory, list):
            return [trajectory[0]] * target_grid_length
        # if not list, assume it is a series
        else:
            return [trajectory.iloc[0]] * target_grid_length

    # skip resampling if grids are (almost) the same
    if (target_grid.shape == source_grid.shape) and all(target_grid == source_grid):
        return list(values)
    values = np.array(values)

    # check requested portion of trajectory, whether the most recent value in the
    # source grid is older than the first value in the MHE trajectory
    if target_grid[0] >= source_grid[-1]:
        # return the last value of the trajectory if requested sample
        # starts out of range
        logger.warning(
            "Latest value of source grid %s is older than "
            "current time (%s. Returning latest value anyway.",
            source_grid[-1],
            current,
        )
        return [values[-1]] * target_grid_length

    # determine whether the target grid lies within the available source grid, and
    # how many entries to extrapolate on either side
    source_grid_oldest_time: float = source_grid[0]
    source_grid_newest_time: float = source_grid[-1]
    source_is_recent_enough: np.ndarray = target_grid < source_grid_newest_time
    source_is_old_enough: np.ndarray = target_grid > source_grid_oldest_time
    number_of_missing_old_entries: int = target_grid_length - np.count_nonzero(
        source_is_old_enough
    )
    number_of_missing_new_entries: int = target_grid_length - np.count_nonzero(
        source_is_recent_enough
    )
    # shorten target interpolation grid by extra points that go above or below
    # available data range
    target_grid = target_grid[source_is_recent_enough * source_is_old_enough]

    # interpolate data to match new grid
    sequence_new = sample_values_to_target_grid(
        values=values, original_grid=source_grid, target_grid=target_grid, method=method
    )

    # extrapolate sequence with last available value if necessary
    interpolated_trajectory = (
        [values[0]] * number_of_missing_old_entries
        + sequence_new
        + [values[-1]] * number_of_missing_new_entries
    )

    return interpolated_trajectory


def pairwise(iterable: Iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def earliest_index(time, arr, stop, start=0):
    """Helper function for interpolate_to_previous.
    Finds the current index to which we should forwardfill."""
    for i in range(start, stop):
        if arr[i] > time:
            return i - 1
    return 0


def interpolate_to_previous(
    target_grid: Iterable[float],
    original_grid: Iterable[float],
    values: Sequence[float],
) -> List[float]:
    """Interpolates to previous value of original grid, i.e. a forward fill.

    Stand-in for the following scipy code:
    tck = interpolate.interp1d(list(original_grid), values, kind="previous")
    result = list(tck(target_grid))
    """
    result = []
    _grid_index = 0
    stop = len(original_grid)
    for target_point in target_grid:
        _grid_index = earliest_index(
            target_point, original_grid, stop, start=_grid_index
        )
        result.append(values[_grid_index])
    return result
