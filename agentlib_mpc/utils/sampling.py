import itertools
import logging
from typing import Union, Iterable

import numpy as np
import pandas as pd
from scipy import interpolate

from agentlib_mpc.data_structures.interpolation import InterpolationMethods

logger = logging.getLogger(__name__)


def sample_values_to_target_grid(
    values: Iterable[float],
    original_grid: Iterable[float],
    target_grid: Iterable[float],
    method: Union[str, InterpolationMethods],
) -> list[float]:
    if method == InterpolationMethods.linear:
        tck = interpolate.interp1d(x=original_grid, y=values, kind="linear")
        return list(tck(target_grid))
    elif method == InterpolationMethods.spline3:
        raise NotImplementedError("Spline interpolation is currently not supported")
    elif method == InterpolationMethods.previous:
        tck = interpolate.interp1d(list(original_grid), values, kind="previous")
        return list(tck(target_grid))
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
    trajectory: Union[float, int, pd.Series, list[Union[float, int]]],
    grid: Union[list, np.ndarray],
    current: float = 0,
    method: str = "linear",
) -> list:
    """
    Obtain the specified portion of the trajectory.

    Args:
        trajectory:  The trajectory to be sampled. Scalars will be
            expanded onto the grid. Lists need to exactly match the provided
            grid. Otherwise, a list of tuples is accepted with the form (
            timestamp, value). A dict with the keys 'grid' and 'value' is also
            accepted.
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
        source_grid = np.array(trajectory.index)
        values = trajectory.values
    else:
        raise TypeError(
            f"Passed trajectory of type '{type(trajectory)}' cannot be sampled."
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
    if number_of_missing_new_entries > 0 or number_of_missing_old_entries > 0:
        logger.debug(
            "Available data for interpolation is not sufficient. Missing "
            f"{number_of_missing_new_entries} of recent data, and missing "
            f"{number_of_missing_old_entries} of old data.,"
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
