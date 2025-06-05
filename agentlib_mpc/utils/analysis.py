import warnings
from ast import literal_eval
import datetime
from pathlib import Path
from typing import NewType, Literal, Union, Optional, Iterable

import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np

from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


SimulationTime = NewType("SimulationTime", float)


def load_admm(file: Union[Path, str]) -> pd.DataFrame:
    return load_mpc(file)


def load_mpc(file: Union[Path, str]) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=[0], header=[0, 1])
    new_ind = [literal_eval(i) for i in df.index]
    df.index = pd.MultiIndex.from_tuples(new_ind)
    return df


def load_mpc_stats_hdf5(file: Union[Path, str]) -> Optional[pd.DataFrame]:
    """Load MPC statistics from HDF5 file."""
    if not HAS_H5PY:
        from agentlib.core.errors import OptionalDependencyError

        raise OptionalDependencyError(
            used_object="HDF5 stats loading",
            dependency_install="agentlib[hdf5]",
            dependency_name="hdf5",
        )

    try:
        with h5py.File(file, "r") as f:
            stats_group = f["stats"]

            # Reconstruct stats dataframe
            stats_data = []
            for key in sorted(stats_group.keys()):
                line = stats_group[key][()].decode("utf-8")
                stats_data.append(line)

            # Parse stats lines back into dataframe
            # This depends on the format of stats_line
            # You might need to adjust based on actual format
            return pd.DataFrame([eval(line) for line in stats_data])
    except Exception:
        return None


def load_mpc_hdf5(file: Union[Path, str]) -> pd.DataFrame:
    """Load MPC results from HDF5 file."""
    if not HAS_H5PY:
        from agentlib.core.errors import OptionalDependencyError

        raise OptionalDependencyError(
            used_object="HDF5 result loading",
            dependency_install="agentlib[hdf5]",
            dependency_name="hdf5",
        )

    with h5py.File(file, "r") as f:
        # Reconstruct column structure
        meta = f["metadata"]
        n_levels = meta.attrs["n_column_levels"]

        if n_levels > 1:
            # Reconstruct MultiIndex columns - decode from bytes
            col_tuples = [eval(col.decode("utf-8")) for col in meta["columns"][:]]
            columns = pd.MultiIndex.from_tuples(col_tuples)
        else:
            # Decode column names from bytes
            columns = [col.decode("utf-8") for col in meta["columns"][:]]

        # Load all results
        dfs = []
        results_group = f["results"]

        for time_key in sorted(
            results_group.keys(), key=lambda x: float(x.split("_")[1])
        ):
            time = float(time_key.split("_")[1])
            time_group = results_group[time_key]

            for iter_key in sorted(time_group.keys()):
                if "_index" in iter_key:
                    continue

                iteration = int(iter_key.split("_")[1])
                data = time_group[iter_key][:]
                index = time_group[iter_key + "_index"][:]

                df = pd.DataFrame(data, index=index, columns=columns)

                # Create appropriate MultiIndex
                if iteration == 0:  # MPC
                    df.index = pd.MultiIndex.from_tuples([(time, idx) for idx in index])
                else:  # ADMM
                    df.index = pd.MultiIndex.from_tuples(
                        [(time, iteration, idx) for idx in index]
                    )

                dfs.append(df)

        return pd.concat(dfs)


def load_mpc_stats(results_file: Union[str, Path]) -> Optional[pd.DataFrame]:
    stats_file = mpc_datamodels.stats_path(results_file)
    try:
        df = pd.read_csv(stats_file, index_col=0)
    except Exception:
        return None
    if is_float_dtype(df.index):
        return df
    new_ind = [literal_eval(i) for i in df.index]
    df.index = pd.MultiIndex.from_tuples(new_ind)
    return df


def load_sim(file: Path, causality=None) -> pd.DataFrame:
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
    if causality:
        df = df[causality]
        return df.droplevel(level=1, axis=1)
    return df.droplevel(level=2, axis=1).droplevel(level=0, axis=1)


def convert_multi_index(
    data: pd.DataFrame, convert_to: Union[TimeConversionTypes, Literal["datetime"]]
):
    """Converts an index of an MPC or ADMM results Dataframe to a different unit,
    assuming it is passed in seconds."""
    # last = data.index.nlevels - 1  # should be 1 for mpc, 2 for admm
    outer = convert_index(convert_to, data.index.unique(0))
    return data.set_index(
        data.index.set_levels(outer, level=0)
    )  # .set_levels(inner, level=last)


def convert_index(
    convert_to: Union[TimeConversionTypes, Literal["datetime"]], index: pd.Index
):
    """
    Converts an index from seconds to datetime or another unit
    Args:
        convert_to: unit, e.g. minutes, hours, datetime
        index: pandas index object

    Returns:

    """
    if convert_to == "datetime":
        return pd.to_datetime(index.astype(int), unit="s")
    else:
        return index / TIME_CONVERSION[convert_to]


def perform_index_update(
    data: pd.DataFrame, offset: Union[float, Literal["auto"], bool], admm: bool = False
) -> pd.DataFrame:
    """Updates the index of a raw mpc/admm result dataframe, to be offset by a desired
    time value."""
    if not offset:
        return data
    outer_index = data.index.get_level_values(0)
    if offset == "auto" or offset is True:
        _index_offset = outer_index[0]
    else:
        _index_offset = offset
    outer_index = outer_index - _index_offset

    if admm:
        arrays = [
            outer_index,
            data.index.get_level_values(1),
            data.index.get_level_values(2),
        ]
    else:  # mpc
        arrays = [outer_index, data.index.get_level_values(1)]

    # set index like this, because set_index() only works for dataframes, not series
    data_copy = data.copy()
    data_copy.index = pd.MultiIndex.from_arrays(arrays)
    return data_copy


def mpc_at_time_step(
    data: pd.DataFrame,
    time_step: float,
    variable=None,
    variable_type="variable",
    index_offset: Union[float, Literal["auto"], bool] = True,
) -> pd.DataFrame:
    """
    Gets the results of an optimization at a time step.

    Args:
        data: The multi-indexed results data frame from the mpc
        time_step: The time step from which results should be shown.
            If no exact match, shows closest.
        variable: If specified, only returns results
            with regard to a certain variable.
        variable_type: The type of the variable provided (parameter, variable, lower, ...)
        index_offset: Determines how the index will be updated when loading the data.
        The offset will be subtracted from the time-index. This is useful for results
        of realtime systems, where the time value with be a unix time stamp and we want
         to cut the number down to something understandable. For example, if the time
         index (level 0 of the input Dataframe) is [105, 115, 125] and we give an
         index_offset of 100, the data will be handled as if the index was [5, 15, 25].
          If "auto" or True is provided as an argument, the index will be modified to
           start at 0. If 0 or False are provided, no modifications will be made.

    Returns:
        pd.DataFrame: A single-indexed Dataframe of the optimization results
            at the specified time step. If variable is not specified,
            returns all variables with a double column index, if it
            is specified returns only values and/or bounds with
            single indexed columns.
    """

    # get the closest matching (outer) index matching the requested time step
    data = perform_index_update(data, index_offset, admm=False)
    outer_index = data.index.get_level_values(0)
    idx = np.searchsorted(outer_index, time_step, side="left")
    if idx > 0 and (
        idx == len(outer_index)
        or np.fabs(time_step - outer_index[idx - 1])
        < np.fabs(time_step - outer_index[idx])
    ):
        closest = outer_index[idx - 1]
    else:
        closest = outer_index[idx]

    # select the data at this index and increment the inner index
    if variable:
        data_at_ts = data[variable_type][variable].loc[closest]
    else:
        data_at_ts = data.loc[closest]
    data_at_ts = data_at_ts.copy()
    data_at_ts.index = data_at_ts.index + closest

    return data_at_ts


def admm_at_time_step(
    data: Union[pd.DataFrame, pd.Series],
    time_step: float = None,
    variable=None,
    iteration: float = -1,
    index_offset: Union[float, Literal["auto"], bool] = True,
    convert_to: TimeConversionTypes = "seconds",
) -> pd.DataFrame:
    """
    Gets the results of an optimization at a time step.
    Args:
        index_offset: Determines how the index will be updated when loading the data.
        The offset will be subtracted from the time-index. This is useful for results
        of realtime systems, where the time value with be a unix time stamp and we want
         to cut the number down to something understandable. For example, if the time
         index (level 0 of the input Dataframe) is [105, 115, 125] and we give an
         index_offset of 100, the data will be handled as if the index was [5, 15, 25].
          If "auto" or True is provided as an argument, the index will be modified to
           start at 0. If 0 or False are provided, no modifications will be made.
        data: The multi-indexed results data frame from the mpc
        time_step: The time step from which results should be shown.
                   If no exact match, shows closest.
        variable: If specified, only returns results
                  with regard to a certain variable.
        iteration: Specifies, from which inner ADMM iteration data should be
            from. If negative, counts from last iteration. Default -1.
        convert_to: Whether the data should be converted to datetime, minutes etc.


    Returns:
        A single-indexed Dataframe of the optimization results
        at the specified time step. If variable is not specified,
        returns all variables with a double column index, if it
        is specified returns only values and/or bounds with
        single indexed columns.
    """

    # get the closest matching (outer) index matching the requested time step
    data = convert_multi_index(data, convert_to=convert_to)
    if not convert_to == "datetime":
        data = perform_index_update(data, index_offset, admm=True)
    outer_index = data.index.get_level_values(0)

    if time_step is None:
        time_step = 0 if not convert_to == "datetime" else datetime.datetime.now()

    idx = np.searchsorted(outer_index, time_step, side="left")
    if idx > 0 and (
        idx == len(outer_index)
        or np.fabs(time_step - outer_index[idx - 1])
        < np.fabs(time_step - outer_index[idx])
    ):
        closest = outer_index[idx - 1]
    else:
        closest = outer_index[idx]

    data_at_ts = data.loc[closest]

    # if iteration provided is negative we count backwards (like list indexing)
    if iteration < 0:
        number_of_admm_iterations = data_at_ts.index.get_level_values(0).max()
        iteration = number_of_admm_iterations + 1 + iteration

    # select the data at this index and increment the inner index
    if variable:
        data_at_it = data_at_ts.xs(variable, axis=1, level="variable").loc[iteration]
    else:
        data_at_it = data_at_ts.loc[iteration]
    data_at_it = data_at_it.copy()

    if convert_to == "datetime":
        index = convert_index(convert_to, data_at_it.index + closest.value // 1e9)
    else:
        index = convert_index(convert_to, data_at_it.index) + closest
    data_at_it.index = index
    return data_at_it


def get_number_of_iterations(data: pd.DataFrame) -> dict[SimulationTime, int]:
    """Returns the number of iterations at each time instance of the ADMM simulation."""

    ind_full = data.index
    ind = ind_full.droplevel(2).drop_duplicates()
    time_stamps = ind.droplevel(1).drop_duplicates()
    result = {}
    for t in time_stamps:
        _slice = ind.get_loc(t)
        result[SimulationTime(t)] = len(ind[_slice])

    return result


def get_time_steps(data: pd.DataFrame) -> Iterable[float]:
    """Returns the time steps at which an MPC step was performed."""
    return sorted(set(data.index.get_level_values(0)))


def first_vals_at_trajectory_index(data: Union[pd.DataFrame, pd.Series]):
    """Gets the first values at each time step of a results trajectory."""
    time_steps = get_time_steps(data)
    first_vals = pd.Series(
        {time_step: data.loc[time_step].iloc[0] for time_step in time_steps}
    )
    if np.nan in first_vals:
        warnings.warn(
            "Nan detected in first values. You may need to select the "
            "correct column of the DataFrame and drop NaN before."
        )
    return first_vals


def last_vals_at_trajectory_index(data: Union[pd.DataFrame, pd.Series]):
    """Gets the last values at each time step of a results trajectory."""
    time_steps = get_time_steps(data)
    # -1 covers for parameters (only one entry) and states (-horizon until 0)
    last_vals = pd.Series(
        {time_step: data.at[time_step].iloc[-1] for time_step in time_steps}
    )

    if np.nan in last_vals:
        warnings.warn(
            "Nan detected in first values. You may need to select the "
            "correct column of the DataFrame and drop NaN before."
        )
    return last_vals
