import pandas as pd
import numpy as np

def determine_expression_type(expr_name, system):
    """Determine the type of an expression (CasadiInput, CasadiState, CasadiParameter, etc.)"""
    try:
        # Check in model inputs
        for inp in system.model.inputs:
            if hasattr(inp, 'name') and inp.name == expr_name:
                return "CasadiInput"

        # Check in model states
        for state in system.model.states:
            if hasattr(state, 'name') and state.name == expr_name:
                return "CasadiState"

        # Check in model parameters
        for param in system.model.parameters:
            if hasattr(param, 'name') and param.name == expr_name:
                return "CasadiParameter"

        # Check in model outputs
        for output in system.model.outputs:
            if hasattr(output, 'name') and output.name == expr_name:
                return "CasadiOutput"
    except (AttributeError, TypeError):
        pass

    return "Unknown"


def determine_common_grid(grids, types):
    """
    Determine a common grid based on the expression types.
    Prioritize CasadiInput grid, then CasadiState grid.
    """
    input_grid = None
    state_grid = None

    # Find Input and State grids
    for i, t in enumerate(types):
        if t == "CasadiInput" and input_grid is None:
            input_grid = grids[i]
        elif t == "CasadiState" and state_grid is None:
            state_grid = grids[i]

    # Prioritize input grid, then state grid, or use the first grid if neither is found
    if input_grid is not None:
        return input_grid
    elif state_grid is not None:
        return state_grid
    elif grids:
        return grids[0]
    else:
        return np.array([])


def align_to_grid(values, original_grid, target_grid, expr_type):
    """
    Align values to a target grid based on expression type.

    For parameters: Forward/backward fill
    For states when target grid is from inputs: Replace NaNs with mean of following values
    For others: Linear interpolation
    """
    # Ensure we have numpy arrays
    values = np.asarray(values)
    original_grid = np.asarray(original_grid)
    target_grid = np.asarray(target_grid)

    # If grids are identical, no need to align
    if np.array_equal(original_grid, target_grid):
        return values

    # Convert to pandas Series for easier handling
    original_series = pd.Series(values, index=original_grid)

    # Reindex to target grid (creates NaNs where values don't exist)
    aligned_series = original_series.reindex(target_grid)

    if expr_type == "CasadiParameter":
        # Forward fill, then backward fill for parameters
        aligned_series = aligned_series.ffill().bfill()
    elif expr_type == "CasadiState" and not np.array_equal(target_grid, original_grid):
        # For states with different grid, use the mean replacement method
        aligned_series = replace_with_mean(aligned_series)
    else:
        # Linear interpolation for other types
        aligned_series = aligned_series.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

    return aligned_series.values


def replace_with_mean(series):
    """
    Replace NaN values with the mean of following values until the next NaN.
    """
    new_series = series.copy()
    nan_indices = np.flatnonzero(new_series.isna())

    for i in range(len(nan_indices)):
        start_index = nan_indices[i]
        if i == len(nan_indices) - 1:
            end_index = len(new_series)
        else:
            end_index = nan_indices[i + 1]

        values = new_series.iloc[start_index + 1:end_index]
        mean_value = values.mean()
        new_series.iloc[start_index] = mean_value

    return new_series


def create_aligned_dataframe(series_dict):
    """
    Create a DataFrame from multiple series with different indices.
    Combines all indices and fills in NaN values where a series doesn't have a value.
    """
    if not series_dict:
        return pd.DataFrame()

    # Combine all indices
    all_indices = set()
    for series in series_dict.values():
        all_indices.update(series.index)

    all_indices = sorted(all_indices)

    # Create DataFrame with all series aligned to the combined index
    df = pd.DataFrame(index=all_indices)

    for name, series in series_dict.items():
        df[name] = series.reindex(df.index)

    return df