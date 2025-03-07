import re
import webbrowser
from pathlib import Path
from typing import Dict, Union, Optional, Literal, Any, List, Tuple

import dash
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
from dash.dependencies import Input, Output, State

# Keep existing imports
from agentlib_mpc.utils import TIME_CONVERSION
from agentlib_mpc.utils.analysis import load_mpc, load_mpc_stats
from agentlib_mpc.utils.plotting.basic import EBCColors
from agentlib_mpc.utils.plotting.interactive import get_port, obj_plot, solver_return
from agentlib_mpc.utils.plotting.mpc import interpolate_colors


def reduce_triple_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a triple-indexed DataFrame to a double index by keeping only the rows
    with the largest level 1 index for each unique level 0 index.

    Args:
        df: DataFrame with either double or triple index

    Returns:
        DataFrame with double index
    """
    if len(df.index.levels) == 2:
        return df

    # Group by level 0 and get the maximum level 1 index for each group
    idx = df.index.get_level_values(0)
    sub_idx = df.index.get_level_values(1)
    max_sub_indices = df.groupby(idx)[[]].max().index

    # Create a mask for rows we want to keep
    mask = pd.Series(False, index=df.index)
    for time in max_sub_indices:
        max_sub_idx = df.loc[time].index.get_level_values(0).max()
        mask.loc[(time, max_sub_idx)] = True

    # Apply the mask and drop the middle level
    return df[mask].droplevel(1)


def plot_mpc_plotly(
    series: pd.Series,
    step: bool = False,
    convert_to: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    y_axis_label: str = "",
    use_datetime: bool = False,
    max_predictions: int = 1000,
) -> go.Figure:
    """
    Create a plotly figure from MPC prediction series.

    Args:
        series: Series of MPC predictions with time steps as index
        step: Whether to display step plots (True) or continuous lines (False)
        convert_to: Unit for time conversion
        y_axis_label: Label for y-axis
        use_datetime: Whether to interpret timestamps as datetime
        max_predictions: Maximum number of predictions to show (for performance)

    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    predictions_grouped = series.groupby(level=0)
    number_of_predictions = predictions_grouped.ngroups

    # Sample predictions if there are too many
    if number_of_predictions > max_predictions:
        # Always include the most recent prediction
        most_recent_time = series.index.unique(level=0)[-1]

        # Calculate step size for the remaining predictions
        remaining_slots = max_predictions - 1
        step_size = (number_of_predictions - 1) // remaining_slots

        # Select evenly spaced predictions and combine with most recent
        selected_times = series.index.unique(level=0)[:-1:step_size][:remaining_slots]
        selected_times = pd.Index(list(selected_times) + [most_recent_time])
        predictions_iterator = ((t, series.xs(t, level=0)) for t in selected_times)
        number_of_predictions = max_predictions
    else:
        selected_times = series.index.unique(level=0)
        predictions_iterator = ((t, series.xs(t, level=0)) for t in selected_times)

    # stores the first value of each prediction (only for selected times)
    actual_values: dict[float, float] = {}

    for i, (time_seconds, prediction) in enumerate(predictions_iterator):
        prediction: pd.Series = prediction.dropna()
        prediction = prediction[prediction.index >= 0]

        if use_datetime:
            time_converted = pd.Timestamp(time_seconds, unit="s", tz="UTC").tz_convert(
                "Europe/Berlin"
            )
            relative_times = prediction.index
            try:
                actual_values[time_converted] = prediction.loc[0]
            except KeyError:
                pass
            timedeltas = pd.to_timedelta(relative_times, unit="s")
            base_time = pd.Timestamp(time_seconds, unit="s", tz="UTC")
            prediction.index = base_time + timedeltas
            prediction.index = prediction.index.tz_convert("Europe/Berlin")
        else:
            time_converted = time_seconds / TIME_CONVERSION[convert_to]
            try:
                actual_values[time_converted] = prediction.loc[0]
            except KeyError:
                pass
            prediction.index = (prediction.index + time_seconds) / TIME_CONVERSION[
                convert_to
            ]

        progress = i / number_of_predictions
        prediction_color = interpolate_colors(
            progress=progress,
            colors=[EBCColors.red, EBCColors.dark_grey],
        )

        trace_kwargs = dict(
            x=prediction.index,
            y=prediction,
            mode="lines",
            line=dict(
                color=f"rgb{prediction_color}",
                width=0.7,
                shape="hv" if step else None,
            ),
            name=(
                f"{time_converted}"
                if use_datetime
                else f"{time_converted} {convert_to[0]}"
            ),
            legendgroup="Prediction",
            legendgrouptitle_text="Predictions",
            visible=True,
            legendrank=i + 2,
        )

        fig.add_trace(go.Scattergl(**trace_kwargs))

    actual_series = pd.Series(actual_values)
    fig.add_trace(
        go.Scattergl(
            x=actual_series.index,
            y=actual_series,
            mode="lines",
            line=dict(color="black", width=1.5, shape="hv" if step else None),
            name="Actual Values",
            legendrank=1,
        )
    )

    x_axis_label = "Time" if use_datetime else f"Time in {convert_to}"
    fig.update_layout(
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        uirevision="same",
    )

    return fig


def make_components(
    data: pd.DataFrame,
    convert_to: str,
    stats: Optional[pd.DataFrame] = None,
    use_datetime: bool = False,
    step: bool = False,
) -> html.Div:
    """
    Create dashboard components from MPC data and stats.

    Args:
        data: DataFrame with MPC data
        convert_to: Time unit for plotting
        stats: Optional DataFrame with MPC statistics
        use_datetime: Whether to interpret timestamps as datetime
        step: Whether to use step plots

    Returns:
        Dash HTML Div containing all components
    """
    components = []

    # Add statistics components if available
    if stats is not None:
        # Add solver iterations plot
        solver_plot = solver_return(stats, convert_to)
        if solver_plot is not None:
            components.insert(0, html.Div([solver_plot]))

        # Add objective plot if available
        obj_value_plot = obj_plot(stats, convert_to)
        if obj_value_plot is not None:
            components.insert(1, html.Div([obj_value_plot]))

    # Create one component for each variable
    try:
        for var_type, column in data.columns:
            if var_type == "variable":
                components.append(
                    html.Div(
                        [
                            dcc.Graph(
                                id=f"plot-{column}",
                                figure=plot_mpc_plotly(
                                    data[var_type][column],
                                    step=step,
                                    convert_to=convert_to,
                                    y_axis_label=column,
                                    use_datetime=use_datetime,
                                ),
                                style={
                                    "min-width": "600px",
                                    "min-height": "400px",
                                    "max-width": "900px",
                                    "max-height": "450px",
                                },
                            ),
                        ],
                        className="draggable",
                    )
                )
    except (AttributeError, ValueError) as e:
        # If data is not multi-indexed
        print(f"Error creating components: {e}")

        if isinstance(data.columns, pd.MultiIndex):
            # Handle case with MultiIndex columns but different structure
            for column in data.columns:
                if isinstance(column, tuple) and len(column) >= 2:
                    var_type, col_name = column[0], column[1]
                    if var_type == "variable":
                        try:
                            components.append(
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id=f"plot-{col_name}",
                                            figure=plot_mpc_plotly(
                                                data[column],
                                                step=step,
                                                convert_to=convert_to,
                                                y_axis_label=str(col_name),
                                                use_datetime=use_datetime,
                                            ),
                                            style={
                                                "min-width": "600px",
                                                "min-height": "400px",
                                                "max-width": "900px",
                                                "max-height": "450px",
                                            },
                                        ),
                                    ],
                                    className="draggable",
                                )
                            )
                        except Exception as e2:
                            print(f"Error plotting column {column}: {e2}")

    return html.Div(
        components,
        style={
            "display": "grid",
            "grid-template-columns": "repeat(auto-fit, minmax(600px, 1fr))",
            "grid-gap": "20px",
            "padding": "20px",
            "min-width": "600px",
            "min-height": "200px",
        },
        id="plot-container",
    )


def detect_index_type(data: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Detect the type of index in the DataFrame.

    Args:
        data: DataFrame to check

    Returns:
        Tuple of (is_multi_index, is_datetime)
    """
    is_multi_index = isinstance(data.index, pd.MultiIndex)

    # Check if it's a datetime index (or the first level is datetime)
    if is_multi_index:
        first_level = data.index.levels[0]
        is_datetime = pd.api.types.is_datetime64_any_dtype(first_level)
        if not is_datetime:
            # Check if it might be a Unix timestamp (large integer values)
            if pd.api.types.is_numeric_dtype(first_level):
                is_datetime = (
                    first_level.max() > 1e9
                )  # Simple heuristic for Unix timestamp
    else:
        is_datetime = pd.api.types.is_datetime64_any_dtype(data.index)
        if not is_datetime and pd.api.types.is_numeric_dtype(data.index):
            is_datetime = data.index.max() > 1e9

    return is_multi_index, is_datetime


def show_multi_room_dashboard(
    results: Dict[str, Dict[str, Any]], scale: str = "hours", step: bool = False
):
    """
    Show a dashboard with dropdown selection for different agents/rooms.

    Args:
        results: Dictionary with agent results from mas.get_results()
        scale: Time scale for plotting ("seconds", "minutes", "hours", "days")
        step: Whether to use step plots
    """
    app = dash.Dash(__name__, title="Multi-Agent MPC Results")

    # Get all agents
    agent_ids = list(results.keys())

    if not agent_ids:
        print("No agents found in results dictionary")
        return

    # Find first valid MPC data to determine index type
    first_agent_id = None
    first_module_id = None
    for agent_id in agent_ids:
        for module_id, module_data in results[agent_id].items():
            if isinstance(module_data, pd.DataFrame):
                first_agent_id = agent_id
                first_module_id = module_id
                break
        if first_agent_id:
            break

    if not first_agent_id:
        print("No valid MPC data found in results")
        return

    first_data = results[first_agent_id][first_module_id]
    is_multi_index, use_datetime = detect_index_type(first_data)

    # Create agent and module selector dropdowns
    app.layout = html.Div(
        [
            html.H1("Multi-Agent MPC Results"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Agent:"),
                            dcc.Dropdown(
                                id="agent-selector",
                                options=[
                                    {"label": agent_id, "value": agent_id}
                                    for agent_id in agent_ids
                                ],
                                value=first_agent_id,
                            ),
                        ],
                        style={
                            "width": "300px",
                            "margin": "10px",
                            "display": "inline-block",
                        },
                    ),
                    html.Div(
                        [
                            html.Label("Select Module:"),
                            dcc.Dropdown(
                                id="module-selector",
                                # Options will be set by callback
                            ),
                        ],
                        style={
                            "width": "300px",
                            "margin": "10px",
                            "display": "inline-block",
                        },
                    ),
                ],
            ),
            html.Div(
                html.Button(
                    "Toggle Step Plot", id="toggle-step", style={"margin": "10px"}
                )
            ),
            html.Div(id="agent-dashboard"),
            dcc.Store(id="step-state", data=step),
        ]
    )

    @app.callback(
        [Output("module-selector", "options"), Output("module-selector", "value")],
        [Input("agent-selector", "value")],
    )
    def update_module_options(selected_agent):
        if not selected_agent:
            return [], None

        module_options = []
        first_module = None

        for module_id, module_data in results[selected_agent].items():
            if isinstance(module_data, pd.DataFrame):
                module_options.append({"label": module_id, "value": module_id})
                if first_module is None:
                    first_module = module_id

        return module_options, first_module

    @app.callback(
        Output("step-state", "data"),
        [Input("toggle-step", "n_clicks")],
        [State("step-state", "data")],
    )
    def toggle_step_plot(n_clicks, current_step):
        if n_clicks:
            return not current_step
        return current_step

    @app.callback(
        Output("agent-dashboard", "children"),
        [
            Input("agent-selector", "value"),
            Input("module-selector", "value"),
            Input("step-state", "data"),
        ],
    )
    def update_dashboard(selected_agent, selected_module, step_state):
        if not selected_agent or not selected_module:
            return html.Div("Please select both an agent and a module")

        try:
            data = results[selected_agent][selected_module]

            if not isinstance(data, pd.DataFrame):
                return html.Div(f"Selected module does not contain valid MPC data")

            # Reduce triple index to double index if needed
            if isinstance(data.index, pd.MultiIndex) and len(data.index.levels) > 2:
                data = reduce_triple_index(data)

            # Check if data needs time normalization
            if is_multi_index and not use_datetime:
                try:
                    # Normalize time to start from zero
                    first_time = data.index.levels[0][0]
                    data.index = data.index.set_levels(
                        data.index.levels[0] - first_time, level=0
                    )
                except Exception as e:
                    print(f"Error normalizing time: {e}")

            # Get stats data if available
            stats = None
            if f"{selected_module}_stats" in results[selected_agent]:
                stats = results[selected_agent][f"{selected_module}_stats"]

            # Create the dashboard components
            return make_components(
                data=data,
                convert_to=scale,
                stats=stats,
                use_datetime=use_datetime,
                step=step_state,
            )
        except Exception as e:
            return html.Div(f"Error creating dashboard: {str(e)}")

    # Launch the dashboard
    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def launch_dashboard_from_results(
    results: Dict[str, Dict[str, Any]], scale: str = "hours", step: bool = False
) -> bool:
    """
    Launch the multi-agent dashboard from results dictionary returned by mas.get_results().

    This function checks if the data has the correct shape for visualization,
    and launches the dashboard if valid data is found.

    Args:
        results: Dictionary with agent results from mas.get_results()
        scale: Time scale for plotting ("seconds", "minutes", "hours", "days")
        step: Whether to use step plots

    Returns:
        bool: True if dashboard was launched, False otherwise
    """
    if not results or not isinstance(results, dict):
        print("Invalid results: Expected non-empty dictionary")
        return False

    # Validate results structure
    valid_data_found = False

    for agent_id, agent_data in results.items():
        if not isinstance(agent_data, dict):
            print(f"Warning: Agent '{agent_id}' data is not a dictionary, skipping")
            continue

        for module_id, module_data in agent_data.items():
            if not isinstance(module_data, pd.DataFrame):
                continue

            try:
                # Check if this DataFrame has the expected structure for MPC data
                if isinstance(module_data.index, pd.MultiIndex):
                    if len(module_data.index.levels) > 1:
                        # This looks like MPC data with multi-level index
                        valid_data_found = True
                        break
                else:
                    # Single level index might still be valid for some data
                    valid_data_found = module_data.shape[0] > 0
                    break
            except Exception as e:
                print(f"Error checking data for {agent_id}.{module_id}: {e}")
                continue

        if valid_data_found:
            break

    if not valid_data_found:
        print("No valid MPC data found in results")
        return False

    # Launch the dashboard
    try:
        print(f"Launching dashboard with scale={scale}")
        show_multi_room_dashboard(results, scale=scale, step=step)
        return True
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        return False


# Additional utility function to process results from LocalMASAgency
def process_mas_results(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Process results from LocalMASAgency to prepare them for visualization.

    This function:
    1. Identifies modules with MPC data
    2. Reduces triple indices to double indices if needed
    3. Associates stats data with corresponding MPC data

    Args:
        results: Raw results from mas.get_results()

    Returns:
        Processed results ready for dashboard visualization
    """
    processed_results = {}

    for agent_id, agent_data in results.items():
        processed_results[agent_id] = {}

        # Find all DataFrame modules that could be MPC data
        for module_id, module_data in agent_data.items():
            if not isinstance(module_data, pd.DataFrame):
                continue

            try:
                # Check if this looks like MPC data
                if isinstance(module_data.index, pd.MultiIndex):
                    if isinstance(module_data.columns, pd.MultiIndex):
                        # This is likely MPC data with variables, parameters, etc.
                        processed_results[agent_id][module_id] = module_data
                    elif any(
                        col.startswith(("variable_", "parameter_"))
                        for col in module_data.columns
                    ):
                        # This might be MPC data with flattened column names
                        processed_results[agent_id][module_id] = module_data

                    # Check for stats data with matching prefix
                    stats_module_id = f"{module_id}_stats"
                    if stats_module_id in agent_data and isinstance(
                        agent_data[stats_module_id], pd.DataFrame
                    ):
                        processed_results[agent_id][stats_module_id] = agent_data[
                            stats_module_id
                        ]

            except Exception as e:
                print(f"Error processing {agent_id}.{module_id}: {e}")

    return processed_results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        # If a path is provided as an argument, try to load from files
        path = Path(sys.argv[1])
        if path.exists() and path.is_dir():
            print(f"Loading data from directory: {path}")
            show_multi_room_dashboard_from_files(path, scale="hours")
        else:
            print(f"Directory not found: {path}")
    else:
        print("No directory specified. Please provide a directory path.")
