import os
import webbrowser
from pathlib import Path
from typing import List, Dict, Optional, Literal

from agentlib.core.errors import OptionalDependencyError
import pandas as pd


from agentlib_mpc.utils.analysis import load_mpc
from agentlib_mpc.utils.plotting.admm_residuals import load_residuals
from agentlib_mpc.utils.plotting.interactive import get_port

try:
    import dash
    from dash import html, dcc
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
    import dash_daq as daq
except ImportError as e:
    raise OptionalDependencyError(
        dependency_name="interactive",
        dependency_install="plotly, dash",
        used_object="interactive",
    ) from e


def load_agent_data(directory: str) -> Dict[str, pd.DataFrame]:
    """
    Load MPC data for multiple agents from files containing 'admm' in their name.

    Args:
    directory (str): Directory path containing the data files.

    Returns:
    Dict[str, pd.DataFrame]: Dictionary with agent names as keys and their data as values.
    """
    agent_data = {}
    for filename in os.listdir(directory):
        if (
            "admm" in filename.casefold()
            and filename.endswith(".csv")
            and "stats" not in filename.casefold()
        ):
            file_path = os.path.join(directory, filename)
            agent_name = f"Agent_{len(agent_data) + 1}"
            try:
                agent_data[agent_name] = load_mpc(file_path)
            except Exception as e:
                print(f"Error loading file {filename}: {str(e)}")
    return agent_data


def get_coupling_variables(df: pd.DataFrame) -> List[str]:
    """
    Identify coupling variables in the dataframe.

    Args:
    df (pd.DataFrame): The MPC data for an agent.

    Returns:
    List[str]: List of coupling variable names.
    """
    coupling_vars = []
    for col in df.columns:
        if col[0] == "parameter" and col[1].startswith("admm_coupling_mean_"):
            var_name = col[1].replace("admm_coupling_mean_", "")
            if ("variable", var_name) in df.columns:
                coupling_vars.append(var_name)
    return coupling_vars


def get_data_for_plot(
    agent_data: Dict[str, pd.DataFrame],
    time_step: float,
    iteration: int,
    coupling_var: str,
) -> Dict[str, List[float]]:
    """
    Extract data for the coupling variable plot.

    Args:
    agent_data (Dict[str, pd.DataFrame]): Dictionary containing data for each agent.
    time_step (float): Selected time step.
    iteration (int): Selected iteration number.
    coupling_var (str): Name of the selected coupling variable.

    Returns:
    Dict[str, List[float]]: Dictionary with agent names as keys and their values as lists.
    """
    plot_data = {}
    prediction_grid = None

    for agent_name, df in agent_data.items():
        try:
            agent_data_at_step = df.loc[(time_step, iteration)]
            agent_values = agent_data_at_step[("variable", coupling_var)].values
            plot_data[agent_name] = agent_values.tolist()

            if prediction_grid is None:
                prediction_grid = agent_data_at_step.index.tolist()

            # Get mean value (assuming it's the same for all agents)
            if "Mean" not in plot_data:
                mean_values = agent_data_at_step[
                    ("parameter", f"admm_coupling_mean_{coupling_var}")
                ].values
                plot_data["Mean"] = mean_values.tolist()
        except KeyError:
            continue  # Skip this agent if data is not available for the selected time step and iteration

    return plot_data, prediction_grid


def create_coupling_var_plot(
    plot_data: Dict[str, List[float]], prediction_grid: List[float], coupling_var: str
) -> go.Figure:
    """
    Create a plotly figure for the coupling variable plot.

    Args:
    plot_data (Dict[str, List[float]]): Dictionary with agent names as keys and their values as lists.
    prediction_grid (List[float]): List of prediction grid values.
    coupling_var (str): Name of the coupling variable.

    Returns:
    go.Figure: Plotly figure object.
    """
    fig = go.Figure()

    for agent_name, values in plot_data.items():
        if agent_name == "Mean":
            line_style = dict(color="red", dash="dash", width=2)
        else:
            line_style = dict(width=1)

        fig.add_trace(
            go.Scatter(
                x=prediction_grid,
                y=values,
                mode="lines",
                name=agent_name,
                line=line_style,
            )
        )

    fig.update_layout(
        title=f"Coupling Variable: {coupling_var}",
        xaxis_title="Prediction Grid",
        yaxis_title="Value",
        legend_title="Legend",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def get_max_iterations_per_timestep(
    agent_data: Dict[str, pd.DataFrame],
) -> Dict[float, int]:
    max_iterations = {}
    for df in agent_data.values():
        for time_step in df.index.get_level_values(0).unique():
            iterations = df.loc[time_step].index.get_level_values(0).max()
            if (
                time_step not in max_iterations
                or iterations > max_iterations[time_step]
            ):
                max_iterations[time_step] = iterations
    return max_iterations


def create_residuals_plot(residuals_df: pd.DataFrame, time_step: float) -> go.Figure:
    """
    Create a plotly figure for the residuals plot.

    Args:
    residuals_df (pd.DataFrame): DataFrame containing residuals data.
    time_step (float): Selected time step.

    Returns:
    go.Figure: Plotly figure object.
    """
    fig = go.Figure()

    residuals_data = residuals_df.loc[time_step]

    if len(residuals_data) == 1:  # Only one iteration (iteration = 0)
        primal_residual = residuals_data["primal_residual"].iloc[0]
        dual_residual = residuals_data["dual_residual"].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[primal_residual, primal_residual],
                mode="lines",
                name="Primal Residual",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[dual_residual, dual_residual],
                mode="lines",
                name="Dual Residual",
            )
        )

        fig.update_layout(
            xaxis_range=[0, 1],
            xaxis_title="Iteration",
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=residuals_data.index,
                y=residuals_data["primal_residual"],
                mode="lines",
                name="Primal Residual",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=residuals_data.index,
                y=residuals_data["dual_residual"],
                mode="lines",
                name="Dual Residual",
            )
        )

        fig.update_layout(
            xaxis_title="Iteration",
        )

    fig.update_layout(
        title="Primal and Dual Residuals",
        yaxis_title="Residual Value",
        yaxis_type="log",
        yaxis=dict(
            tickformat=".2e",  # Use scientific notation with 2 decimal places
            exponentformat="e",  # Use "e" notation for exponents
        ),
        legend_title="Legend",
    )

    return fig


def create_app(agent_data: Dict[str, pd.DataFrame], residuals_df: pd.DataFrame):
    """
    Create and configure the Dash app.

    Args:
    agent_data (Dict[str, pd.DataFrame]): Dictionary containing data for each agent.
    residuals_df (pd.DataFrame): DataFrame containing residuals data.

    Returns:
    dash.Dash: Configured Dash app.
    """
    app = dash.Dash(__name__)

    # Get time steps and iteration numbers
    first_agent_data = next(iter(agent_data.values()))
    time_steps = sorted(first_agent_data.index.get_level_values(0).unique())
    max_iterations_per_timestep = get_max_iterations_per_timestep(agent_data)
    overall_max_iterations = max(max_iterations_per_timestep.values())

    # Get coupling variables
    coupling_vars = get_coupling_variables(first_agent_data)

    app.layout = html.Div(
        [
            html.H1("Distributed MPC with ADMM Dashboard"),
            # time step slider
            html.Div(
                [
                    html.Label("Time Step"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Slider(
                                        id="time-step-slider",
                                        min=0,
                                        max=len(time_steps) - 1,
                                        value=0,
                                        marks={
                                            i: f"{time_steps[i]:.2f}"
                                            for i in range(
                                                0,
                                                len(time_steps),
                                                max(1, len(time_steps) // 10),
                                            )
                                        },
                                        step=1,
                                    )
                                ],
                                style={
                                    "width": "80%",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                },
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "◀",
                                        id="prev-time-step",
                                        n_clicks=0,
                                        style={"marginRight": "5px"},
                                    ),
                                    html.Button(
                                        "▶",
                                        id="next-time-step",
                                        n_clicks=0,
                                        style={"marginRight": "5px"},
                                    ),
                                    html.Div(
                                        id="time-step-display",
                                        style={
                                            "display": "inline-block",
                                            "marginRight": "10px",
                                        },
                                    ),
                                    daq.NumericInput(
                                        id="time-step-input",
                                        min=0,
                                        max=len(time_steps) - 1,
                                        value=0,
                                        size=60,
                                    ),
                                ],
                                style={
                                    "width": "20%",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                    "textAlign": "right",
                                },
                            ),
                        ]
                    ),
                ]
            ),
            # iteration slide
            html.Div(
                [
                    html.Label("Iteration Number"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Slider(
                                        id="iteration-slider",
                                        min=0,
                                        max=overall_max_iterations,
                                        value=0,
                                        marks={
                                            i: str(i)
                                            for i in range(
                                                0,
                                                overall_max_iterations + 1,
                                                max(1, overall_max_iterations // 10),
                                            )
                                        },
                                        step=1,
                                    )
                                ],
                                style={
                                    "width": "80%",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                },
                            ),
                            html.Div(
                                [
                                    daq.NumericInput(
                                        id="iteration-input",
                                        min=0,
                                        max=overall_max_iterations,
                                        value=0,
                                        size=60,
                                    )
                                ],
                                style={
                                    "width": "20%",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                    "textAlign": "right",
                                },
                            ),
                        ]
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Coupling Variable"),
                    dcc.Dropdown(
                        id="coupling-var-dropdown",
                        options=[{"label": var, "value": var} for var in coupling_vars],
                        value=coupling_vars[0] if coupling_vars else None,
                    ),
                ]
            ),
            dcc.Graph(id="coupling-var-plot"),
            dcc.Graph(id="residuals-plot"),
            dcc.Store(id="y-axis-range"),
        ]
    )

    @app.callback(
        [
            Output("time-step-input", "value"),
            Output("time-step-display", "children"),
            Output("time-step-slider", "value"),
        ],
        [
            Input("time-step-slider", "value"),
            Input("prev-time-step", "n_clicks"),
            Input("next-time-step", "n_clicks"),
            Input("time-step-input", "value"),
        ],
        [State("time-step-input", "value")],
    )
    def update_time_step(
        slider_value, prev_clicks, next_clicks, input_value, current_value
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return (
                current_value,
                f"Current time step: {time_steps[current_value]:.2f}",
                current_value,
            )

        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if input_id == "time-step-slider":
            new_value = slider_value
        elif input_id == "prev-time-step":
            new_value = max(0, current_value - 1)
        elif input_id == "next-time-step":
            new_value = min(len(time_steps) - 1, current_value + 1)
        elif input_id == "time-step-input":
            new_value = input_value
        else:
            new_value = current_value

        return new_value, f"Current time step: {time_steps[new_value]:.2f}", new_value

    @app.callback(
        [
            Output("iteration-slider", "max"),
            Output("iteration-slider", "marks"),
            Output("iteration-input", "max"),
        ],
        [Input("time-step-input", "value")],
    )
    def update_iteration_range(time_step_index):
        time_step = time_steps[time_step_index]
        max_iter = max_iterations_per_timestep[time_step]
        marks = {i: str(i) for i in range(0, max_iter + 1, max(1, max_iter // 10))}
        return max_iter, marks, max_iter

    @app.callback(
        Output("coupling-var-plot", "figure"),
        [
            Input("time-step-input", "value"),
            Input("iteration-input", "value"),
            Input("coupling-var-dropdown", "value"),
            Input("y-axis-range", "data"),
        ],
    )
    def update_coupling_var_plot(time_step_index, iteration, coupling_var, y_range):
        if coupling_var is None:
            return go.Figure()

        time_step = time_steps[time_step_index]
        max_iter = max_iterations_per_timestep[time_step]
        iteration = min(iteration, max_iter)

        plot_data, prediction_grid = get_data_for_plot(
            agent_data, time_step, iteration, coupling_var
        )
        fig = create_coupling_var_plot(plot_data, prediction_grid, coupling_var)

        if y_range is not None:
            fig.update_layout(yaxis_range=y_range)

        return fig

    @app.callback(
        Output("y-axis-range", "data"),
        [Input("time-step-input", "value"), Input("coupling-var-dropdown", "value")],
    )
    def compute_y_axis_range(time_step_index, coupling_var):
        if coupling_var is None:
            return None

        time_step = time_steps[time_step_index]

        max_vals = []
        min_vals = []
        for agent, data in agent_data.items():
            try:
                step_data = data[("variable", coupling_var)][time_step]
            except KeyError:
                continue
            max_vals.append(step_data.max())
            min_vals.append(step_data.min())

        y_min = min(min_vals)
        y_max = max(max_vals)

        y_range = [y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)]
        return y_range

    @app.callback(
        Output("residuals-plot", "figure"),
        [Input("time-step-input", "value")],
    )
    def update_residuals_plot(time_step_index):
        time_step = time_steps[time_step_index]

        # Check if residuals data exists for this time step
        if time_step not in residuals_df.index:
            # If no data, return an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No residuals data available for this time step",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        return create_residuals_plot(residuals_df, time_step)

    return app


def main():
    # Specify the directory containing the data files
    data_directory = Path(r"D:\repos\juelich_mpc\juelich_mpc\mpc\simple_model\results")

    # Load agent data
    agent_data = load_agent_data(data_directory)
    agent_data["heating"] = load_mpc(Path(data_directory, "heating_agent_res.csv"))
    # "room_1": load_mpc(Path(data_directory, "room_1_admm.csv")),
    # "room_2": load_mpc(Path(data_directory, "room_2_admm.csv")),
    # "room_3": load_mpc(Path(data_directory, "room_3_admm.csv")),
    # "heating": load_mpc(Path(data_directory, "heating_agent_res.csv")),

    # Load residuals data
    residuals_file = os.path.join(
        data_directory, "residuals.csv"
    )  # Adjust the filename as needed
    residuals_df = load_residuals(residuals_file)

    # Create and run the app
    app = create_app(agent_data, residuals_df)
    port = get_port()

    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run_server(debug=False, port=port)


def show_admm_dashboard(
    data: dict[str, pd.DataFrame],
    residuals: Optional[pd.DataFrame] = None,
    scale: Literal["seconds", "minutes", "hours", "days"] = "seconds",
):
    app = create_app(data, residuals)
    port = get_port()

    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run_server(debug=False, port=port)


if __name__ == "__main__":
    main()
