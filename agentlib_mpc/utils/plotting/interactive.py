import pandas as pd
from typing import Literal, Optional

import socket
import webbrowser

from agentlib.core.errors import OptionalDependencyError

from agentlib_mpc.utils import TIME_CONVERSION
from agentlib_mpc.utils.analysis import load_mpc
from agentlib_mpc.utils.plotting.basic import EBCColors
from agentlib_mpc.utils.plotting.mpc import interpolate_colors

try:
    import dash
    from dash import html, dcc
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
except ImportError as e:
    raise OptionalDependencyError(
        dependency_name="interactive",
        dependency_install="plotly, dash",
        used_object="interactive",
    ) from e


def make_figure_plotly() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Interactive Plot Example",
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(args=["type", "scatter"], label="Line", method="restyle"),
                        # dict(
                        #     args=["type", "bar"],
                        #     label="Bar",
                        #     method="restyle"
                        # ),
                    ]
                ),
            ),
        ],
    )

    # This function toggles the visibility of a trace (line) in the plot
    def toggle_traces(trace, points, selector):
        if len(points.trace_indexes) == 0:  # No point has been clicked, do nothing
            return
        trace_index = points.trace_indexes[0]  # Get the index of the clicked trace
        fig.data[trace_index].visible = not fig.data[
            trace_index
        ].visible  # Toggle visibility

    # Add Click event handler to traces
    fig.for_each_trace(lambda trace: trace.on_click(toggle_traces))
    return fig


def plot_mpc_plotly(
    series: pd.Series,
    step: bool = False,
    convert_to: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    y_axis_label: str = "",
) -> go.Figure:
    """
    Args:
        title:
        y_axis_label:
        series: A column of the MPC results Dataframe
        plot_actual_values: whether the closed loop actual values at the start of each
         optimization should be plotted (default True)
        plot_predictions: whether all predicted trajectories should be plotted
        step: whether to use a step plot or a line plot
        convert_to: Will convert the index of the returned series to the specified unit
         (seconds, minutes, hours, days)

    Returns:
        Figure
    """
    fig = go.Figure()
    number_of_predictions: int = series.index.unique(level=0).shape[0]

    # stores the first value of each prediction
    actual_values: dict[float, float] = {}

    for i, (time_seconds, prediction) in enumerate(series.groupby(level=0)):
        prediction: pd.Series = prediction.dropna().droplevel(0)

        time_converted = time_seconds / TIME_CONVERSION[convert_to]
        actual_values[time_converted] = prediction.loc[0]
        prediction.index = (prediction.index + time_seconds) / TIME_CONVERSION[
            convert_to
        ]

        progress = i / number_of_predictions
        prediction_color = interpolate_colors(
            progress=progress,
            colors=[EBCColors.red, EBCColors.dark_grey],
        )
        if not step:
            fig.add_trace(
                go.Scatter(
                    x=prediction.index,
                    y=prediction,
                    mode="lines",
                    line=dict(color=f"rgb{prediction_color}", width=0.7),
                    name=f"{time_converted} {convert_to[0]}",
                    legendgroup=f"Prediction",
                    legendgrouptitle_text=f"Predictions",
                    visible=True,
                    legendrank=i + 2,
                    # id=f"trace-{y_axis_label}-{i}",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=prediction.index,
                    y=prediction,
                    mode="lines",
                    line=dict(
                        color=f"rgb{prediction_color}",
                        width=0.7,
                        shape="hv",
                    ),
                    name=f"{time_converted} {convert_to[0]}",
                    legendgroup=f"Prediction",
                    legendgrouptitle_text=f"Predictions",
                    visible=True,
                    legendrank=i + 2,
                    # id=f"trace-{y_axis_label}-{i}",
                )
            )

    actual_series = pd.Series(actual_values)
    if not step:
        fig.add_trace(
            go.Scatter(
                x=actual_series.index,
                y=actual_series,
                mode="lines",
                line=dict(color="black", width=1.5),
                name="Actual Values",
                legendrank=1,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=actual_series.index,
                y=actual_series,
                mode="lines",
                line=dict(color="black", width=1.5, shape="hv"),
                name="Actual Values",
                legendrank=1,
            )
        )

    # Update x-axis label based on convert_to argument
    x_axis_label = f"Time in {convert_to}"

    fig.update_layout(
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        uirevision="same",  # Add this line
    )

    return fig


def plot_admm_plotly(
    series: pd.Series,
    plot_actual_values: bool = True,
    plot_predictions: bool = False,
    step: bool = False,
    convert_to: Literal["seconds", "minutes", "hours", "days"] = "seconds",
):
    """
    Args:
        series: A column of the MPC results Dataframe
        fig: Plotly figure to plot on
        plot_actual_values: whether the closed loop actual values at the start of each
         optimization should be plotted (default True)
        plot_predictions: whether all predicted trajectories should be plotted
        step: whether to use a step plot or a line plot
        convert_to: Will convert the index of the returned series to the specified unit
         (seconds, minutes, hours, days)

    Returns:
        None
    """
    grid = series.index.get_level_values(2).unique()
    tail_length = len(grid[grid >= 0])
    series_final_predictions = series.groupby(level=0).tail(tail_length).droplevel(1)
    plot_mpc_plotly(
        series=series_final_predictions,
        step=step,
        convert_to=convert_to,
    )


def show_dashboard(
    data: pd.DataFrame,
    stats: Optional[pd.DataFrame] = None,
    scale: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    port: Optional[int] = None,
):
    app = dash.Dash(__name__, title="MPC Results")

    # Get the list of columns from the DataFrame, and check if they can be plotted
    columns = data["variable"].columns
    columns_okay = []
    for column in columns:
        try:
            fig = plot_mpc_plotly(
                data["variable"][column],
                convert_to=scale,
                y_axis_label=column,
            )
            columns_okay.append(column)
        except Exception:
            pass

    # Store initial figures
    initial_figures = {}
    for column in columns_okay:
        fig = plot_mpc_plotly(
            data["variable"][column],
            convert_to=scale,
            y_axis_label=column,
        )
        # Add uirevision to maintain legend state
        fig.update_layout(uirevision="same")
        initial_figures[column] = fig

    # Define the layout of the webpage
    app.layout = html.Div(
        [
            html.H1("MPC Results"),
            # Store for keeping track of trace visibility
            dcc.Store(id="trace-visibility", data={}),
            make_components(columns_okay, data, stats=stats, convert_to=scale),
        ]
    )

    if port is None:
        port = get_port()

    @app.callback(
        [Output(f"plot-{column}", "figure") for column in columns_okay],
        [Input(f"plot-{column}", "restyleData") for column in columns_okay],
        [State(f"plot-{column}", "figure") for column in columns_okay],
    )
    def update_plots(*args):
        ctx = dash.callback_context
        if not ctx.triggered:
            return [dash.no_update] * len(columns_okay)

        n_plots = len(columns_okay)
        restyle_data = args[:n_plots]
        current_figures = args[n_plots:]

        # Find which plot was changed
        triggered_prop = ctx.triggered[0]["prop_id"].split(".")[0]
        triggered_index = next(
            i for i, col in enumerate(columns_okay) if f"plot-{col}" == triggered_prop
        )
        triggered_data = restyle_data[triggered_index]

        if not triggered_data:
            return [dash.no_update] * n_plots

        # Get the visibility update from the triggered plot
        visibility_update = triggered_data[0].get("visible", [None])[0]
        trace_indices = triggered_data[1]

        # Update all figures
        updated_figures = []
        for fig in current_figures:
            # Ensure uirevision is set
            fig["layout"]["uirevision"] = "same"
            # Update visibility for the corresponding traces
            for idx in trace_indices:
                fig["data"][idx]["visible"] = visibility_update
            updated_figures.append(fig)

        return updated_figures

    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def make_components(
    columns, data, convert_to, stats: Optional[pd.DataFrame] = None
) -> [html.Div]:
    components = [
        html.Div(
            [
                # html.H3(column),
                dcc.Graph(
                    id=f"plot-{column}",
                    figure=plot_mpc_plotly(
                        data["variable"][column],
                        convert_to=convert_to,
                        y_axis_label=column,
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
        for column in columns
    ]
    if stats is not None:
        components.insert(0, html.Div([solver_return(stats, convert_to)]))
        components.insert(
            1, html.Div([obj_plot(stats, convert_to)])
        )  # Add the "obj" plot

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


def obj_plot(
    data, convert_to: Literal["seconds", "minutes", "hours", "days"] = "seconds"
) -> dcc.Graph:
    df = data.copy()
    index = df.index.values / TIME_CONVERSION[convert_to]

    trace = go.Scatter(
        x=index,
        y=df["obj"],
        mode="lines",
        name="Objective Value",
    )

    layout = go.Layout(
        title="Objective Value",
        xaxis_title=f"Time in {convert_to}",
        yaxis_title="Objective Value",
        showlegend=True,
    )

    fig = go.Figure(data=[trace], layout=layout)

    return dcc.Graph(
        id="plot-obj",
        figure=fig,
        style={
            "min-width": "600px",
            "min-height": "400px",
            "max-width": "900px",
            "max-height": "450px",
        },
    )


def solver_return(
    data, convert_to: Literal["seconds", "minutes", "hours", "days"] = "seconds"
) -> dcc.Graph:
    solver_data = []
    indices = []
    j = 0
    for i in reversed(data.index.values):
        if i in indices:
            break
        j += 1
        indices.append(i)
        solver_data.append(data.iloc[len(data) - j])
    df = pd.DataFrame(solver_data)
    df = df.iloc[::-1]

    return_status = {}
    for idx, success in df.success.items():
        if success:
            solver_return = df.return_status[idx]
        else:
            solver_return = "Solve_Not_Succeeded"
        return_status[idx] = solver_return

    solver_returns = pd.Series(return_status)
    index = solver_returns.index.values / TIME_CONVERSION[convert_to]

    colors = {
        "Solve_Succeeded": "green",
        "Solved_To_Acceptable_Level": "orange",
        "Solve_Not_Succeeded": "red",
    }
    legend_names = {
        "Solved_To_Acceptable_Level": "Acceptable",
        "Solve_Succeeded": "Optimal",
        "Solve_Not_Succeeded": "Failure",
    }

    traces = []
    for status in colors:
        mask = solver_returns.values == status
        if mask.any():
            trace = go.Scatter(
                x=index[mask],
                y=df.loc[solver_returns.index[mask], "iter_count"],
                mode="markers",
                marker=dict(
                    color=colors[status],
                    size=10,
                ),
                name=legend_names[status],
            )
        else:
            trace = go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color=colors[status],
                    size=10,
                ),
                name=legend_names[status],
            )
        traces.append(trace)

    layout = go.Layout(
        title="Solver Return Status",
        xaxis_title=f"Time in {convert_to}",
        yaxis_title="Iterations",
        showlegend=True,
    )

    fig = go.Figure(data=traces, layout=layout)

    return dcc.Graph(
        id="plot-solver-return",
        figure=fig,
        style={
            "min-width": "600px",
            "min-height": "400px",
            "max-width": "900px",
            "max-height": "450px",
        },
    )


def draggable_script():
    return html.Script(
        """
        var draggableElements = document.getElementsByClassName('draggable');
        for (var i = 0; i < draggableElements.length; i++) {
            var element = draggableElements[i];
            element.addEventListener('mousedown', function(e) {
                var offset = [
                    this.offsetLeft - e.clientX,
                    this.offsetTop - e.clientY
                ];
                var moveHandler = function(e) {
                    element.style.left = (e.clientX + offset[0]) + 'px';
                    element.style.top = (e.clientY + offset[1]) + 'px';
                };
                document.addEventListener('mousemove', moveHandler);
                document.addEventListener('mouseup', function() {
                    document.removeEventListener('mousemove', moveHandler);
                });
            });
        }
    """
    )


def get_port():
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_free = s.connect_ex(("localhost", port)) != 0
        if is_free:
            return port
        else:
            port += 1


if __name__ == "__main__":
    data_ = load_mpc(
        r"D:\repos\agentlib_mpc\examples\one_room_mpc\physical\results\mpc.csv"
    )
    show_dashboard(data_)
    # fig = plot_mpc_plotly(
    #     data["variable"]["T"] - 273.15,
    #     y_axis_label="Room temperature",
    #     convert_to="minutes",
    #     step=False,
    # )
    # fig.show()
