# visualization_app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import threading
import logging
import time

# Global variable to hold the data for the app instance
# This is simpler for a single, temporary debugging instance launched programmatically.
# For more robust applications, consider Dash's dcc.Store or other state management.
APP_DATA = {}
app = None  # Global app instance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Helper Functions for Plotting ---


def create_spy_plot(matrix, x_labels=None, y_labels=None, title="Sparsity Pattern"):
    """Creates a Plotly heatmap mimicking a spy plot."""
    if matrix is None or matrix.size == 0:
        return go.Figure(layout=go.Layout(title=title + " (No Data)"))

    y_indices, x_indices = np.where(np.abs(matrix) > 1e-9)  # Find non-zero elements

    fig = go.Figure(
        data=go.Scattergl(
            x=x_indices,
            y=y_indices,
            mode="markers",
            marker=dict(
                color="black", size=max(2, 80 / np.sqrt(max(matrix.shape)))
            ),  # Adjust size based on matrix dims
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Column Index",
        yaxis_title="Row Index",
        yaxis=dict(
            autorange="reversed", scaleanchor="x", scaleratio=1
        ),  # Ensure square aspect ratio
        xaxis=dict(range=[-0.5, matrix.shape[1] - 0.5]),
        margin=dict(l=100, r=20, t=50, b=100),  # Adjust margins for labels if needed
        hovermode="closest",
    )

    # Add labels if provided (can be slow for large matrices)
    if x_labels is not None:
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(x_labels))),
                ticktext=x_labels,
                range=[-0.5, len(x_labels) - 0.5],
            )
        )
    if y_labels is not None:
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(y_labels))),
                ticktext=y_labels,
                autorange="reversed",
                scaleanchor="x",
                scaleratio=1,
            )
        )

    return fig


def create_heatmap(matrix, x_labels=None, y_labels=None, title="Heatmap"):
    """Creates a Plotly heatmap."""
    if matrix is None or matrix.size == 0:
        return go.Figure(layout=go.Layout(title=title + " (No Data)"))

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels if x_labels else list(range(matrix.shape[1])),
            y=y_labels if y_labels else list(range(matrix.shape[0])),
            colorscale="Viridis",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Columns",
        yaxis_title="Rows",
        yaxis=dict(
            autorange="reversed", scaleanchor="x", scaleratio=1
        ),  # Ensure square aspect ratio for matrices
        xaxis=dict(tickangle=-45 if x_labels else 0),
        margin=dict(l=100, r=20, t=50, b=100),  # Adjust margins for labels
    )
    if x_labels:
        fig.update_layout(
            xaxis_tickvals=list(range(len(x_labels)))
        )  # Ensure ticks align if labels provided
    if y_labels:
        fig.update_layout(yaxis_tickvals=list(range(len(y_labels))))

    return fig


def create_constraint_plot(
    constraints,
    lbg,
    ubg,
    active_mask,
    lam_g,
    constraint_labels,
    title="Constraint Activity",
):
    """Plots constraint values, bounds, activity, and multipliers."""
    n_constraints = len(constraints)
    if n_constraints == 0:
        return go.Figure(layout=go.Layout(title=title + " (No Data)"))

    indices = list(range(n_constraints))

    # Determine activity status text
    status = []
    for i in range(n_constraints):
        is_active = active_mask[i]
        is_eq = np.isclose(lbg[i], ubg[i])
        if is_active:
            if is_eq:
                status.append("Active (Eq)")
            elif np.isclose(constraints[i], lbg[i]):
                status.append("Active (LB)")
            elif np.isclose(constraints[i], ubg[i]):
                status.append("Active (UB)")
            else:  # Should not happen with correct margin, but good fallback
                status.append("Active (Margin)")
        else:
            status.append("Inactive")

    # Create hover text
    hover_texts = [
        f"Index: {i}<br>Label: {constraint_labels[i]}<br>Value: {constraints[i]:.4g}<br>LB: {lbg[i]:.4g}<br>UB: {ubg[i]:.4g}<br>Multiplier (lam_g): {lam_g[i]:.4g}<br>Status: {status[i]}"
        for i in range(n_constraints)
    ]

    fig = go.Figure()

    # Plot constraint values (color by activity)
    colors = ["red" if active else "blue" for active in active_mask]
    fig.add_trace(
        go.Scattergl(
            x=indices,
            y=constraints,
            mode="markers",
            marker=dict(color=colors, size=6),
            name="Constraint Value",
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    # Plot bounds (only for inequality constraints where bounds differ)
    ineq_indices = [i for i in indices if not np.isclose(lbg[i], ubg[i])]
    if ineq_indices:
        fig.add_trace(
            go.Scattergl(
                x=ineq_indices,
                y=[lbg[i] for i in ineq_indices],
                mode="markers",
                marker=dict(symbol="triangle-down", color="black", size=5),
                name="Lower Bound (Ineq)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=ineq_indices,
                y=[ubg[i] for i in ineq_indices],
                mode="markers",
                marker=dict(symbol="triangle-up", color="black", size=5),
                name="Upper Bound (Ineq)",
                hoverinfo="skip",
            )
        )

    # Plot bounds for equality constraints
    eq_indices = [i for i in indices if np.isclose(lbg[i], ubg[i])]
    if eq_indices:
        fig.add_trace(
            go.Scattergl(
                x=eq_indices,
                y=[lbg[i] for i in eq_indices],  # lbg == ubg here
                mode="markers",
                marker=dict(symbol="x", color="purple", size=5),
                name="Equality Target",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Constraint Index",
        yaxis_title="Value",
        hovermode="closest",
    )
    return fig


def create_vector_plot(vector, labels, title="Vector Plot"):
    """Plots a vector as a bar chart or heatmap."""
    if vector is None or vector.size == 0:
        return go.Figure(layout=go.Layout(title=title + " (No Data)"))

    if vector.size > 100:  # Use heatmap for large vectors
        fig = go.Figure(
            data=go.Heatmap(
                z=[vector.flatten()],  # Requires 2D array
                x=labels,
                colorscale="RdBu",
                zmid=0,  # Center color scale around zero
            )
        )
        fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False)
    else:  # Use bar chart for smaller vectors
        fig = go.Figure(
            data=go.Bar(
                x=labels,
                y=vector.flatten(),
                marker_color=["red" if v < 0 else "blue" for v in vector.flatten()],
            )
        )
        fig.update_layout(xaxis_tickangle=-90)

    fig.update_layout(
        title=title, xaxis_title="Variable Index/Name", yaxis_title="Value"
    )
    return fig


# --- App Layout and Callbacks ---


def build_app_layout(data):
    """Creates the Dash app layout dynamically based on available data."""

    matrix_options = []
    if data.get("jacobian_constraints", None) is not None:
        matrix_options.append(
            {"label": "Jacobian (Constraints)", "value": "jacobian_constraints"}
        )
    if data.get("jacobian_all", None) is not None:
        matrix_options.append(
            {"label": "Jacobian (All Constraints - Debug)", "value": "jacobian_all"}
        )
    if data.get("hessian", None) is not None:
        matrix_options.append({"label": "Hessian (Regularized)", "value": "hessian"})
    if data.get("original_hessian", None) is not None:
        matrix_options.append(
            {"label": "Hessian (Original)", "value": "original_hessian"}
        )

    vector_options = []
    if data.get("gradient", None) is not None:
        vector_options.append({"label": "Gradient (g)", "value": "gradient"})
    if data.get("opt_vars_vector", None) is not None:
        vector_options.append(
            {"label": "Optimal Variables (x)", "value": "opt_vars_vector"}
        )
    if data.get("constraint_multipliers", None) is not None:
        vector_options.append(
            {
                "label": "Constraint Multipliers (lam_g)",
                "value": "constraint_multipliers",
            }
        )

    # Basic mapping info
    mapping_info = "Variable Mapping (Flattened Order):\n\n"
    start_idx = 0
    if data.get("opt_var_labels", None):
        for i, label in enumerate(data["opt_var_labels"]):
            mapping_info += f"{i}: {label}\n"
    else:
        mapping_info += "No variable label data available.\n"

    mapping_info += "\n\nParameter Mapping (Flattened Order):\n\n"
    if data.get("opt_par_labels", None):
        for i, label in enumerate(data["opt_par_labels"]):
            mapping_info += f"{i}: {label}\n"
    else:
        mapping_info += "No parameter label data available.\n"

    return html.Div(
        [
            html.H1("MPC/ALADIN Debug Visualization"),
            dcc.Tabs(
                id="tabs",
                value="tab-matrices",
                children=[
                    # --- Matrices Tab ---
                    dcc.Tab(
                        label="Sparsity & Heatmaps",
                        value="tab-matrices",
                        children=[
                            html.Div(
                                [
                                    html.Label("Select Matrix:"),
                                    dcc.Dropdown(
                                        id="matrix-selector",
                                        options=matrix_options,
                                        value=(
                                            matrix_options[0]["value"]
                                            if matrix_options
                                            else None
                                        ),
                                    ),
                                    html.Label("Visualization Type:"),
                                    dcc.RadioItems(
                                        id="matrix-plot-type",
                                        options=[
                                            {
                                                "label": "Sparsity (Spy Plot)",
                                                "value": "spy",
                                            },
                                            {"label": "Heatmap", "value": "heatmap"},
                                        ],
                                        value="spy",
                                        labelStyle={
                                            "display": "inline-block",
                                            "margin-right": "10px",
                                        },
                                    ),
                                    html.Label(
                                        "Show Labels (Slow for large matrices):"
                                    ),
                                    dcc.Checklist(
                                        id="matrix-show-labels",
                                        options=[{"label": "", "value": "show"}],
                                        value=[],
                                    ),
                                ],
                                style={
                                    "width": "30%",
                                    "display": "inline-block",
                                    "vertical-align": "top",
                                    "padding": "10px",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Loading(
                                        id="loading-matrix-plot",
                                        type="circle",
                                        children=[
                                            dcc.Graph(
                                                id="matrix-plot",
                                                style={"height": "80vh"},
                                            )
                                        ],
                                    )
                                ],
                                style={"width": "70%", "display": "inline-block"},
                            ),
                        ],
                    ),
                    # --- Vectors Tab ---
                    dcc.Tab(
                        label="Vectors",
                        value="tab-vectors",
                        children=[
                            html.Div(
                                [
                                    html.Label("Select Vector:"),
                                    dcc.Dropdown(
                                        id="vector-selector",
                                        options=vector_options,
                                        value=(
                                            vector_options[0]["value"]
                                            if vector_options
                                            else None
                                        ),
                                    ),
                                ],
                                style={"width": "30%", "padding": "10px"},
                            ),
                            dcc.Loading(
                                id="loading-vector-plot",
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="vector-plot", style={"height": "80vh"}
                                    )
                                ],
                            ),
                        ],
                    ),
                    # --- Mapping Tab ---
                    dcc.Tab(
                        label="Variable Mapping",
                        value="tab-mapping",
                        children=[
                            html.H3("NLP Variable/Parameter Order"),
                            dcc.Textarea(
                                id="mapping-textarea",
                                value=mapping_info,
                                style={"width": "100%", "height": "70vh"},
                                readOnly=True,
                            ),
                        ],
                    ),
                    # --- Constraints Tab ---
                    dcc.Tab(
                        label="Constraint Activity",
                        value="tab-constraints",
                        children=[
                            html.Div(
                                "Displays constraint values relative to bounds, activity status, and multipliers."
                            ),
                            dcc.Loading(
                                id="loading-constraint-plot",
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="constraint-plot", style={"height": "80vh"}
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )


# --- Callback Definitions ---
def register_callbacks(app, data):
    @app.callback(
        Output("matrix-plot", "figure"),
        [
            Input("matrix-selector", "value"),
            Input("matrix-plot-type", "value"),
            Input("matrix-show-labels", "value"),
        ],
    )
    def update_matrix_plot(selected_matrix_key, plot_type, show_labels_flag):
        if not selected_matrix_key or selected_matrix_key not in data:
            return go.Figure(layout=go.Layout(title="Select a matrix"))

        matrix = data[selected_matrix_key]
        title = selected_matrix_key.replace("_", " ").title()

        # Determine labels based on matrix type
        x_labels = None
        y_labels = None
        show_labels = "show" in show_labels_flag

        if show_labels:
            if "hessian" in selected_matrix_key.lower():
                x_labels = data.get("opt_var_labels", None)
                y_labels = data.get("opt_var_labels", None)
            elif "jacobian" in selected_matrix_key.lower():
                x_labels = data.get("opt_var_labels", None)
                # Use active constraint labels for the main Jacobian, all for the debug one
                if selected_matrix_key == "jacobian_constraints":
                    y_labels = data.get("active_constraint_labels", None)
                else:  # jacobian_all
                    y_labels = data.get("all_constraint_labels", None)

        if plot_type == "spy":
            return create_spy_plot(
                matrix, x_labels, y_labels, title=f"Sparsity: {title}"
            )
        else:  # heatmap
            return create_heatmap(matrix, x_labels, y_labels, title=f"Heatmap: {title}")

    @app.callback(Output("vector-plot", "figure"), [Input("vector-selector", "value")])
    def update_vector_plot(selected_vector_key):
        if not selected_vector_key or selected_vector_key not in data:
            return go.Figure(layout=go.Layout(title="Select a vector"))

        vector = data[selected_vector_key]
        title = selected_vector_key.replace("_", " ").title()

        labels = None
        if selected_vector_key in ["gradient", "opt_vars_vector"]:
            labels = data.get("opt_var_labels", None)
        elif selected_vector_key == "constraint_multipliers":
            labels = data.get(
                "all_constraint_labels", None
            )  # Multipliers correspond to all constraints

        return create_vector_plot(vector, labels, title)

    @app.callback(
        Output("constraint-plot", "figure"),
        [Input("tabs", "value")],  # Trigger when constraint tab is selected
    )
    def update_constraint_plot(tab_value):
        if tab_value != "tab-constraints":
            return dash.no_update  # Don't update if tab not visible

        constraints = data.get("constraint_values", None)
        lbg = data.get("constraint_lbg", None)
        ubg = data.get("constraint_ubg", None)
        active_mask = data.get("active_constraint_mask", None)
        lam_g = data.get("constraint_multipliers", None)
        labels = data.get("all_constraint_labels", None)

        if any(v is None for v in [constraints, lbg, ubg, active_mask, lam_g, labels]):
            logger.warning("Constraint data missing, cannot generate plot.")
            return go.Figure(
                layout=go.Layout(title="Constraint Activity (Missing Data)")
            )

        # Ensure consistent lengths
        min_len = min(
            len(constraints),
            len(lbg),
            len(ubg),
            len(active_mask),
            len(lam_g),
            len(labels),
        )
        if not (
            len(constraints)
            == len(lbg)
            == len(ubg)
            == len(active_mask)
            == len(lam_g)
            == len(labels)
        ):
            logger.warning(
                f"Constraint data arrays have inconsistent lengths! Truncating to {min_len}."
            )
            constraints = constraints[:min_len]
            lbg = lbg[:min_len]
            ubg = ubg[:min_len]
            active_mask = active_mask[:min_len]
            lam_g = lam_g[:min_len]
            labels = labels[:min_len]

        return create_constraint_plot(constraints, lbg, ubg, active_mask, lam_g, labels)


# --- Launch Function ---


def launch_visualization_app(prepared_data: dict, port=8051):
    """
    Launches the Dash app in a separate thread.

    Args:
        prepared_data: A dictionary containing all necessary data (NumPy arrays, lists).
        port: The port number to run the app on.
    """
    global APP_DATA, app
    APP_DATA = prepared_data  # Make data available globally for callbacks

    # Create app instance only if it doesn't exist or needs reset
    # This is basic; more complex scenarios might need better instance management
    if app is None:
        app = dash.Dash(__name__, suppress_callback_exceptions=True)

    app.layout = build_app_layout(APP_DATA)  # Build layout with current data
    register_callbacks(app, APP_DATA)  # Register callbacks with access to current data

    def run():
        logger.info(f"Starting Dash server on http://127.0.0.1:{port}")
        # Run without reloader and debug mode when launched programmatically
        app.run_server(port=port, debug=False, use_reloader=False, threaded=True)

    # Check if a server is already running (simple check, might need refinement)
    # This prevents multiple servers if called rapidly in a loop
    if (
        not hasattr(launch_visualization_app, "server_thread")
        or not launch_visualization_app.server_thread.is_alive()
    ):
        launch_visualization_app.server_thread = threading.Thread(target=run)
        launch_visualization_app.server_thread.daemon = (
            True  # Allows main program to exit even if thread is running
        )
        launch_visualization_app.server_thread.start()
        logger.info("Dash server thread started.")
    else:
        logger.info("Dash server thread already running. Layout updated.")
        # If you need to forcefully update the layout of an *already running* app,
        # it's more complex and usually involves callbacks triggering updates based
        # on dcc.Store or similar. For simple debugging, restarting the script
        # or just letting the layout update on the next launch might be sufficient.


# --- Helper to Prepare Data (Example Structure) ---
# This function should ideally be a method within your ALADINDiscretization class


def _prepare_visualization_data(
    discretization_obj,
    nlp_inputs,
    nlp_output,
    constraints,
    active_constraints,
    sensitivities_result,
    debug_sensitivities,
):
    """
    Extracts and formats data from the discretization object for the Dash app.
    This should be called from within ALADINDiscretization.solve().

    Args:
        discretization_obj: The instance of ALADINDiscretization (self).
        nlp_inputs: The dictionary of inputs passed to the NLP solver.
        nlp_output: The dictionary of outputs from the NLP solver.
        constraints: Numerical values of constraints at optimum.
        active_constraints: Boolean mask for active constraints.
        sensitivities_result: Dictionary from _sensitivities_func.
        debug_sensitivities: Dictionary holding intermediate sensitivity calculations.

    Returns:
        A dictionary containing data ready for the Dash app.
    """
    data = {}

    # --- Variable/Parameter Labels ---
    opt_var_labels = []
    opt_vars_flat_list = (
        []
    )  # Store the symbolic vars in flattened order if needed later
    for denotation, container in discretization_obj.mpc_opt_vars.items():
        sys_var = next(
            (v for v in discretization_obj._system.variables if v.name == denotation),
            None,
        )  # Get OptimizationVariable definition
        if sys_var:
            full_names = sys_var.full_names
            for i, t in enumerate(container.grid):
                opt_vars_flat_list.extend(container.var[i])  # Add symbolic var
                for name in full_names:
                    opt_var_labels.append(
                        f"{name}_t{t:.2f}"
                    )  # Or use index k if preferred
        else:
            logger.warning(
                f"Could not find system variable definition for {denotation}"
            )
            # Fallback if definition not found (less informative labels)
            for i, t in enumerate(container.grid):
                for j in range(container.var[i].shape[0]):
                    opt_var_labels.append(f"{denotation}_{j}_t{t:.2f}")

    opt_par_labels = []
    opt_pars_flat_list = []
    for denotation, container in discretization_obj.mpc_opt_pars.items():
        sys_par = next(
            (p for p in discretization_obj._system.parameters if p.name == denotation),
            None,
        )  # Get OptimizationParameter definition
        if sys_par:
            full_names = sys_par.full_names
            for i, t in enumerate(container.grid):
                opt_pars_flat_list.extend(container.var[i])
                for name in full_names:
                    opt_par_labels.append(f"{name}_t{t:.2f}")
        else:
            logger.warning(
                f"Could not find system parameter definition for {denotation}"
            )
            for i, t in enumerate(container.grid):
                for j in range(container.var[i].shape[0]):
                    opt_par_labels.append(f"{denotation}_{j}_t{t:.2f}")

    data["opt_var_labels"] = opt_var_labels
    data["opt_par_labels"] = opt_par_labels

    # --- Constraint Labels ---
    # This is tricky without direct access to how constraints were added.
    # We might need to modify add_constraint to store labels.
    # For now, using generic labels.
    num_constraints_total = constraints.shape[0]
    all_constraint_labels = [f"C_{i}" for i in range(num_constraints_total)]
    data["all_constraint_labels"] = all_constraint_labels

    # Try to get labels for *active* constraints based on Jacobian shape
    if (
        sensitivities_result
        and "J" in sensitivities_result
        and sensitivities_result["J"] is not None
    ):
        num_active_constraints = sensitivities_result["J"].shape[0]
        # Assuming J rows correspond to the *first* N active constraints found
        active_indices = np.where(active_constraints)[0][:num_active_constraints]
        data["active_constraint_labels"] = [
            all_constraint_labels[i] for i in active_indices
        ]
        # If J includes inactive constraints (rows are zero), this needs adjustment
    else:
        data["active_constraint_labels"] = [
            f"ActiveC_{i}" for i in np.where(active_constraints)[0]
        ]  # Fallback

    # --- Numerical Data ---
    # Convert CasADi DM to numpy arrays where necessary
    def to_np(val):
        if isinstance(val, (ca.DM, ca.MX)):
            return val.toarray().flatten()  # Flatten vectors
        elif isinstance(val, np.ndarray):
            # Ensure it's flat if it's supposed to be a vector
            if val.ndim > 1 and 1 in val.shape:
                return val.flatten()
            return val  # Keep matrices as 2D
        return val  # Keep other types (like floats, ints)

    data["opt_vars_vector"] = to_np(nlp_output.get("x"))
    data["constraint_values"] = to_np(constraints)
    data["constraint_lbg"] = to_np(nlp_inputs.get("lbg"))
    data["constraint_ubg"] = to_np(nlp_inputs.get("ubg"))
    data["active_constraint_mask"] = to_np(
        active_constraints
    )  # Should already be bool numpy
    data["constraint_multipliers"] = to_np(nlp_output.get("lam_g"))

    # Sensitivities (handle potential None)
    if sensitivities_result:
        data["gradient"] = to_np(sensitivities_result.get("g"))
        data["jacobian_constraints"] = sensitivities_result.get("J")  # Keep as 2D numpy
        data["hessian"] = sensitivities_result.get("H")  # Keep as 2D numpy

    # Debug Sensitivities (handle potential None)
    if debug_sensitivities:
        data["jacobian_all"] = debug_sensitivities.get(
            "jacobian_all"
        )  # Keep as 2D numpy
        data["original_hessian"] = debug_sensitivities.get(
            "H_original"
        )  # Assuming you store it like this
        # Add others if needed, e.g., data['debug_x'] = to_np(debug_sensitivities.get('x'))

    # Filter out None values from data before returning
    return {k: v for k, v in data.items() if v is not None}
