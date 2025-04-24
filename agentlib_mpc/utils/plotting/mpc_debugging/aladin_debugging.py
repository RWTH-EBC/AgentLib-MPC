# visualization_app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import threading
import logging
import time
import casadi as ca

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
        constraints: Numerical values of constraints at optimum (numpy array).
        active_constraints: Boolean mask for active constraints (numpy array).
        sensitivities_result: Dictionary from _sensitivities_func (numpy arrays).
        debug_sensitivities: Dictionary holding intermediate sensitivity calculations (numpy arrays).

    Returns:
        A dictionary containing data ready for the Dash app.
    """
    data = {}
    logger.info("Preparing visualization data...")

    # --- Variable/Parameter Labels ---
    opt_var_labels = []
    # REMOVED: opt_vars_flat_list = [] # Store the symbolic vars in flattened order if needed later
    if not hasattr(discretization_obj, "_system") or discretization_obj._system is None:
        logger.error(
            "Discretization object missing '_system' attribute. Cannot generate detailed labels."
        )
        # Add fallback logic or return empty data if labels are critical
        return {}  # Or handle differently

    if not hasattr(discretization_obj, "mpc_opt_vars"):
        logger.error(
            "Discretization object missing 'mpc_opt_vars'. Cannot generate labels."
        )
        return {}

    logger.debug(
        f"Generating labels for opt_vars: {list(discretization_obj.mpc_opt_vars.keys())}"
    )
    for denotation, container in discretization_obj.mpc_opt_vars.items():
        # Find the corresponding system variable definition
        sys_var = next(
            (v for v in discretization_obj._system.variables if v.name == denotation),
            None,
        )
        if sys_var:
            full_names = sys_var.full_names
            if not full_names:  # Skip if variable group is empty
                logger.debug(f"Skipping empty variable group: {denotation}")
                continue
            if not container.grid:
                logger.debug(f"Skipping variable group with empty grid: {denotation}")
                continue

            logger.debug(
                f"  Processing {denotation}: {len(full_names)} names, {len(container.grid)} grid points."
            )
            for i, t in enumerate(container.grid):
                # REMOVED: opt_vars_flat_list.extend(container.var[i]) # Add symbolic var - THIS CAUSED THE ERROR
                # Check if container.var has enough elements for index i
                if i < len(container.var):
                    # Generate labels based on names and time
                    for name in full_names:
                        opt_var_labels.append(
                            f"{name}_t{t:.2f}"
                        )  # Or use index k if preferred
                else:
                    logger.warning(
                        f"Grid length mismatch for {denotation}. Grid has {len(container.grid)} points, but var list has {len(container.var)} elements."
                    )
                    break  # Stop processing this variable group if mismatch occurs
        else:
            logger.warning(
                f"Could not find system variable definition for {denotation}. Using basic labels."
            )
            # Fallback if definition not found (less informative labels)
            if not container.grid:
                continue
            num_vars_per_step = container.var[0].shape[0] if container.var else 0
            if num_vars_per_step == 0:
                continue
            for i, t in enumerate(container.grid):
                if i < len(container.var):
                    for j in range(num_vars_per_step):
                        opt_var_labels.append(f"{denotation}_{j}_t{t:.2f}")
                else:
                    logger.warning(f"Grid length mismatch for {denotation} (fallback).")
                    break

    opt_par_labels = []
    # REMOVED: opt_pars_flat_list = []
    if not hasattr(discretization_obj, "mpc_opt_pars"):
        logger.error(
            "Discretization object missing 'mpc_opt_pars'. Cannot generate labels."
        )
        return {}

    logger.debug(
        f"Generating labels for opt_pars: {list(discretization_obj.mpc_opt_pars.keys())}"
    )
    for denotation, container in discretization_obj.mpc_opt_pars.items():
        sys_par = next(
            (p for p in discretization_obj._system.parameters if p.name == denotation),
            None,
        )
        if sys_par:
            full_names = sys_par.full_names
            if not full_names:
                continue
            if not container.grid:
                continue
            logger.debug(
                f"  Processing {denotation}: {len(full_names)} names, {len(container.grid)} grid points."
            )
            for i, t in enumerate(container.grid):
                # REMOVED: opt_pars_flat_list.extend(container.var[i])
                if i < len(container.var):
                    for name in full_names:
                        opt_par_labels.append(f"{name}_t{t:.2f}")
                else:
                    logger.warning(f"Grid length mismatch for parameter {denotation}.")
                    break
        else:
            logger.warning(
                f"Could not find system parameter definition for {denotation}. Using basic labels."
            )
            if not container.grid:
                continue
            num_pars_per_step = container.var[0].shape[0] if container.var else 0
            if num_pars_per_step == 0:
                continue
            for i, t in enumerate(container.grid):
                if i < len(container.var):
                    for j in range(num_pars_per_step):
                        opt_par_labels.append(f"{denotation}_{j}_t{t:.2f}")
                else:
                    logger.warning(
                        f"Grid length mismatch for parameter {denotation} (fallback)."
                    )
                    break

    data["opt_var_labels"] = opt_var_labels
    data["opt_par_labels"] = opt_par_labels
    logger.info(
        f"Generated {len(opt_var_labels)} variable labels and {len(opt_par_labels)} parameter labels."
    )

    # --- Constraint Labels ---
    num_constraints_total = constraints.shape[0]
    all_constraint_labels = [f"C_{i}" for i in range(num_constraints_total)]
    data["all_constraint_labels"] = all_constraint_labels
    logger.debug(f"Generated {num_constraints_total} generic constraint labels.")

    # Try to get labels for *active* constraints based on Jacobian shape
    if (
        sensitivities_result
        and "J" in sensitivities_result
        and sensitivities_result["J"] is not None
    ):
        num_active_rows_in_J = sensitivities_result["J"].shape[0]
        # Find the indices of the active constraints
        active_indices = np.where(active_constraints)[0]

        # Check if J's row count matches the number of active constraints
        if num_active_rows_in_J == len(active_indices):
            data["active_constraint_labels"] = [
                all_constraint_labels[i] for i in active_indices
            ]
            logger.debug(
                f"Generated {len(active_indices)} active constraint labels based on J shape."
            )
        else:
            # If J doesn't match the active count, the assumption is wrong.
            # Maybe J includes inactive constraints, or the active_constraints mask is off.
            logger.warning(
                f"Jacobian J has {num_active_rows_in_J} rows, but found {len(active_indices)} active constraints. Using generic active labels."
            )
            # Fallback: Generate generic labels for the number of rows in J
            data["active_constraint_labels"] = [
                f"ActiveC_J_{i}" for i in range(num_active_rows_in_J)
            ]
    elif active_constraints is not None:
        # Fallback if J is not available but active mask is
        active_indices = np.where(active_constraints)[0]
        data["active_constraint_labels"] = [
            all_constraint_labels[i] for i in active_indices
        ]
        logger.debug(
            f"Generated {len(active_indices)} active constraint labels based on mask (J not available)."
        )
    else:
        data["active_constraint_labels"] = []  # No active constraint info

    # --- Numerical Data ---
    # Convert CasADi DM to numpy arrays where necessary
    def to_np(val, name="variable"):
        if isinstance(val, (ca.DM, ca.MX)):
            try:
                arr = val.toarray()
                # Flatten if it's clearly a vector (one dimension is 1)
                if arr.ndim > 1 and (arr.shape[0] == 1 or arr.shape[1] == 1):
                    return arr.flatten()
                return arr  # Keep matrices as 2D
            except Exception as e:
                logger.error(f"Could not convert CasADi matrix '{name}' to numpy: {e}")
                return None
        elif isinstance(val, np.ndarray):
            # Ensure it's flat if it's supposed to be a vector
            if val.ndim > 1 and (val.shape[0] == 1 or val.shape[1] == 1):
                return val.flatten()
            return val  # Keep matrices as 2D
        elif val is None:
            logger.warning(f"Numerical data for '{name}' is None.")
            return None
        return val  # Keep other types (like floats, ints, bools)

    logger.debug("Converting numerical data...")
    data["opt_vars_vector"] = to_np(nlp_output.get("x"), name="nlp_output['x']")
    data["constraint_values"] = to_np(
        constraints, name="constraints"
    )  # Should already be numpy
    data["constraint_lbg"] = to_np(nlp_inputs.get("lbg"), name="nlp_inputs['lbg']")
    data["constraint_ubg"] = to_np(nlp_inputs.get("ubg"), name="nlp_inputs['ubg']")
    data["active_constraint_mask"] = to_np(
        active_constraints, name="active_constraints"
    )  # Should already be bool numpy
    data["constraint_multipliers"] = to_np(
        nlp_output.get("lam_g"), name="nlp_output['lam_g']"
    )

    # Sensitivities (handle potential None)
    if sensitivities_result:
        logger.debug("Processing sensitivity results...")
        data["gradient"] = to_np(
            sensitivities_result.get("g"), name="sensitivities_result['g']"
        )
        data["jacobian_constraints"] = to_np(
            sensitivities_result.get("J"), name="sensitivities_result['J']"
        )  # Keep as 2D numpy
        data["hessian"] = to_np(
            sensitivities_result.get("H"), name="sensitivities_result['H']"
        )  # Keep as 2D numpy
        # Ensure sensitivities_result['x'] is also converted if used
        data["sens_opt_vars"] = to_np(
            sensitivities_result.get("x"), name="sensitivities_result['x']"
        )

    # Debug Sensitivities (handle potential None)
    if debug_sensitivities:
        logger.debug("Processing debug sensitivity results...")
        data["jacobian_all"] = to_np(
            debug_sensitivities.get("jacobian_all"),
            name="debug_sensitivities['jacobian_all']",
        )  # Keep as 2D numpy
        data["original_hessian"] = to_np(
            debug_sensitivities.get("H_original"),
            name="debug_sensitivities['H_original']",
        )  # Keep as 2D numpy

    # Filter out None values from data before returning
    final_data = {k: v for k, v in data.items() if v is not None}
    logger.info(
        f"Data preparation complete. Returning {len(final_data)} items for visualization."
    )
    logger.debug(f"Prepared data keys: {list(final_data.keys())}")

    # Sanity check dimensions (optional but helpful)
    if "opt_vars_vector" in final_data and len(final_data["opt_vars_vector"]) != len(
        opt_var_labels
    ):
        logger.warning(
            f"Mismatch! opt_vars_vector length ({len(final_data['opt_vars_vector'])}) != opt_var_labels length ({len(opt_var_labels)})"
        )
    if "constraint_multipliers" in final_data and len(
        final_data["constraint_multipliers"]
    ) != len(all_constraint_labels):
        logger.warning(
            f"Mismatch! constraint_multipliers length ({len(final_data['constraint_multipliers'])}) != all_constraint_labels length ({len(all_constraint_labels)})"
        )
    if "jacobian_constraints" in final_data and data["jacobian_constraints"].shape[
        1
    ] != len(opt_var_labels):
        logger.warning(
            f"Mismatch! jacobian_constraints columns ({data['jacobian_constraints'].shape[1]}) != opt_var_labels length ({len(opt_var_labels)})"
        )
    # Add more checks as needed

    return final_data
