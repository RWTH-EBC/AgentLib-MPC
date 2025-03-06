# app.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from overview_panel import create_overview_panel
from variables_panel import create_variables_panel
from constraints_panel import create_constraints_panel
from dash.dependencies import Input, Output
import pandas as pd


def create_app():
    """Creates the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(
        [
            html.H1("CasADi NMPC Debugger"),
            dbc.Tabs(
                [
                    dbc.Tab(create_overview_panel(), label="Overview"),
                    dbc.Tab(create_variables_panel(), label="Variables"),
                    dbc.Tab(create_constraints_panel(), label="Constraints"),
                ]
            ),
        ],
        fluid=True,
    )

    # --- Callbacks ---
    @app.callback(
        Output("variable-table", "data"), Input("variable-type-dropdown", "value")
    )
    def update_variable_table(selected_type):
        # --- Dummy Data --- (Same as in variables_panel.py)
        dummy_variables = pd.DataFrame(
            {
                "Variable": ["x1", "x2", "u1", "u2", "T1", "T2"],
                "Initial Value": [1.0, 100.0, 0.1, 2.0, 25.0, 2000.0],
                "Final Value": [1.2, 95.0, 0.15, 1.8, 26.5, 1950.0],
                "Units": ["m", "m", "m/s", "m/s", "°C", "°C"],
                "Type": ["State", "State", "Control", "Control", "State", "State"],
            }
        )
        filtered_df = dummy_variables[dummy_variables["Type"] == selected_type]
        return filtered_df.to_dict("records")

    @app.callback(
        Output("constraint-table", "data"),
        [
            Input("constraint-type-filter", "value"),
            Input("constraint-time-filter", "value"),
            Input("constraint-sort", "value"),
        ],
    )
    def update_constraint_table(selected_type, selected_time, sort_order):
        # --- Dummy Data --- (Same as in constraints_panel.py)
        dummy_constraints = pd.DataFrame(
            {
                "Constraint": ["c1", "c2", "c3", "c4"],
                "Time Step": [0, 0, 1, 1],
                "Function Value": [1.0, -0.5, 2.2, 0.1],
                "Lower Bound": [0.0, -1.0, 0.0, 0.0],
                "Upper Bound": [2.0, 0.0, 2.0, 1.0],
                "Violation": [0.0, 0.0, 0.2, 0.0],
                "Status": ["Satisfied", "Satisfied", "Violated", "Satisfied"],
                "Variables": [
                    ["x1", "u1"],
                    ["x2"],
                    ["x1", "T1"],
                    ["u2", "T2"],
                ],  # List of involved variables
                "Type": ["Equality", "Inequality", "Inequality", "Equality"],
            }
        )

        filtered_df = dummy_constraints.copy()

        if selected_type:
            filtered_df = filtered_df[filtered_df["Type"] == selected_type]
        if selected_time is not None:  # Handle the case where selected_time is 0
            filtered_df = filtered_df[filtered_df["Time Step"] == selected_time]
        if sort_order:
            filtered_df = filtered_df.sort_values(
                by="Violation", ascending=(sort_order == "ascending")
            )
        # Convert the 'Variables' column to strings for display in the table.
        filtered_df["Variables"] = filtered_df["Variables"].apply(
            lambda x: ", ".join(x)
        )
        return filtered_df.to_dict("records")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=True)
