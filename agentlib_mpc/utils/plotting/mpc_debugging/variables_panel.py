# variables_panel.py
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import plotly.express as px
import pandas as pd


def create_variables_panel():
    """Creates the layout for the variables panel."""

    # --- Dummy Data ---
    dummy_variables = pd.DataFrame(
        {
            "Variable": ["x1", "x2", "u1", "u2", "T1", "T2"],
            "Initial Value": [1.0, 100.0, 0.1, 2.0, 25.0, 2000.0],
            "Final Value": [1.2, 95.0, 0.15, 1.8, 26.5, 1950.0],
            "Units": ["m", "m", "m/s", "m/s", "°C", "°C"],
            "Type": ["State", "State", "Control", "Control", "State", "State"],
        }
    )
    dummy_variables_initial_hist = px.histogram(
        dummy_variables,
        x="Initial Value",
        log_x=True,
        title="Initial Value Distribution (Dummy)",
    ).update_layout(xaxis_title="Initial Value (Log Scale)", yaxis_title="Count")
    dummy_variables_final_hist = px.histogram(
        dummy_variables,
        x="Final Value",
        log_x=True,
        title="Final Value Distribution (Dummy)",
    ).update_layout(xaxis_title="Final Value (Log Scale)", yaxis_title="Count")

    # --- Layout ---
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Variables"),
                            dcc.Dropdown(
                                id="variable-type-dropdown",
                                options=[
                                    {"label": t, "value": t}
                                    for t in dummy_variables["Type"].unique()
                                ],
                                value=dummy_variables["Type"].unique()[
                                    0
                                ],  # Set an initial value
                                clearable=False,
                            ),
                            dash_table.DataTable(
                                id="variable-table",
                                columns=[
                                    {
                                        "name": i,
                                        "id": i,
                                        "deletable": False,
                                        "selectable": False,
                                    }
                                    for i in dummy_variables.columns
                                ],
                                data=dummy_variables.to_dict("records"),
                                editable=False,
                                sort_action="native",
                                sort_mode="multi",
                                page_action="native",
                                page_current=0,
                                page_size=10,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="initial-value-histogram",
                                figure=dummy_variables_initial_hist,
                            ),
                            dcc.Graph(
                                id="final-value-histogram",
                                figure=dummy_variables_final_hist,
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )
