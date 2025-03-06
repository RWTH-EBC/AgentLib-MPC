# constraints_panel.py
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import pandas as pd


def create_constraints_panel():
    """Creates the layout for the constraints panel."""

    # --- Dummy Data ---
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

    # --- Layout ---
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Constraints"),
                            html.Div(
                                [
                                    dbc.Label("Filter by Type:"),
                                    dcc.Dropdown(
                                        id="constraint-type-filter",
                                        options=[
                                            {"label": t, "value": t}
                                            for t in dummy_constraints["Type"].unique()
                                        ],
                                        value=None,  # Initially, show all
                                        clearable=True,
                                        placeholder="Select constraint type...",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    dbc.Label("Filter by Time Step:"),
                                    dcc.Dropdown(
                                        id="constraint-time-filter",
                                        options=[
                                            {"label": str(t), "value": t}
                                            for t in dummy_constraints[
                                                "Time Step"
                                            ].unique()
                                        ],
                                        value=None,
                                        clearable=True,
                                        placeholder="Select time step...",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    dbc.Label("Sort by Violation:"),
                                    dcc.Dropdown(
                                        id="constraint-sort",
                                        options=[
                                            {
                                                "label": "Ascending",
                                                "value": "ascending",
                                            },
                                            {
                                                "label": "Descending",
                                                "value": "descending",
                                            },
                                        ],
                                        value=None,
                                        clearable=True,
                                        placeholder="Sort by violation...",
                                    ),
                                ]
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dash_table.DataTable(
                                id="constraint-table",
                                columns=[
                                    {
                                        "name": i,
                                        "id": i,
                                        "deletable": False,
                                        "selectable": False,
                                    }
                                    if i != "Variables"
                                    else {  # Format 'Variables' column differently
                                        "name": i,
                                        "id": i,
                                        "deletable": False,
                                        "selectable": False,
                                        "presentation": "markdown",
                                    }
                                    for i in dummy_constraints.columns
                                ],
                                data=dummy_constraints.to_dict("records"),
                                editable=False,
                                style_data_conditional=[
                                    {
                                        "if": {
                                            "filter_query": '{Status} = "Violated"',
                                            "column_id": "Status",
                                        },
                                        "backgroundColor": "tomato",
                                        "color": "white",
                                    },
                                    {
                                        "if": {
                                            "filter_query": '{Status} = "Satisfied"',
                                            "column_id": "Status",
                                        },
                                        "backgroundColor": "green",
                                        "color": "white",
                                    },
                                ],
                                sort_action="native",
                                filter_action="native",  # Enable filtering
                                page_action="native",
                                page_current=0,
                                page_size=10,
                                markdown_options={
                                    "html": True
                                },  # Allow HTML in markdown
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ]
    )
