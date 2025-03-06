# overview_panel.py
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_overview_panel():
    """Creates the layout for the overview panel."""
    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader(html.H4("Solver Overview", className="card-title")),
                    dbc.CardBody(
                        [
                            html.Div(
                                id="solver-status",
                                children=[
                                    html.B("Solver Status: "),
                                    html.Span("Optimal (Dummy)"),  # Placeholder
                                ],
                                style={"fontSize": "1.2em"},
                            ),
                            html.Div(
                                id="objective-value",
                                children=[
                                    html.B("Objective Value: "),
                                    html.Span("1234.56 (Dummy)"),  # Placeholder
                                ],
                                style={"fontSize": "1.2em"},
                            ),
                            html.Div(
                                id="solve-time",
                                children=[
                                    html.B("Solve Time: "),
                                    html.Span("0.123 seconds (Dummy)"),  # Placeholder
                                ],
                                style={"fontSize": "1.2em"},
                            ),
                            html.Div(
                                id="num-iterations",
                                children=[
                                    html.B("Iterations: "),
                                    html.Span("10 (Dummy)"),  # Placeholder
                                ],
                                style={"fontSize": "1.2em"},
                            ),
                            html.Div(
                                id="violated-constraints",
                                children=[
                                    html.B("Violated Constraints: "),
                                    html.Span("0 (Dummy)"),  # Placeholder
                                ],
                                style={"fontSize": "1.2em"},
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )
