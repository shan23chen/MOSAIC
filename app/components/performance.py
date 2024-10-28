from dash import html
import dash_bootstrap_components as dbc


def create_model_performance_section(linear_metrics, tree_metrics):
    """Create an enhanced model performance section with background."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H3("Model Performance", className="mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            "Training Score",
                                            className="text-muted mb-2",
                                        ),
                                        html.Div(
                                            f"{linear_metrics['accuracy']*100:.1f}%",
                                            className="display-4 text-success",
                                        ),
                                        dbc.Progress(
                                            value=linear_metrics["accuracy"] * 100,
                                            color="success",
                                            className="mt-2",
                                            style={"height": "8px"},
                                        ),
                                    ]
                                ),
                                className="h-100",
                                style={
                                    "backgroundColor": "#E8F5E9"
                                },  # Light green background
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            "Test Score", className="text-muted mb-2"
                                        ),
                                        html.Div(
                                            f"{tree_metrics['accuracy']*100:.1f}%",
                                            className="display-4 text-primary",
                                        ),
                                        dbc.Progress(
                                            value=tree_metrics["accuracy"] * 100,
                                            color="primary",
                                            className="mt-2",
                                            style={"height": "8px"},
                                        ),
                                    ]
                                ),
                                className="h-100",
                                style={
                                    "backgroundColor": "#E3F2FD"
                                },  # Light blue background
                            ),
                            width=6,
                        ),
                    ]
                ),
            ]
        ),
        className="mb-3 bg-white",
    )
