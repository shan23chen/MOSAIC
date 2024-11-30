from dash import html, dcc
from typing import Dict, List, Any
import dash_bootstrap_components as dbc


def create_model_selector(options: Dict[str, Any]) -> dbc.Card:
    """Create selector components for model, dataset, layer, width, and additional filters."""
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Model", className="fw-bold"),
                                dcc.Dropdown(
                                    id="model-selector",
                                    options=[
                                        {"label": model, "value": model}
                                        for model in options["models"]
                                    ],
                                    placeholder="Select a model",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Dataset", className="fw-bold"),
                                dcc.Dropdown(
                                    id="dataset-selector",
                                    placeholder="Select a dataset",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Layer", className="fw-bold"),
                                dcc.Dropdown(
                                    id="layer-selector",
                                    placeholder="Select a layer",
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Width", className="fw-bold"),
                                dcc.Dropdown(
                                    id="width-selector",
                                    placeholder="Select a width",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Top N", className="fw-bold"),
                                dcc.Dropdown(
                                    id="top-n-selector",
                                    placeholder="Select Top N",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Config Name", className="fw-bold"),
                                dcc.Dropdown(
                                    id="config-name-selector",
                                    placeholder="Select a config name",
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Binarise Value", className="fw-bold"),
                                dcc.Dropdown(
                                    id="binarise-selector",
                                    placeholder="Select binarise value",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Hidden", className="fw-bold"),
                                dcc.Dropdown(
                                    id="hidden-selector",
                                    placeholder="Select hidden value",
                                ),
                            ],
                            width=6,
                        ),
                    ]
                ),
                html.Br(),
                dbc.Button("Load Data", id="load-data-button", color="primary"),
                html.Div(id="loaded-data", style={"display": "none"}),
            ]
        ),
        className="mt-3",
    )
