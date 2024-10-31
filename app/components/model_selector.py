from dash import html, dcc
from typing import Dict, List, Any
import dash_bootstrap_components as dbc


def create_model_selector(options: Dict[str, Any]) -> dbc.Card:
    """Create selector components for model, dataset, split, and layer."""
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
                                    value=(
                                        options["models"][0]
                                        if options["models"]
                                        else None
                                    ),
                                    clearable=False,
                                    className="mb-2",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Dataset", className="fw-bold"),
                                dcc.Dropdown(
                                    id="dataset-selector",
                                    clearable=False,
                                    className="mb-2",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Split", className="fw-bold"),
                                dcc.Dropdown(
                                    id="split-selector",
                                    clearable=False,
                                    className="mb-2",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Layer", className="fw-bold"),
                                dcc.Dropdown(
                                    id="layer-selector",
                                    clearable=False,
                                    className="mb-2",
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="g-2",
                ),
            ]
        ),
        className="mb-3 bg-white",
    )
