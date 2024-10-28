from dash import html, dcc
from typing import Dict, List


def create_model_selector(models: List[Dict[str, str]]) -> html.Div:
    """Create model selection dropdown."""
    return html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "Select Model",
                        style={
                            "fontWeight": "bold",
                            "marginBottom": "8px",
                            "display": "block",
                        },
                    ),
                    dcc.Dropdown(
                        id="model-selector",
                        options=[
                            {"label": model["display_name"], "value": model["path"]}
                            for model in models
                        ],
                        value=models[0]["path"] if models else None,
                        style={"width": "100%"},
                    ),
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "4px",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.12)",
                    "marginBottom": "20px",
                },
            )
        ]
    )
