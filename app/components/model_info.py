import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output

from typing import Dict, Any


# Updated Model Info Banner Component
def create_model_info_banner(data: Dict[str, Any]) -> html.Div:
    """Create a compact model info banner with additional fields."""
    if not data or not data.get("metadata", {}).get("model"):
        return html.Div()

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(
                                    data["metadata"]["model"]["name"],
                                    className="text-primary font-weight-bold",
                                ),
                                html.Span(" | ", className="text-muted mx-2"),
                                html.Span(
                                    f"Layer {data['metadata']['model']['layer']}",
                                    className="text-muted",
                                ),
                                html.Span(" | ", className="text-muted mx-2"),
                                html.Span(
                                    data["metadata"]["model"]["type"],
                                    className="text-muted",
                                ),
                                html.Span(" | ", className="text-muted mx-2"),
                                html.Span(
                                    f"Top N: {data['metadata']['args'].get('top_n', 'N/A')}",
                                    className="text-muted",
                                ),
                                html.Span(" | ", className="text-muted mx-2"),
                                html.Span(
                                    f"Binarize: {data['metadata']['args'].get('binarize_value', 'N/A')}",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        className="d-flex align-items-center",
                    )
                ]
            ),
            className="py-2",  # Reduced padding for more compact appearance
        ),
        className="mt-3 border-0 bg-light",
    )
