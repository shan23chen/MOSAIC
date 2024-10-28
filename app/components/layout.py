import json
import dash
from dash import Dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from typing import List, Dict, Any
import dash_bootstrap_components as dbc
from dash import html
from typing import List, Dict, Any
from dash.dependencies import Input, Output, State, MATCH, ALL
from app.components.model_selector import create_model_selector
from app.components.performance import create_model_performance_section
from app.components.tree_viz import create_tree_visualization
from app.styles import COLORS, STYLES
from app.utils.data_loader import load_dashboard_data  # Updated import
from app.components.feature_importance import create_feature_importance_section
from app.components.model_info import create_model_info_banner


def register_callbacks(app: Dash) -> None:
    """Register all dashboard callbacks."""

    # Model data callback
    @app.callback(
        Output("current-model-data", "data"), Input("model-selector", "value")
    )
    def update_model_data(selected_path):
        if not selected_path:
            return {}
        try:
            return load_dashboard_data(selected_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}

    # Dashboard content and model info callback
    @app.callback(
        [
            Output("dashboard-content", "children"),
            Output("model-info-banner", "children"),
        ],
        Input("current-model-data", "data"),
    )
    def update_dashboard_content(data):
        if not data:
            return html.Div("Please select a model to view analysis"), html.Div()
        return create_dashboard_content(data), create_model_info_banner(data)

    # Tree node toggle callback
    @app.callback(
        Output({"type": "node-children", "index": MATCH}, "style"),
        Input({"type": "node-toggle", "index": MATCH}, "n_clicks"),
        State({"type": "node-children", "index": MATCH}, "style"),
        prevent_initial_call=True,
    )
    def toggle_node(n_clicks, current_style):
        if n_clicks is None:
            return dash.no_update

        if current_style.get("display") == "none":
            return {"display": "block"}
        return {"display": "none"}


def create_info_badge(label: str, value: str) -> dbc.Badge:
    """Create a styled info badge component."""
    return dbc.Badge(
        [
            html.Span(label, className="text-muted small mr-1"),
            html.Span(value, className="font-weight-bold"),
        ],
        color="light",
        className="p-2 mr-2 mb-2",
    )


def create_layout(models: List[Dict[str, Any]]) -> html.Div:
    """Create main dashboard layout with improved styling."""
    return html.Div(
        [
            # Header and Model Selector
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H1(
                            "Model Analysis Dashboard",
                            className="text-primary text-center mb-4",
                        ),
                        create_model_selector(models),
                    ]
                ),
                className="mb-3 bg-white",
            ),
            # Model Info Banner
            html.Div(id="model-info-banner"),
            # Main content container
            html.Div(id="dashboard-content"),
            # Loading spinner
            dcc.Loading(
                id="loading", type="circle", children=html.Div(id="loading-output")
            ),
            # Store for current model data
            dcc.Store(id="current-model-data"),
        ],
        className="p-3",
        style={"backgroundColor": "#FAF9F6"},  # Cream background
    )


def create_dashboard_content(data: Dict[str, Any]) -> html.Div:
    """Create dashboard content with improved styling."""
    if not data:
        return html.Div("No data available")

    linear_metrics = data["models"]["linearProbe"]
    tree_metrics = data["models"]["decisionTree"]

    return html.Div(
        [
            # Model Performance Section
            create_model_performance_section(
                linear_metrics.get("performance", {}),
                tree_metrics.get("performance", {}),
            ),
            # Feature Importance Section
            dbc.Card(
                dbc.CardBody(
                    create_feature_importance_section(
                        {
                            "importance_scores": linear_metrics.get(
                                "feature_analysis", {}
                            ).get("importance_scores", []),
                            "top_features": linear_metrics.get(
                                "feature_analysis", {}
                            ).get("top_features", []),
                        },
                        {
                            "importance_scores": tree_metrics.get(
                                "feature_analysis", {}
                            ).get("importance_scores", []),
                            "top_features": tree_metrics.get(
                                "feature_analysis", {}
                            ).get("top_features", []),
                        },
                    )
                ),
                className="mb-3 bg-white",
            ),
            # Decision Tree Visualization
            dbc.Card(
                dbc.CardBody(create_tree_visualization(data)), className="mb-3 bg-white"
            ),
        ]
    )
