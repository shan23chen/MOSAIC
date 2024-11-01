from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Any
import logging
import json
from dash.dependencies import Input, Output, State
from app.utils.data_loader import (
    # load_model_data,
    # get_model_path,
    # get_top_features,
    create_path_key,
)
from app.components.model_info import create_model_info_banner
from app.components.model_selector import create_model_selector
from app.components.performance import create_model_performance_section
from app.components.feature_importance import create_feature_importance_section
from app.components.tree_viz import create_tree_visualization


def create_layout(options: Dict[str, Any]) -> html.Div:
    """Create main dashboard layout with options banner."""
    return html.Div(
        [
            # Header
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H1(
                            "Model Analysis Dashboard",
                            className="text-primary text-center mb-4",
                        ),
                        # Options selector
                        create_model_selector(options),
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
            # Store components
            dcc.Store(id="dashboard-options", data=options),
            dcc.Store(id="current-model-data"),
        ],
        className="p-3",
        style={"backgroundColor": "#FAF9F6"},
    )


def register_callbacks(app):
    """Register callbacks for the dashboard components."""

    @app.callback(
        Output("dataset-selector", "options"),
        Output("dataset-selector", "value"),
        Input("model-selector", "value"),
        State("dashboard-options", "data"),
    )
    def update_dataset_options(selected_model: str, options: Dict[str, Any]):
        if not selected_model or not options:
            return [], None

        datasets = options["datasets"].get(selected_model, [])
        logging.info(f"Available datasets for {selected_model}: {datasets}")
        return [{"label": d, "value": d} for d in datasets], (
            datasets[0] if datasets else None
        )

    @app.callback(
        Output("layer-selector", "options"),
        Output("layer-selector", "value"),
        Input("model-selector", "value"),
        State("dashboard-options", "data"),
    )
    def update_layer_options(selected_model: str, options: Dict[str, Any]):
        if not selected_model or not options:
            return [], None

        layers = options["layers"].get(selected_model, [])
        logging.info(f"Available layers for {selected_model}: {layers}")
        return [{"label": f"Layer {l}", "value": l} for l in layers], (
            layers[0] if layers else None
        )

    @app.callback(
        [
            Output("current-model-data", "data"),
            Output("dashboard-content", "children"),
            Output("model-info-banner", "children"),
        ],
        [
            Input("model-selector", "value"),
            Input("dataset-selector", "value"),
            Input("result-type-selector", "value"),
            Input("layer-selector", "value"),
        ],
        State("dashboard-options", "data"),
    )
    def update_dashboard_content(model, dataset, result_type, layer, options):
        if not all([model, dataset, result_type, layer, options]):
            return {}, html.Div("Please select all options"), html.Div()

        try:
            # Create the path key and get the file path
            path_key = create_path_key(model, dataset, layer, result_type)
            file_path = options["paths"].get(path_key)

            if not file_path:
                error_msg = f"No data found for combination: {path_key}"
                logging.error(error_msg)
                return {}, html.Div(error_msg), html.Div()

            # Load model data from the file path
            try:
                with open(file_path, "r") as f:
                    model_data = json.load(f)
            except Exception as e:
                error_msg = f"Error loading data from {file_path}: {str(e)}"
                logging.error(error_msg)
                return {}, html.Div(error_msg), html.Div()

            # Create content components
            try:
                dashboard_content = create_dashboard_content(model_data)
                info_banner = create_model_info_banner(model_data)
            except Exception as e:
                error_msg = f"Error creating dashboard components: {str(e)}"
                logging.error(error_msg)
                return model_data, html.Div(error_msg), html.Div()

            logging.info(f"Successfully loaded and rendered data for {path_key}")
            return model_data, dashboard_content, info_banner

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(error_msg)
            return {}, html.Div(error_msg), html.Div()


def create_dashboard_content(data: Dict[str, Any]) -> html.Div:
    """Create dashboard content with all visualizations."""
    if not data:
        return html.Div("No data available")
    # Extract metrics with more detailed error handling
    linear_metrics = data["models"]["linearProbe"]
    tree_metrics = data["models"]["decisionTree"]

    try:
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
                        dbc.Card(
                            dbc.CardBody(
                                create_feature_importance_section(
                                    data["models"]["linearProbe"]["feature_analysis"],
                                    data["models"]["decisionTree"]["feature_analysis"],
                                )
                            ),
                            className="mb-3 bg-white",
                        ),
                    ),
                    className="mb-3 bg-white",
                ),
                # Decision Tree Visualization
                dbc.Card(
                    dbc.CardBody(create_tree_visualization(data)),
                    className="mb-3 bg-white",
                ),
            ]
        )
    except Exception as e:
        error_msg = f"Error creating dashboard content: {str(e)}"
        logging.error(error_msg)
        return html.Div(error_msg)
