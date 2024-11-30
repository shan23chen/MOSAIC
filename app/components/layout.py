import dash
from dash import Dash, html
from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Any
import logging
import json
from dash.dependencies import Input, Output, State
from app.utils.data_loader import create_path_key, get_model_path, parse_path_key
from app.components.model_info import create_model_info_banner
from app.components.model_selector import create_model_selector
from app.components.performance import create_model_performance_section
from app.components.feature_importance import create_feature_importance_section
from app.components.tree_viz import create_tree_visualization
from app.components.compare_models_plot import create_compare_models_tab
import logging

logging.basicConfig(level=logging.DEBUG)


def create_layout(options: Dict[str, Any]) -> html.Div:
    """Create main dashboard layout with tabs for different views."""
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    html.H1(
                        "Model Analysis Dashboard",
                        className="text-primary text-center mb-4",
                    )
                ),
                className="mb-3 bg-white",
            ),
            dcc.Tabs(
                id="tabs",
                value="tab-model-analysis",  # Default tab
                children=[
                    dcc.Tab(
                        label="Model Analysis",
                        value="tab-model-analysis",
                        children=create_model_analysis_tab(options),
                    ),
                    dcc.Tab(
                        label="Compare Models",
                        value="tab-compare-models",
                        children=create_compare_models_tab(options),
                    ),
                ],
            ),
        ],
        style={"backgroundColor": "#FAF9F6"},
    )


def create_model_analysis_tab(options: Dict[str, Any]) -> html.Div:
    """Create the layout for the Model Analysis tab."""
    return html.Div(
        [
            create_model_selector(options),
            html.Div(id="dashboard-content"),
            dcc.Store(id="current-model-data"),
        ]
    )


def register_callbacks(app, options):
    """Register callbacks for the dashboard components."""

    @app.callback(
        [Output("dataset-selector", "options"), Output("dataset-selector", "value")],
        [Input("model-selector", "value")],
    )
    def update_dataset_options(selected_model):
        if not selected_model:
            return [], None
        datasets = {
            opt["dataset"]
            for opt in options["options"]
            if opt["model"] == selected_model
        }
        sorted_datasets = sorted(datasets)
        options_list = [
            {"label": dataset, "value": dataset} for dataset in sorted_datasets
        ]
        default_value = options_list[0]["value"] if options_list else None
        return options_list, default_value

    @app.callback(
        [Output("layer-selector", "options"), Output("layer-selector", "value")],
        [Input("model-selector", "value"), Input("dataset-selector", "value")],
    )
    def update_layer_options(selected_model, selected_dataset):
        if not selected_model or not selected_dataset:
            return [], None
        layers = {
            opt["layer"]
            for opt in options["options"]
            if opt["model"] == selected_model and opt["dataset"] == selected_dataset
        }
        sorted_layers = sorted(layers)
        options_list = [
            {"label": str(layer), "value": str(layer)} for layer in sorted_layers
        ]
        default_value = options_list[0]["value"] if options_list else None
        return options_list, default_value

    @app.callback(
        [Output("width-selector", "options"), Output("width-selector", "value")],
        [
            Input("model-selector", "value"),
            Input("dataset-selector", "value"),
            Input("layer-selector", "value"),
        ],
    )
    def update_width_options(selected_model, selected_dataset, selected_layer):
        if not selected_model or not selected_dataset or not selected_layer:
            return [], None
        widths = {
            opt["width"]
            for opt in options["options"]
            if opt["model"] == selected_model
            and opt["dataset"] == selected_dataset
            and str(opt["layer"]) == selected_layer
        }
        sorted_widths = sorted(widths)
        options_list = [{"label": width, "value": width} for width in sorted_widths]
        default_value = options_list[0]["value"] if options_list else None
        return options_list, default_value

    @app.callback(
        [Output("top-n-selector", "options"), Output("top-n-selector", "value")],
        [
            Input("model-selector", "value"),
            Input("dataset-selector", "value"),
            Input("layer-selector", "value"),
            Input("width-selector", "value"),
        ],
    )
    def update_top_n_options(
        selected_model, selected_dataset, selected_layer, selected_width
    ):
        if not all([selected_model, selected_dataset, selected_layer, selected_width]):
            return [], None
        top_ns = {
            opt["top_n"]
            for opt in options["options"]
            if opt["model"] == selected_model
            and opt["dataset"] == selected_dataset
            and str(opt["layer"]) == selected_layer
            and opt["width"] == selected_width
        }
        sorted_top_ns = sorted(top_ns)
        options_list = [
            {"label": str(top_n), "value": str(top_n)} for top_n in sorted_top_ns
        ]
        default_value = options_list[0]["value"] if options_list else None
        return options_list, default_value

    @app.callback(
        [
            Output("config-name-selector", "options"),
            Output("config-name-selector", "value"),
        ],
        [
            Input("model-selector", "value"),
            Input("dataset-selector", "value"),
            Input("layer-selector", "value"),
            Input("width-selector", "value"),
            Input("top-n-selector", "value"),
        ],
    )
    def update_config_name_options(
        selected_model, selected_dataset, selected_layer, selected_width, selected_top_n
    ):
        if not all(
            [
                selected_model,
                selected_dataset,
                selected_layer,
                selected_width,
                selected_top_n,
            ]
        ):
            return [], None
        config_names = {
            opt["config_name"]
            for opt in options["options"]
            if opt["model"] == selected_model
            and opt["dataset"] == selected_dataset
            and str(opt["layer"]) == selected_layer
            and opt["width"] == selected_width
            and str(opt["top_n"]) == selected_top_n
        }
        sorted_config_names = sorted(config_names, key=lambda x: (x is None, x))
        options_list = [
            {"label": config_name or "None", "value": config_name or ""}
            for config_name in sorted_config_names
        ]
        default_value = options_list[0]["value"] if options_list else None
        return options_list, default_value

    @app.callback(
        [Output("binarise-selector", "options"), Output("binarise-selector", "value")],
        [
            Input("model-selector", "value"),
            Input("dataset-selector", "value"),
            Input("layer-selector", "value"),
            Input("width-selector", "value"),
            Input("top-n-selector", "value"),
            Input("config-name-selector", "value"),
        ],
    )
    def update_binarise_options(
        selected_model,
        selected_dataset,
        selected_layer,
        selected_width,
        selected_top_n,
        selected_config_name,
    ):
        if not all(
            [
                selected_model,
                selected_dataset,
                selected_layer,
                selected_width,
                selected_top_n,
                selected_config_name is not None,  # Allow empty string
            ]
        ):
            logging.debug("Missing inputs for binarise options, returning empty list.")
            return [], None

        logging.debug(f"Selected model: {selected_model}")
        logging.debug(f"Selected dataset: {selected_dataset}")
        logging.debug(f"Selected layer: {selected_layer}")
        logging.debug(f"Selected width: {selected_width}")
        logging.debug(f"Selected top_n: {selected_top_n}")
        logging.debug(f"Selected config_name: '{selected_config_name}'")

        matching_options = [
            opt
            for opt in options["options"]
            if opt["model"] == selected_model
            and opt["dataset"] == selected_dataset
            and str(opt["layer"]) == selected_layer
            and opt["width"] == selected_width
            and str(opt["top_n"]) == selected_top_n
            and (opt["config_name"] or "") == (selected_config_name or "")
        ]

        logging.debug(f"Found {len(matching_options)} matching options for binarise.")

        binarise_values = {opt["binarize_value"] for opt in matching_options}

        logging.debug(f"Binarise values collected: {binarise_values}")

        # Custom sorting to handle None values
        def none_sorter(x):
            if x is None:
                return (1, 0)
            else:
                return (0, x)

        sorted_binarise_values = sorted(binarise_values, key=none_sorter)
        options_list = []
        for binarise in sorted_binarise_values:
            if binarise == 1.0:
                label = "True"
                value = "1.0"
            elif binarise == 0.0:
                label = "False"
                value = "0.0"
            elif binarise is None:
                label = "None"
                value = "None"
            else:
                label = str(binarise)
                value = str(binarise)
            options_list.append({"label": label, "value": value})
        default_value = options_list[0]["value"] if options_list else None
        return options_list, default_value

    @app.callback(
        [Output("hidden-selector", "options"), Output("hidden-selector", "value")],
        [
            Input("model-selector", "value"),
            Input("dataset-selector", "value"),
            Input("layer-selector", "value"),
            Input("width-selector", "value"),
            Input("top-n-selector", "value"),
            Input("config-name-selector", "value"),
            Input("binarise-selector", "value"),
        ],
    )
    def update_hidden_options(
        selected_model,
        selected_dataset,
        selected_layer,
        selected_width,
        selected_top_n,
        selected_config_name,
        selected_binarise,
    ):
        if not all(
            [
                selected_model,
                selected_dataset,
                selected_layer,
                selected_width,
                selected_top_n,
                selected_config_name is not None,  # Allow empty string
                selected_binarise is not None,
            ]
        ):
            logging.debug("Missing inputs for hidden options, returning empty list.")
            return [], None

        logging.debug(f"Selected binarise: '{selected_binarise}'")

        # Convert selected_binarise to the appropriate type
        if selected_binarise == "None":
            selected_binarise_value = None
        elif selected_binarise == "1.0":
            selected_binarise_value = 1.0
        elif selected_binarise == "0.0":
            selected_binarise_value = 0.0
        else:
            selected_binarise_value = float(selected_binarise)

        logging.debug(f"Converted selected binarise to: {selected_binarise_value}")

        matching_options = [
            opt
            for opt in options["options"]
            if opt["model"] == selected_model
            and opt["dataset"] == selected_dataset
            and str(opt["layer"]) == selected_layer
            and opt["width"] == selected_width
            and str(opt["top_n"]) == selected_top_n
            and (opt["config_name"] or "") == (selected_config_name or "")
            and opt["binarize_value"] == selected_binarise_value
        ]

        logging.debug(f"Found {len(matching_options)} matching options for hidden.")

        hidden_values = {opt["hidden"] for opt in matching_options}

        logging.debug(f"Hidden values collected: {hidden_values}")

        # Custom sorting to handle None values
        def none_sorter(x):
            if x is None:
                return (1, "")
            else:
                return (0, x)

        sorted_hidden_values = sorted(hidden_values, key=none_sorter)
        options_list = []
        for hidden in sorted_hidden_values:
            if hidden is True:
                label = "True"
                value = "True"
            elif hidden is False:
                label = "False"
                value = "False"
            elif hidden is None:
                label = "None"
                value = "None"
            else:
                label = str(hidden)
                value = str(hidden)
            options_list.append({"label": label, "value": value})
        default_value = options_list[0]["value"] if options_list else None
        return options_list, default_value

    @app.callback(
        Output("current-model-data", "data"),
        Input("load-data-button", "n_clicks"),
        State("model-selector", "value"),
        State("dataset-selector", "value"),
        State("layer-selector", "value"),
        State("width-selector", "value"),
        State("top-n-selector", "value"),
        State("config-name-selector", "value"),
        State("binarise-selector", "value"),
        State("hidden-selector", "value"),
    )
    def load_data(
        n_clicks,
        selected_model,
        selected_dataset,
        selected_layer,
        selected_width,
        selected_top_n,
        selected_config_name,
        selected_binarise,
        selected_hidden,
    ):
        if n_clicks is None:
            return None

        logging.debug("Loading data with the following selections:")
        logging.debug(f"Model: {selected_model}")
        logging.debug(f"Dataset: {selected_dataset}")
        logging.debug(f"Layer: {selected_layer}")
        logging.debug(f"Width: {selected_width}")
        logging.debug(f"Top N: {selected_top_n}")
        logging.debug(f"Config Name: '{selected_config_name}'")
        logging.debug(f"Binarise: {selected_binarise}")
        logging.debug(f"Hidden: {selected_hidden}")

        # Convert selected values to appropriate types
        selected_layer = int(selected_layer)
        selected_top_n = int(selected_top_n)
        selected_binarise_value = (
            None if selected_binarise == "None" else float(selected_binarise)
        )
        selected_hidden_value = (
            None if selected_hidden == "None" else selected_hidden == "True"
        )
        selected_config_name = selected_config_name or ""

        logging.debug(f"Converted binarise value: {selected_binarise_value}")
        logging.debug(f"Converted hidden value: {selected_hidden_value}")

        # Find the matching option
        matching_options = [
            opt
            for opt in options["options"]
            if opt["model"] == selected_model
            and opt["dataset"] == selected_dataset
            and opt["layer"] == selected_layer
            and opt["width"] == selected_width
            and opt["top_n"] == selected_top_n
            and (opt["config_name"] or "") == selected_config_name
            and opt["binarize_value"] == selected_binarise_value
            and opt["hidden"] == selected_hidden_value
        ]

        logging.debug(
            f"Found {len(matching_options)} matching options for data loading."
        )

        if not matching_options:
            logging.error("No matching data found for the selected combination.")
            return None

        file_path = matching_options[0]["filepath"]
        logging.debug(f"Attempting to load data from file: {file_path}")

        # Load the JSON data
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            logging.debug(f"Data loaded successfully from {file_path}")
            # Return data as JSON string to store in dcc.Store
            return json.dumps(data)
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            return None

    @app.callback(
        Output("dashboard-content", "children"),
        Input("current-model-data", "data"),
    )
    def update_dashboard_content(data_json):
        if data_json is None:
            logging.debug("No data received in update_dashboard_content")
            return html.Div("No data loaded")
        data = json.loads(data_json)
        logging.debug(f"Data received in update_dashboard_content: {data}")
        return create_dashboard_content(data)

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
                                        data["models"]["linearProbe"][
                                            "feature_analysis"
                                        ],
                                        data["models"]["decisionTree"][
                                            "feature_analysis"
                                        ],
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
