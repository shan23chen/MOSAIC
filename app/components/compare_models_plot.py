from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import dash_table
import logging
from dash_table.Format import Format, Scheme
import dash


def create_compare_models_tab(options) -> html.Div:
    """Create the layout for the Compare Models tab with improved filter sections, a DataTable, and a graph."""
    # Extract unique values for filters
    unique_models = sorted(set(opt["model"] for opt in options["options"]))
    unique_datasets = sorted(set(opt["dataset"] for opt in options["options"]))
    unique_layers = sorted(set(str(opt["layer"]) for opt in options["options"]))
    unique_widths = sorted(set(opt["width"] for opt in options["options"]))
    unique_top_ns = sorted(set(str(opt["top_n"]) for opt in options["options"]))
    unique_config_names = sorted(
        set(opt["config_name"] or "None" for opt in options["options"])
    )
    unique_binarize_values = sorted(
        set(
            "None" if opt["binarize_value"] is None else str(opt["binarize_value"])
            for opt in options["options"]
        )
    )
    unique_hidden_values = sorted(
        set(
            "None" if opt["hidden"] is None else str(opt["hidden"])
            for opt in options["options"]
        )
    )
    unique_splits = sorted(set(opt["split"] for opt in options["options"]))

    # Define metric columns with formatting
    metric_columns = [
        {
            "name": "LP Mean CV Acc",
            "id": "linearProbe_mean_cv_accuracy",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "LP Test Acc",
            "id": "linearProbe_test_accuracy",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "LP Macro Prec",
            "id": "linearProbe_macro_precision",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "LP Macro Recall",
            "id": "linearProbe_macro_recall",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "LP Macro F1",
            "id": "linearProbe_macro_f1_score",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "DT Mean CV Acc",
            "id": "decisionTree_mean_cv_accuracy",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "DT Test Acc",
            "id": "decisionTree_test_accuracy",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "DT Macro Prec",
            "id": "decisionTree_macro_precision",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "DT Macro Recall",
            "id": "decisionTree_macro_recall",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "DT Macro F1",
            "id": "decisionTree_macro_f1_score",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
    ]

    # Function to create filter components
    def create_filter_component(label, id_suffix, unique_values):
        return dbc.Col(
            html.Div(
                [
                    html.Label(label),
                    dcc.Dropdown(
                        id=f"filter-{id_suffix}-dropdown",
                        options=[{"label": "Select All", "value": "All"}]
                        + [{"label": value, "value": value} for value in unique_values],
                        value=[unique_values[0]],  # default to first option selected
                        multi=True,
                        placeholder=f"Select {label}",
                    ),
                ]
            ),
            width=4,
        )

    return html.Div(
        [
            # Page Title
            dbc.Row(
                dbc.Col(
                    html.H3("Compare Models", className="text-center mb-4"),
                    width=12,
                )
            ),
            # Top Section: Collapsible panels for filtering parameters
            dbc.Row(
                [
                    dbc.Col(
                        html.H5("Filter Runs", className="fw-bold"),
                        width=12,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Filter by Model, Dataset, Layer, etc.",
                                                id="filter-collapse-button",
                                                className="mb-3",
                                                n_clicks=0,
                                            ),
                                            dbc.Collapse(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    create_filter_component(
                                                                        "Model",
                                                                        "model",
                                                                        unique_models,
                                                                    ),
                                                                    create_filter_component(
                                                                        "Dataset",
                                                                        "dataset",
                                                                        unique_datasets,
                                                                    ),
                                                                    create_filter_component(
                                                                        "Layer",
                                                                        "layer",
                                                                        unique_layers,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    create_filter_component(
                                                                        "Width",
                                                                        "width",
                                                                        unique_widths,
                                                                    ),
                                                                    create_filter_component(
                                                                        "Top N",
                                                                        "top-n",
                                                                        unique_top_ns,
                                                                    ),
                                                                    create_filter_component(
                                                                        "Config Name",
                                                                        "config-name",
                                                                        unique_config_names,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    create_filter_component(
                                                                        "Binarize Value",
                                                                        "binarize",
                                                                        unique_binarize_values,
                                                                    ),
                                                                    create_filter_component(
                                                                        "Hidden",
                                                                        "hidden",
                                                                        unique_hidden_values,
                                                                    ),
                                                                    create_filter_component(
                                                                        "Split",
                                                                        "split",
                                                                        unique_splits,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                        ]
                                                    ),
                                                    className="mb-3",
                                                ),
                                                id="filter-collapse",
                                                is_open=False,
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ),
                        id="run-filter-card",
                        is_open=True,
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
            # Middle Section: DataTable to display all possible runs
            dbc.Row(
                [
                    dbc.Col(
                        dash_table.DataTable(
                            id="runs-table",
                            columns=[
                                {"name": "Model", "id": "model"},
                                {"name": "Dataset", "id": "dataset"},
                                {"name": "Layer", "id": "layer"},
                                {"name": "Width", "id": "width"},
                                {"name": "Top N", "id": "top_n"},
                                {"name": "Config Name", "id": "config_name"},
                                {"name": "Binarize", "id": "binarize_value"},
                                {"name": "Hidden", "id": "hidden"},
                                {"name": "Split", "id": "split"},
                            ]
                            + metric_columns,
                            data=[],  # Data will be populated via callback
                            style_table={"overflowX": "auto"},
                            row_selectable="multi",
                            filter_action="native",
                            sort_action="native",  # Enable sorting
                            page_action="native",
                            page_size=10,
                            style_cell={"textAlign": "left", "minWidth": "80px"},
                            tooltip_duration=None,
                        ),
                        width=12,
                    ),
                ],
                className="mb-3",
            ),
            # Bottom Section: Graph to visualize selected comparisons
            dbc.Row(
                [
                    dbc.Col(
                        html.Label("Select Metric", className="fw-bold"),
                        width=12,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="compare-metric-selector",
                            options=[
                                {
                                    "label": "Linear Probe - Mean CV Accuracy",
                                    "value": "linearProbe_mean_cv_accuracy",
                                },
                                {
                                    "label": "Linear Probe - Test Accuracy",
                                    "value": "linearProbe_test_accuracy",
                                },
                                {
                                    "label": "Linear Probe - Macro Precision",
                                    "value": "linearProbe_macro_precision",
                                },
                                {
                                    "label": "Linear Probe - Macro Recall",
                                    "value": "linearProbe_macro_recall",
                                },
                                {
                                    "label": "Linear Probe - Macro F1 Score",
                                    "value": "linearProbe_macro_f1_score",
                                },
                                {
                                    "label": "Decision Tree - Mean CV Accuracy",
                                    "value": "decisionTree_mean_cv_accuracy",
                                },
                                {
                                    "label": "Decision Tree - Test Accuracy",
                                    "value": "decisionTree_test_accuracy",
                                },
                                {
                                    "label": "Decision Tree - Macro Precision",
                                    "value": "decisionTree_macro_precision",
                                },
                                {
                                    "label": "Decision Tree - Macro Recall",
                                    "value": "decisionTree_macro_recall",
                                },
                                {
                                    "label": "Decision Tree - Macro F1 Score",
                                    "value": "decisionTree_macro_f1_score",
                                },
                            ],
                            value="linearProbe_test_accuracy",  # Default metric
                            placeholder="Select a metric to compare",
                        ),
                        width=12,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="compare-models-bar-plot"),
                        width=12,
                    ),
                ],
            ),
        ]
    )


def register_compare_models_callbacks(app, options):
    """Register callbacks for the Compare Models tab."""
    # Extract unique values for filters
    unique_models = sorted(set(opt["model"] for opt in options["options"]))
    unique_datasets = sorted(set(opt["dataset"] for opt in options["options"]))
    unique_layers = sorted(set(str(opt["layer"]) for opt in options["options"]))
    unique_widths = sorted(set(opt["width"] for opt in options["options"]))
    unique_top_ns = sorted(set(str(opt["top_n"]) for opt in options["options"]))
    unique_config_names = sorted(
        set(opt["config_name"] or "None" for opt in options["options"])
    )
    unique_binarize_values = sorted(
        set(
            "None" if opt["binarize_value"] is None else str(opt["binarize_value"])
            for opt in options["options"]
        )
    )
    unique_hidden_values = sorted(
        set(
            "None" if opt["hidden"] is None else str(opt["hidden"])
            for opt in options["options"]
        )
    )
    unique_splits = sorted(set(opt["split"] for opt in options["options"]))

    # Toggle collapse section for filters
    @app.callback(
        Output("filter-collapse", "is_open"),
        [Input("filter-collapse-button", "n_clicks")],
        [State("filter-collapse", "is_open")],
    )
    def toggle_filter_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    # Function to handle "Select All" functionality for dropdowns
    def select_all_dropdown(filter_id, unique_values):
        @app.callback(
            Output(filter_id, "value"),
            [Input(filter_id, "value")],
            [State(filter_id, "options")],
        )
        def update_dropdown(selected_values, options):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            else:
                if selected_values is None:
                    selected_values = []
                if "All" in selected_values:
                    # If "Select All" is selected, return all values except "All"
                    all_values = [
                        option["value"]
                        for option in options
                        if option["value"] != "All"
                    ]
                    return all_values
                elif len(selected_values) == len(options) - 1:
                    # If all options are selected, include "All" in selection
                    return ["All"] + selected_values
                else:
                    # Remove "All" if not all options are selected
                    return [value for value in selected_values if value != "All"]

    # Apply "Select All" functionality to each dropdown
    select_all_dropdown("filter-model-dropdown", unique_models)
    select_all_dropdown("filter-dataset-dropdown", unique_datasets)
    select_all_dropdown("filter-layer-dropdown", unique_layers)
    select_all_dropdown("filter-width-dropdown", unique_widths)
    select_all_dropdown("filter-top-n-dropdown", unique_top_ns)
    select_all_dropdown("filter-config-name-dropdown", unique_config_names)
    select_all_dropdown("filter-binarize-dropdown", unique_binarize_values)
    select_all_dropdown("filter-hidden-dropdown", unique_hidden_values)
    select_all_dropdown("filter-split-dropdown", unique_splits)

    # Populate the DataTable based on selected filters and automatically select all displayed rows
    @app.callback(
        [
            Output("runs-table", "data"),
            Output("runs-table", "selected_rows"),
            Output("runs-table", "tooltip_data"),
        ],
        [
            Input("filter-model-dropdown", "value"),
            Input("filter-dataset-dropdown", "value"),
            Input("filter-layer-dropdown", "value"),
            Input("filter-width-dropdown", "value"),
            Input("filter-top-n-dropdown", "value"),
            Input("filter-config-name-dropdown", "value"),
            Input("filter-binarize-dropdown", "value"),
            Input("filter-hidden-dropdown", "value"),
            Input("filter-split-dropdown", "value"),
            Input("runs-table", "sort_by"),
        ],
    )
    def update_runs_table(
        selected_models,
        selected_datasets,
        selected_layers,
        selected_widths,
        selected_top_ns,
        selected_config_names,
        selected_binarize_values,
        selected_hidden_values,
        selected_splits,
        sort_by,
    ):
        # Filter options based on selected values
        filtered_options = [
            opt
            for opt in options["options"]
            if opt["model"] in selected_models
            and opt["dataset"] in selected_datasets
            and str(opt["layer"]) in selected_layers
            and opt["width"] in selected_widths
            and str(opt["top_n"]) in selected_top_ns
            and (opt["config_name"] or "None") in selected_config_names
            and (
                "None" if opt["binarize_value"] is None else str(opt["binarize_value"])
            )
            in selected_binarize_values
            and ("None" if opt["hidden"] is None else str(opt["hidden"]))
            in selected_hidden_values
            and opt["split"] in selected_splits
        ]

        # Create data for the DataTable
        run_data = []
        tooltip_data = []
        for opt in filtered_options:
            row = {
                "model": opt["model"],
                "dataset": opt["dataset"],
                "layer": str(opt["layer"]),
                "width": opt["width"],
                "top_n": str(opt["top_n"]),
                "config_name": opt["config_name"] or "None",
                "binarize_value": (
                    "None"
                    if opt["binarize_value"] is None
                    else str(opt["binarize_value"])
                ),
                "hidden": "None" if opt["hidden"] is None else str(opt["hidden"]),
                "split": opt["split"],
            }

            # Add metrics to the row, rounded to 2 decimal places
            for metric in [
                "linearProbe_mean_cv_accuracy",
                "linearProbe_test_accuracy",
                "linearProbe_macro_precision",
                "linearProbe_macro_recall",
                "linearProbe_macro_f1_score",
                "decisionTree_mean_cv_accuracy",
                "decisionTree_test_accuracy",
                "decisionTree_macro_precision",
                "decisionTree_macro_recall",
                "decisionTree_macro_f1_score",
            ]:
                metric_value = opt.get(metric, None)
                row[metric] = (
                    round(metric_value, 2) if metric_value is not None else None
                )

            run_data.append(row)
            tooltip_row = {
                column: {"value": str(value), "type": "markdown"}
                for column, value in row.items()
            }
            tooltip_data.append(tooltip_row)

        # Sort the data if sort_by is provided
        if sort_by:
            run_data = sorted(
                run_data,
                key=lambda x: x[sort_by[0]["column_id"]],
                reverse=sort_by[0]["direction"] == "desc",
            )

        # Automatically select all displayed rows
        selected_rows = list(range(len(run_data)))

        return run_data, selected_rows, tooltip_data

    # Generate comparison plot based on selected runs
    @app.callback(
        Output("compare-models-bar-plot", "figure"),
        [
            Input("runs-table", "derived_virtual_data"),
            Input("runs-table", "derived_virtual_selected_rows"),
            Input("compare-metric-selector", "value"),
            Input("runs-table", "sort_by"),
        ],
    )
    def update_compare_models_bar_plot(rows, selected_rows, selected_metric, sort_by):
        if not rows or not selected_metric:
            logging.debug("No data or metric selected.")
            return go.Figure()

        # Use derived_virtual_data and derived_virtual_selected_rows
        selected_runs = [rows[i] for i in selected_rows if i < len(rows)]

        # If sort_by is provided, sort the selected_runs accordingly
        if sort_by:
            selected_runs.sort(
                key=lambda x: x.get(sort_by[0]["column_id"], 0),
                reverse=sort_by[0]["direction"] == "desc",
            )

        metric_values = []
        labels = []
        colors = []
        for run in selected_runs:
            metric_value = run.get(selected_metric, None)
            if metric_value is not None:
                metric_values.append(metric_value)
                label = (
                    f"Model: {run['model']}<br>"
                    f"Dataset: {run['dataset']}<br>"
                    f"Layer: {run['layer']}<br>"
                    f"Top N: {run['top_n']}<br>"
                    f"Bin: {run['binarize_value']}<br>"
                    f"Hidden: {run['hidden']}<br>"
                    f"Split: {run['split']}"
                )
                labels.append(label)
                colors.append("#1f77b4")

        if not metric_values:
            logging.debug("No metric values were extracted.")
            return go.Figure()

        # Create the horizontal bar plot
        fig = go.Figure(
            data=[
                go.Bar(
                    x=metric_values,
                    y=labels,
                    text=[f"{v:.2f}" for v in metric_values],
                    textposition="auto",
                    orientation="h",
                    marker=dict(color=colors),
                )
            ],
        )

        # Update layout with adjusted settings for stacked labels
        fig.update_layout(
            title=f"Comparison of {selected_metric.replace('_', ' ').capitalize()}",
            yaxis_title="Configurations",
            xaxis_title=selected_metric.replace("_", " ").capitalize(),
            template="plotly_white",
            height=max(
                400, len(labels) * 150
            ),  # Dynamic height based on number of entries
            margin=dict(l=250),  # Increased left margin for stacked labels
            yaxis=dict(
                categoryorder="array",
                categoryarray=labels[::-1],  # Reverse the order to match the table
            ),
        )

        return fig
