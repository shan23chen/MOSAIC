import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import webbrowser
from threading import Timer
import numpy as np
import os


def open_browser():
    webbrowser.open("http://127.0.0.1:8050")


def get_tree_info(tree_model):
    tree = tree_model.tree_
    return {
        "children_left": tree.children_left,  # Array of left child indices
        "children_right": tree.children_right,  # Array of right child indices
        "feature": tree.feature,  # Feature used for splitting at each node
        "threshold": tree.threshold,  # Threshold values for splits
        "n_node_samples": tree.n_node_samples,  # Number of samples at each node
        "impurity": tree.impurity,  # Gini impurity at each node
        "value": tree.value,  # Class distribution at each node
    }


def create_class_distribution_badges(node):
    """Create class distribution badges with consistent styling."""
    return html.Div(
        [
            html.Span(
                f"C{i}: {(value/node['samples']*100):.1f}%",
                style={
                    "backgroundColor": "rgba(156, 39, 176, 0.1)",
                    "padding": "2px 6px",
                    "borderRadius": "4px",
                    "marginRight": "4px",
                    "display": "inline-block",
                    "color": (
                        "#C0392B" if i == 0 else "#27AE60" if i == 1 else "#2980B9"
                    ),
                },
            )
            for i, value in enumerate(node["value"][0])
        ],
        style={"marginTop": "8px"},
    )


def create_tree_visualization(tree_info):
    """Create tree visualization from saved tree information."""

    def create_node_component(node_id):
        """Create a single node component."""
        # Check if it's a leaf node (no children)
        is_leaf = tree_info["children_left"][node_id] == -1

        # Get node information
        samples = tree_info["n_node_samples"][node_id]
        gini = tree_info["impurity"][node_id]
        values = tree_info["value"][node_id][0]  # Class distribution
        total_samples = sum(values)

        # Calculate class percentages
        class_percentages = [(value / total_samples) * 100 for value in values]

        # Create class distribution badges
        class_badges = html.Div(
            [
                html.Span(
                    f"C{i}: {pct:.1f}%",
                    style={
                        "backgroundColor": "rgba(156, 39, 176, 0.1)",
                        "padding": "2px 6px",
                        "borderRadius": "4px",
                        "marginRight": "4px",
                        "display": "inline-block",
                        "color": ["#C0392B", "#27AE60", "#2980B9"][
                            i % 3
                        ],  # Rotating colors
                    },
                )
                for i, pct in enumerate(class_percentages)
            ],
            style={"marginTop": "8px"},
        )

        if not is_leaf:
            # Split node
            content = [
                html.Div(
                    [
                        html.Span("▾", style={"marginRight": "5px"}),
                        html.Strong(f"Feature {tree_info['feature'][node_id]}"),
                    ]
                ),
                html.Div(f"threshold ≤ {tree_info['threshold'][node_id]:.3f}"),
                html.Div(f"Samples: {samples} | Gini: {gini:.3f}"),
                class_badges,
                # Recursively create child nodes
                html.Div(
                    [
                        create_node_component(tree_info["children_left"][node_id]),
                        create_node_component(tree_info["children_right"][node_id]),
                    ],
                    style={"marginLeft": "40px"},
                ),
            ]
        else:
            # Leaf node
            content = [
                html.Div(
                    [
                        html.Span(
                            "▸", style={"marginRight": "5px", "visibility": "hidden"}
                        ),
                        html.Strong("Leaf Node"),
                    ]
                ),
                html.Div(f"Samples: {samples} | Gini: {gini:.3f}"),
                class_badges,
            ]

        return html.Div(
            content,
            style={
                "backgroundColor": "#F3E5F5" if not is_leaf else "#F5F5F5",
                "padding": "15px",
                "borderRadius": "4px",
                "marginBottom": "10px",
            },
        )

    # Start with root node (index 0)
    return html.Div(
        [create_node_component(0)],
        style={
            "backgroundColor": "white",
            "border": "1px solid #E0E0E0",
            "borderRadius": "4px",
            "padding": "20px",
        },
    )


def create_feature_importance_plot(data):
    """Create feature importance visualization from dashboard data."""
    feature_importances = data.get("feature_importances", [])

    importance_data = [
        {"Feature": f"Feature {i}", "Importance": float(importance)}
        for i, importance in enumerate(feature_importances)
    ]

    if importance_data:
        df = pd.DataFrame(importance_data)
        df = df.sort_values("Importance", ascending=False).head(10)

        fig = go.Figure(
            go.Bar(
                x=df["Feature"],
                y=df["Importance"],
                marker_color="rgb(149, 117, 205)",
                hovertemplate="<b>%{x}</b><br>importance : %{y:.3f}<br><extra></extra>",
            )
        )

        fig.update_layout(
            margin={"l": 40, "r": 20, "t": 20, "b": 30},
            height=300,
            yaxis=dict(
                title="Importance",
                range=[0, max(df["Importance"]) * 1.1],
                tickformat=".3f",
            ),
            xaxis=dict(title="", tickangle=-45),
            showlegend=False,
            plot_bgcolor="white",
            bargap=0.5,
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        )
        return fig

    return go.Figure()


def create_per_class_metrics_table(metrics):
    """Create a formatted table for per-class metrics."""
    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Class", style={"textAlign": "left", "padding": "8px"}),
                        html.Th(
                            "Precision", style={"textAlign": "right", "padding": "8px"}
                        ),
                        html.Th(
                            "Recall", style={"textAlign": "right", "padding": "8px"}
                        ),
                        html.Th("F1", style={"textAlign": "right", "padding": "8px"}),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                class_name,
                                style={"textAlign": "left", "padding": "8px"},
                            ),
                            html.Td(
                                f"{metrics['precision']*100:.1f}%",
                                style={"textAlign": "right", "padding": "8px"},
                            ),
                            html.Td(
                                f"{metrics['recall']*100:.1f}%",
                                style={"textAlign": "right", "padding": "8px"},
                            ),
                            html.Td(
                                f"{metrics['f1_score']*100:.1f}%",
                                style={"textAlign": "right", "padding": "8px"},
                            ),
                        ]
                    )
                    for class_name, metrics in metrics["per_class_metrics"].items()
                ]
            ),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


def create_model_performance_section(linear_metrics, tree_metrics):
    """Create the model performance section of the dashboard."""
    return html.Div(
        [
            html.H3(
                "Model Performance",
                style={"marginBottom": "20px", "fontWeight": "bold"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Training Score", style={"marginBottom": "8px"}),
                            html.Div(
                                f"{linear_metrics['accuracy']*100:.1f}%",
                                style={
                                    "fontSize": "24px",
                                    "color": "#2E7D32",
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#E8F5E9",
                            "padding": "20px",
                            "flex": 1,
                            "borderRadius": "4px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div("Test Score", style={"marginBottom": "8px"}),
                            html.Div(
                                f"{tree_metrics['accuracy']*100:.1f}%",
                                style={
                                    "fontSize": "24px",
                                    "color": "#1976D2",
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#E3F2FD",
                            "padding": "20px",
                            "flex": 1,
                            "borderRadius": "4px",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "20px", "marginBottom": "30px"},
            ),
        ]
    )


def create_dashboard(dashboard_path):
    """Create the main Dash application."""
    app = dash.Dash(__name__)

    with open(dashboard_path, "r") as f:
        data = json.load(f)

    linear_metrics = data["metrics"]["linearProbe"]
    tree_metrics = data["metrics"]["decisionTree"]

    app.layout = html.Div(
        [
            create_model_performance_section(linear_metrics, tree_metrics),
            # Feature Importance Section
            html.Div(
                [
                    html.H3(
                        "Feature Importance",
                        style={"marginBottom": "20px", "fontWeight": "bold"},
                    ),
                    dcc.Graph(
                        id="feature-importance-graph",
                        figure=create_feature_importance_plot(data),
                        config={"displayModeBar": False},
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
            # Per-Class Metrics Section
            html.Div(
                [
                    html.H3(
                        "Per-Class Metrics",
                        style={"marginBottom": "20px", "fontWeight": "bold"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4(
                                        "Linear Probe", style={"marginBottom": "15px"}
                                    ),
                                    create_per_class_metrics_table(linear_metrics),
                                ],
                                style={"flex": 1},
                            ),
                            html.Div(
                                [
                                    html.H4(
                                        "Decision Tree", style={"marginBottom": "15px"}
                                    ),
                                    create_per_class_metrics_table(tree_metrics),
                                ],
                                style={"flex": 1},
                            ),
                        ],
                        style={"display": "flex", "gap": "40px"},
                    ),
                ]
            ),
            # Decision Tree Structure Section
            html.Div(
                [
                    html.H3(
                        "Decision Tree Structure",
                        style={"marginBottom": "20px", "fontWeight": "bold"},
                    ),
                    html.Div(
                        create_tree_visualization(data["tree_info"]),
                        style={
                            "backgroundColor": "white",
                            "border": "1px solid #E0E0E0",
                            "borderRadius": "4px",
                            "padding": "20px",
                        },
                    ),
                ],
                style={"marginTop": "30px"},
            ),
        ],
        style={
            "padding": "20px",
            "maxWidth": "1200px",
            "margin": "0 auto",
            "fontFamily": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial',
        },
    )

    return app


def run_dash_server(dashboard_path):
    """Run the Dash server and open the browser."""
    app = create_dashboard(dashboard_path)

    app.run_server(debug=False, port=8050)


if __name__ == "__main__":
    run_dash_server("processed_features_llm/google/gemma-2b-it_12_dashboard.json")
