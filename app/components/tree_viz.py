import numpy as np
import dash_bootstrap_components as dbc
from dash import html
from typing import List, Dict, Any
from dash.dependencies import Input, Output, State, MATCH, ALL


def create_class_distribution_badges(
    values: List[float], total: float, class_names: List[str]
) -> html.Div:
    """
    Create class distribution badges with percentage values and class names.

    Args:
        values: List of values for each class
        total: Total sum of values
        class_names: List of class names corresponding to indices. If None, uses C0, C1, etc.
    """
    if isinstance(values[0], (list, np.ndarray)):
        values = values[0]  # Unnest if needed

    total = sum(values)  # Recalculate total from unnested values
    percentages = [(v / total) * 100 if total > 0 else 0 for v in values]

    # Use provided class names or fallback to C0, C1, etc.
    if class_names is None:
        class_names = [f"C{i}" for i in range(len(values))]

    # Ensure we have enough class names
    while len(class_names) < len(values):
        class_names.append(f"C{len(class_names)}")

    return html.Div(
        [
            dbc.Badge(
                f"{class_names[i]}: {pct:.1f}%",
                color=(
                    "danger"
                    if i == 0  # red for first class
                    else (
                        "success" if i == 1 else "primary"  # green for second class
                    )  # blue for other classes
                ),
                className="mr-1 mb-1",
                style={
                    "opacity": "0.8",
                    "fontSize": "0.85em",  # Slightly smaller font for longer class names
                    "padding": "0.4em 0.6em",  # More padding for readability
                },
            )
            for i, pct in enumerate(percentages)
            if pct > 0  # Only show classes that are present
        ],
        className="d-flex flex-wrap gap-1",  # Added gap for better spacing
    )


def create_tree_visualization(dashboard_data: Dict[str, Any]) -> html.Div:
    """Create an improved tree visualization using embedded feature names."""
    tree_structure = dashboard_data["models"]["decisionTree"]["tree_structure"]
    topology = tree_structure["topology"]
    node_data = tree_structure["node_data"]

    # Get class names from metadata if available
    class_names = dashboard_data.get("metadata", {}).get("class_names", None)

    def create_node_component(node_id: int, depth: int = 0) -> html.Div:
        """Create a single node component with feature descriptions."""
        is_leaf = topology["children_left"][node_id] == -1

        # Get node information
        samples = node_data["samples"][node_id] if "samples" in node_data else 0
        impurity = node_data["impurity"][node_id] if "impurity" in node_data else 0

        # Handle nested values array
        values = []
        if "values" in node_data:
            node_values = node_data["values"][node_id]
            if isinstance(node_values, (list, np.ndarray)):
                values = (
                    node_values[0]
                    if isinstance(node_values[0], (list, np.ndarray))
                    else node_values
                )
            else:
                values = [node_values]

        # Create node header with toggle button for non-leaf nodes
        header_content = []
        if not is_leaf:
            feature_idx = topology["feature_indices"][node_id]
            threshold = node_data["thresholds"][node_id]

            # Get feature name from the embedded feature_names list
            feature_name = topology["feature_names"][node_id]

            header_content = [
                html.Div(
                    [
                        html.Button(
                            [
                                html.I(
                                    className="fas fa-caret-right",
                                    style={"transition": "transform 0.2s"},
                                )
                            ],
                            id={"type": "node-toggle", "index": node_id},
                            className="btn btn-link p-0 mr-2",
                            n_clicks=0,
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            f"Feature {feature_idx}",
                                            className="font-weight-bold",
                                        ),
                                        html.Span(
                                            f" (≤ {threshold:.3f})",
                                            className="text-muted ml-2",
                                        ),
                                    ],
                                    className="d-flex align-items-center",
                                ),
                                html.Div(
                                    feature_name,
                                    className="text-muted small mt-1",
                                    style={
                                        "maxWidth": "400px",
                                        "wordWrap": "break-word",
                                    },
                                ),
                            ]
                        ),
                    ],
                    className="d-flex align-items-start",
                )
            ]
        else:
            header_content = [
                html.Div(
                    [html.I(className="fas fa-leaf mr-2 text-success"), "Leaf Node"],
                    className="d-flex align-items-center",
                )
            ]

        # Add node information
        node_info = []
        if "samples" in node_data:
            node_info.append(f"Samples: {samples}")
        if "impurity" in node_data:
            node_info.append(f"Gini: {impurity:.3f}")

        node_content = [
            html.Div(header_content, className="mb-1"),
            html.Div(" | ".join(node_info), className="small text-muted mb-1"),
        ]

        if values:
            node_content.append(
                create_class_distribution_badges(values, sum(values), class_names)
            )

        # Create children container for non-leaf nodes
        if not is_leaf:
            children_div = html.Div(
                [
                    create_node_component(
                        topology["children_left"][node_id], depth + 1
                    ),
                    create_node_component(
                        topology["children_right"][node_id], depth + 1
                    ),
                ],
                id={"type": "node-children", "index": node_id},
                className="pl-4 mt-2",
                style={"display": "block"},
            )
            node_content.append(children_div)

        return html.Div(
            node_content,
            className="mb-2 p-2 rounded",
            style={
                "backgroundColor": "#f8f9fa" if not is_leaf else "white",
                "border": "1px solid #dee2e6",
            },
        )

    return html.Div(
        [
            html.H3("Decision Tree Structure", className="text-primary mb-3"),
            dbc.Card(dbc.CardBody(create_node_component(0)), className="bg-white"),
        ]
    )


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
