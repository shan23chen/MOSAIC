import dash_bootstrap_components as dbc
from dash import html
from typing import List, Dict, Any
from dash.dependencies import Input, Output, State, MATCH, ALL


def create_class_distribution_badges(values: List[float], total: float) -> html.Div:
    """Create class distribution badges with percentage values."""
    percentages = [(v / total) * 100 for v in values]

    return html.Div(
        [
            dbc.Badge(
                f"C{i}: {pct:.1f}%",
                color=(
                    "danger"
                    if i == 0  # red for class 0
                    else "success" if i == 1 else "primary"  # green for class 1
                ),  # blue for other classes
                className="mr-1 mb-1",
                style={"opacity": "0.8"},
            )
            for i, pct in enumerate(percentages)
            if pct > 0  # Only show classes that are present
        ],
        className="d-flex flex-wrap",
    )


def create_tree_visualization(dashboard_data: Dict[str, Any]) -> html.Div:
    """Create an improved tree visualization with dynamic collapsible nodes."""
    tree_structure = dashboard_data["models"]["decisionTree"]["tree_structure"]

    def create_node_component(node_id: int, depth: int = 0) -> html.Div:
        """Create a single node component with improved styling."""
        topology = tree_structure["topology"]
        node_data = tree_structure["node_data"]

        is_leaf = topology["children_left"][node_id] == -1

        # Get node information
        samples = node_data["samples"][node_id]
        gini = node_data["impurity"][node_id]
        values = node_data["values"][node_id][0]

        # Create node header with toggle button for non-leaf nodes
        header_content = []
        if not is_leaf:
            feature_idx = topology["feature_indices"][node_id]
            threshold = node_data["thresholds"][node_id]
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
                        f"Feature {feature_idx} (≤ {threshold:.3f})",
                    ],
                    className="d-flex align-items-center",
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
        node_content = [
            html.Div(header_content, className="mb-1"),
            html.Div(
                f"Samples: {samples} | Gini: {gini:.3f}",
                className="small text-muted mb-1",
            ),
            create_class_distribution_badges(values, sum(values)),
        ]

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
