import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px


def collect_tree_data(tree_structure):
    topology = tree_structure["topology"]
    nodes = []
    edges = []
    max_depth = 0

    def traverse(node_id, depth, x_pos):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        is_leaf = topology["children_left"][node_id] == -1

        # Adjust x-position scaling factor based on depth for better horizontal spacing
        base_spacing_factor = 2.5  # Base spacing factor
        scaling_factor = 3.5  # Scaling factor to increase spacing with depth
        spacing_factor = base_spacing_factor * (1 + scaling_factor * depth)

        nodes.append(
            {
                "id": node_id,
                "x": x_pos * spacing_factor,
                "y": -depth * 1.2,  # Increase vertical spacing
                "depth": depth,
            }
        )

        if not is_leaf:
            left_child = topology["children_left"][node_id]
            right_child = topology["children_right"][node_id]

            edges.append((node_id, left_child))
            edges.append((node_id, right_child))

            # Adjust child node positioning
            child_spacing = 1 / (2 ** (depth + 0.8))  # Modified spacing formula
            traverse(left_child, depth + 1, x_pos - child_spacing)
            traverse(right_child, depth + 1, x_pos + child_spacing)

    traverse(0, 0, x_pos=0)
    return nodes, edges, max_depth


def create_tree_visualization(dashboard_data: Dict[str, Any]) -> html.Div:
    tree_structure = dashboard_data["models"]["decisionTree"]["tree_structure"]
    topology = tree_structure["topology"]
    node_data = tree_structure["node_data"]
    class_names = dashboard_data.get("metadata", {}).get("class_names", None)

    nodes, edges, max_depth = collect_tree_data(tree_structure)

    # Calculate majority classes
    majority_classes = []
    for node in nodes:
        node_id = node["id"]
        values = node_data.get("values", [[]])[node_id]
        values = values[0] if isinstance(values[0], (list, np.ndarray)) else values
        majority_classes.append(
            np.argmax(values) if values and sum(values) > 0 else None
        )

    # Set up colors
    num_classes = (
        len(class_names)
        if class_names
        else (max(majority_classes) + 1 if majority_classes else 1)
    )
    color_palette = px.colors.qualitative.Plotly[:num_classes]
    class_colors = {i: color_palette[i] for i in range(num_classes)}
    default_color = "rgba(200, 200, 200, 0.6)"

    fig = go.Figure()

    # Add edges with curved paths
    for edge in edges:
        parent_id, child_id = edge
        parent_node = next(node for node in nodes if node["id"] == parent_id)
        child_node = next(node for node in nodes if node["id"] == child_id)

        mid_x = (parent_node["x"] + child_node["x"]) / 2

        fig.add_trace(
            go.Scatter(
                x=[parent_node["x"], mid_x, mid_x, child_node["x"]],
                y=[
                    parent_node["y"],
                    parent_node["y"],
                    child_node["y"],
                    child_node["y"],
                ],
                mode="lines",
                line=dict(color="rgba(128, 128, 128, 0.5)", width=1),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Create hover text and colors for nodes
    hover_texts = []
    node_colors = []

    for node in nodes:
        node_id = node["id"]
        is_leaf = topology["children_left"][node_id] == -1

        # Prepare node information
        feature = topology["feature_names"][node_id] if not is_leaf else "Leaf"
        threshold = node_data["thresholds"][node_id] if not is_leaf else None
        samples = node_data.get("samples", [0])[node_id]
        values = node_data.get("values", [[]])[node_id]
        values = values[0] if isinstance(values[0], (list, np.ndarray)) else values

        # Build hover text
        hover_info = [
            f"<b>{feature}</b>",
            f"Threshold: ≤ {threshold:.3f}" if not is_leaf else None,
            f"Samples: {samples}",
        ]

        if values and sum(values) > 0:
            total = sum(values)
            percentages = [(v / total) * 100 for v in values]
            class_labels = class_names or [f"C{i}" for i in range(len(values))]
            distribution = "<br>".join(
                f"{label}: {pct:.1f}%"
                for label, pct in zip(class_labels, percentages)
                if pct > 0
            )
            hover_info.append(f"<br>Distribution:<br>{distribution}")

        hover_texts.append("<br>".join(filter(None, hover_info)))

        # Set node color based on majority class
        majority_class = majority_classes[node_id]
        node_colors.append(class_colors.get(majority_class, default_color))

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=[node["x"] for node in nodes],
            y=[node["y"] for node in nodes],
            mode="markers",
            marker=dict(
                size=25,
                color=node_colors,
                line=dict(color="black", width=1),
                opacity=0.9,
            ),
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        xaxis=dict(
            showline=False, zeroline=False, showgrid=False, showticklabels=False
        ),
        yaxis=dict(
            showline=False, zeroline=False, showgrid=False, showticklabels=False
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="white",
        height=max(500, 80 * (max_depth + 1)),
        width=max(800, 200 * (max_depth + 1)),
        hovermode="closest",
        dragmode="pan",
    )

    return html.Div(
        [
            html.H3("Decision Tree Structure", className="text-primary mb-3"),
            dcc.Graph(figure=fig, config={"scrollZoom": True, "displayModeBar": True}),
        ],
        style={"width": "100%", "overflowX": "auto"},
    )
