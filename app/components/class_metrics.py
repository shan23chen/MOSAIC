from dash import html, dcc


def create_class_metrics_section(linear_metrics, tree_metrics):
    """Create complete class metrics section."""
    return html.Div(
        [
            html.H3(
                "Class Performance Analysis",
                style={"marginBottom": "20px", "color": "#1a237e"},
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Linear Probe",
                        children=[create_class_metrics_table(linear_metrics)],
                    ),
                    dcc.Tab(
                        label="Decision Tree",
                        children=[create_class_metrics_table(tree_metrics)],
                    ),
                ]
            ),
        ],
        style={"marginBottom": "30px"},
    )


def create_class_metrics_table(metrics):
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
