from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from app.styles import COLORS, STYLES
from typing import Dict, Any
import plotly.express as px
import dash_bootstrap_components as dbc


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


def create_feature_importance_section(
    linear_features: Dict[str, Any], tree_features: Dict[str, Any]
) -> html.Div:
    """Create an enhanced feature importance visualization section with interactive elements."""

    def create_importance_plot(features, title, color_scheme):
        if not features.get("top_features"):
            return html.Div(f"No feature importance data available for {title}")

        feature_data = features["top_features"]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[str(f["index"]) for f in feature_data],
                y=[abs(f["score"]) for f in feature_data],
                name=title,
                marker_color=color_scheme,
                hovertemplate="Feature %{x}<br>Score: %{y:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title={
                "text": f"{title} Feature Importance",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis_title="Feature Index",
            yaxis_title="Importance Score",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Open Sans"),
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return dcc.Graph(
            figure=fig,
            config={"displayModeBar": True, "scrollZoom": True},
            className="shadow-sm",
        )

    return html.Div(
        [
            html.H3("Feature Importance Analysis", className="mb-4 text-primary"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            create_importance_plot(
                                                linear_features,
                                                "Linear Probe",
                                                px.colors.sequential.Blues,
                                            )
                                        ]
                                    )
                                ],
                                className="shadow-sm hover-card",
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            create_importance_plot(
                                                tree_features,
                                                "Decision Tree",
                                                px.colors.sequential.Purples,
                                            )
                                        ]
                                    )
                                ],
                                className="shadow-sm hover-card",
                            )
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )
