from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from app.styles import COLORS, STYLES
from typing import Dict, Any
import plotly.express as px
import dash_bootstrap_components as dbc


def create_importance_plot(feature_names, feature_strengths, feature_indexes, color_scheme, model_title):

    title = f"{model_title} Importance Plot"
    # Create hover text using embedded feature names
    hover_text = []
    for feature_strength, feature_name,feature_index in zip(feature_strengths, feature_names, feature_indexes):
        hover_text.append(
            f"Feature {feature_index}<br>{feature_name}<br>Score: {abs(feature_strength):.3f}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(
                range(len(feature_strengths))
            ),  # Use numeric indices for x-axis positions
            y=[abs(f) for f in feature_strengths],
            name=title,
            marker_color=color_scheme,
            text=None,  # Remove text on bars
            hoverinfo="text",
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    # Create x-axis labels with feature descriptions
    descriptions = feature_names
    short_descriptions = [
        d[:50] + "..." if len(d) > 50 else d for d in descriptions
    ]

    fig.update_layout(
        title={
            "text": f"{title} Feature Importance",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis=dict(
            ticktext=short_descriptions,
            tickvals=list(range(len(feature_names))),
            tickangle=-45,
        ),
        yaxis_title="Importance Score",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Open Sans"),
        showlegend=False,
        margin=dict(
            l=40, r=40, t=60, b=120
        ),  # Increased bottom margin for rotated labels
        height=500,  # Increased height to accommodate labels
    )

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": True, "scrollZoom": True},
        className="shadow-sm",
    )