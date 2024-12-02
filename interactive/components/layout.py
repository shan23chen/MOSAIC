from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Any
import logging
import json
from dash.dependencies import Input, Output, State
import plotly.express as px

from interactive.components.model import process_image
from interactive.components.plots import create_importance_plot
from interactive.components.tree_viz import create_tree_visualization, get_tree_info


def create_layout() -> html.Div:
    """Create main dashboard layout with options banner."""
    return html.Div([
        html.H1("Image Feature Extractor with VLM and SAE"),
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select an Image')]),
            style={
                'width': '50%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
            accept="image/*"
        ),
        html.Div(id='output-image-upload'),
        html.Div(id='feature-display')
    ])



def register_callbacks(app, model_name, model, processor, sae, neuron_cache, sae_layer, classifier, label_encoder, top_n):
    """Register callbacks for the dashboard components."""

    @app.callback(
        [Output('output-image-upload', 'children'),
        Output('feature-display', 'children')],
        [Input('upload-image', 'contents')]
    )

    def update_output(contents):
        if contents is not None:
            # Display uploaded image
            image_data = html.Img(src=contents, style={'height': '300px'})

            feature_graphs = []

            # Process image and extract features
            feature_names, feature_values, feature_indexes, title = process_image(contents, model_name, model, processor, sae, neuron_cache, sae_layer, classifier, label_encoder, top_n) 
            
            feature_graphs.append(create_importance_plot(feature_names, feature_values, feature_indexes, px.colors.sequential.Blues, title))

            if title == "Decision Tree":
                feature_graphs.append(create_tree_visualization(get_tree_info(classifier, feature_names), [str(x) for x in list(label_encoder.classes_)]))
            
            return image_data, feature_graphs
        return None, None



