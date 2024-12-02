import argparse
import webbrowser
from threading import Timer
from pathlib import Path
from dash import Dash
import dash_bootstrap_components as dbc

from interactive.components.layout import create_layout, register_callbacks
from interactive.components.model import load_model

def create_app(args) -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://use.fontawesome.com/releases/v5.15.4/css/all.css",
        ],
        suppress_callback_exceptions=True,  # Add this for pattern-matching callbacks
    )

    # Create layout with loaded models
    app.layout = create_layout()

    model, processor, sae, neuron_cache, classifier, label_encoder = load_model(args)

    # Register all callbacks
    register_callbacks(app, args.vlm_model, model, processor, sae, neuron_cache, args.sae_layer, classifier, label_encoder, args.top_n)

    return app

def open_browser():
    """Open the browser to the dashboard URL."""
    webbrowser.open("http://127.0.0.1:8050")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="../dashboard_data/",
        help="Path to dashboard directory",
    )
    parser.add_argument(
        "--sae_layer",
        type=int,
        default="../dashboard_data/",
        help="Path to dashboard directory",
    )
    parser.add_argument(
        "--neuronpedia_cache",
        type=str,
        default="../dashboard_data/",
        help="Path to dashboard directory",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="../dashboard_data/",
        help="Path to dashboard directory",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Path to dashboard directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with hot reloading",
    )

    args = parser.parse_args()

    # Create and configure app
    app = create_app(args)

    # Open browser only on non-debug mode
    if not args.debug:
        Timer(1, open_browser).start()

    # Run the server
    if args.debug:
        app.run_server(
            debug=True,
            dev_tools_hot_reload=True,
            dev_tools_hot_reload_interval=0.3,
            port=8050,
        )
    else:
        app.run_server(debug=False, port=8050)


if __name__ == "__main__":
    main()