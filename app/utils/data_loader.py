from pathlib import Path
import json
from typing import List, Dict, Any
from app.utils.paths import get_dashboard_dir


def load_available_models(dashboard_dir: str) -> List[Dict[str, str]]:
    """Load all available dashboard files and extract model information."""
    dashboard_path = get_dashboard_dir(dashboard_dir)
    dashboard_files = list(dashboard_path.glob("*_dashboard.json"))
    models = []

    for file_path in dashboard_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            model_info = data["metadata"]["model"]
            models.append(
                {
                    "name": model_info["name"],
                    "layer": model_info["layer"],
                    "type": model_info["type"],
                    "path": str(file_path),
                    "display_name": f"{model_info['name']} (Layer {model_info['layer']})",
                }
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return sorted(models, key=lambda x: (x["name"], x["layer"]))


def load_dashboard_data(file_path: str) -> Dict[str, Any]:
    """Load dashboard data from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dashboard data from {file_path}: {e}")
        raise


def load_model_data(
    models: list, selected_model: str, selected_layer: str
) -> Dict[str, Any]:
    """Get dashboard data for a specific model and layer."""
    # Find the matching model file
    model_info = next(
        (
            m
            for m in models
            if m["name"] == selected_model and str(m["layer"]) == str(selected_layer)
        ),
        None,
    )

    if not model_info:
        raise ValueError(
            f"No data found for model {selected_model} layer {selected_layer}"
        )

    return load_dashboard_data(model_info["path"])


def get_top_features(
    importance_scores: List[float], n: int = 10
) -> List[Dict[str, Any]]:
    """Get top N most important features with their scores."""
    feature_scores = [
        {"index": i, "score": float(score)} for i, score in enumerate(importance_scores)
    ]
    return sorted(feature_scores, key=lambda x: abs(x["score"]), reverse=True)[:n]
