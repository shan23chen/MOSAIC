from pathlib import Path
import json
from typing import Dict, Any, List, Literal, Optional
import logging
from collections import defaultdict
import os

ResultType = Literal["hidden", "sae"]


def create_path_key(
    model: str,
    dataset: str,
    layer: str,
    width: str,
    timestamp: str,
    result_type: ResultType,
) -> str:
    """Create a serializable key for path lookup."""
    return f"{model}||{dataset}||{layer}||{width}||{timestamp}||{result_type}"


def parse_path_key(key: str) -> Dict[str, str]:
    """Parse a path key back into its components."""
    model, dataset, layer, width, timestamp, result_type = key.split("||")
    return {
        "model": model,
        "dataset": dataset,
        "layer": layer,
        "width": width,
        "timestamp": timestamp,
        "result_type": result_type,
    }


def get_result_filename(result_type: ResultType) -> str:
    """Get the appropriate results filename based on type."""
    if result_type == "hidden":
        return "hidden_classifier_results.json"
    elif result_type == "sae":
        return "sae_classifier_results.json"
    else:
        raise ValueError(f"Invalid result type: {result_type}")


def load_available_options(dashboard_dir: str) -> Dict[str, Any]:
    """Load available options for model selection, including method-specific metrics."""
    dashboard_path = Path(dashboard_dir)
    logging.info(f"Loading options from dashboard directory: {dashboard_path}")

    options = []
    models = set()
    datasets = set()
    splits = set()
    widths = set()
    top_ns = set()
    config_names = set()
    binarize_values = set()
    layers = set()
    hidden_values = set()

    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(dashboard_path):
        for file in files:
            if file.endswith("_classifier_results.json"):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    metadata = data.get("metadata", {})
                    args = metadata.get("args", {})

                    model = args.get("model_name", None)
                    dataset = args.get("dataset_name", None)
                    width = args.get("width", None)
                    layer = metadata["model"].get("layer", None)
                    dataset_metadata = metadata.get("dataset", {})
                    split = args.get("dataset_split", None)

                    top_n = args.get("top_n", None)
                    config_name = args.get("dataset_config_name", None)
                    binarise = args.get("binarize_value", None)
                    hidden = dataset_metadata.get("hidden", None)

                    # Extract performance metrics for both linearProbe and decisionTree
                    metrics = {}
                    for method in ["linearProbe", "decisionTree"]:
                        method_data = data.get("models", {}).get(method, {})
                        performance = method_data.get("performance", {})
                        cross_validation = performance.get("cross_validation", {})
                        aggregated_metrics = method_data.get(
                            "aggregated_metrics", {}
                        ).get("macro avg", {})

                        metrics[f"{method}_mean_cv_accuracy"] = cross_validation.get(
                            "mean_accuracy", None
                        )
                        metrics[f"{method}_test_accuracy"] = performance.get(
                            "accuracy", None
                        )
                        metrics[f"{method}_macro_precision"] = aggregated_metrics.get(
                            "precision", None
                        )
                        metrics[f"{method}_macro_recall"] = aggregated_metrics.get(
                            "recall", None
                        )
                        metrics[f"{method}_macro_f1_score"] = aggregated_metrics.get(
                            "f1_score", None
                        )

                    combination = {
                        "model": model,
                        "dataset": dataset,
                        "split": split,
                        "width": width,
                        "top_n": top_n,
                        "config_name": config_name,
                        "binarize_value": binarise,
                        "layer": layer,
                        "hidden": hidden,
                        "filepath": file_path,
                        **metrics,  # Add method-specific metrics
                    }

                    models.add(model)
                    datasets.add(dataset)
                    splits.add(split)
                    widths.add(width)
                    layers.add(layer)
                    top_ns.add(top_n)
                    config_names.add(config_name)
                    binarize_values.add(binarise)
                    hidden_values.add(hidden)
                    options.append(combination)
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {e}")
                    continue

    result = {
        "models": sorted(list(models)),
        "datasets": sorted(list(datasets)),
        "splits": sorted(list(splits)),
        "widths": sorted(list(widths)),
        "top_ns": sorted(list(top_ns)),
        "config_names": sorted(list(config_names)),
        "binarize_values": list(binarize_values),
        "layers": sorted(list(layers)),
        "hidden_values": list(hidden_values),
        "options": options,
    }

    return result


def get_model_path(
    options: Dict[str, Any],
    model: str,
    dataset: str,
    split: str,
    layer: str,
    width: str,
    timestamp: Optional[str],
    result_type: ResultType,
    top_n: Optional[int] = None,
    binarize: Optional[bool] = None,
    config_name: Optional[str] = None,
) -> List[str]:
    """Get the file paths for a specific model configuration, filtered by optional parameters."""
    filtered_paths = []
    for combination_key, path_keys in options["combinations"].items():
        (
            c_model,
            c_dataset,
            c_split,
            c_layer,
            c_width,
            c_result_type,
            c_top_n,
            c_binarize,
            c_config_name,
        ) = combination_key

        if (
            c_model == model
            and c_dataset == dataset
            and c_split == split
            and c_layer == layer
            and c_width == width
            and c_result_type == result_type
            and (top_n is None or c_top_n == top_n)
            and (binarize is None or c_binarize == binarize)
            and (config_name is None or c_config_name == config_name)
        ):
            for path_key in path_keys:
                if timestamp and timestamp not in path_key:
                    continue
                filtered_paths.append(options["paths"][path_key])

    if not filtered_paths:
        logging.error(
            f"No path found for: Model={model}, Dataset={dataset}, "
            f"Layer={layer}, Width={width}, Timestamp={timestamp}, Result type={result_type}, "
            f"Top_n={top_n}, Binarize={binarize}, Config name={config_name}"
        )
        return []

    logging.info(f"Found paths: {filtered_paths}")
    return filtered_paths


def load_model_data(file_path: str) -> Dict[str, Any]:
    """Load the actual model data from a file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

            # Log the structure of the loaded data
            logging.info(f"Successfully loaded model data from {file_path}")
            logging.info("Data structure:")
            logging.info(f"Top-level keys: {list(data.keys())}")

            if "results" in data:
                logging.info("Results keys: %s", list(data["results"].keys()))

                if "linearProbe" in data["results"]:
                    logging.info(
                        "Linear probe keys: %s",
                        list(data["results"]["linearProbe"].keys()),
                    )

                if "decisionTree" in data["results"]:
                    logging.info(
                        "Decision tree keys: %s",
                        list(data["results"]["decisionTree"].keys()),
                    )

                # Log specific performance metrics if they exist
                if "performance" in data["results"].get("linearProbe", {}):
                    logging.info(
                        "Linear probe performance metrics: %s",
                        list(data["results"]["linearProbe"]["performance"].keys()),
                    )

                if "performance" in data["results"].get("decisionTree", {}):
                    logging.info(
                        "Decision tree performance metrics: %s",
                        list(data["results"]["decisionTree"]["performance"].keys()),
                    )

            return data
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error in {file_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading model data from {file_path}: {str(e)}")
        raise


def get_top_features(
    importance_scores: List[float], n: int = 10
) -> List[Dict[str, Any]]:
    """Get top N most important features with their scores."""
    feature_scores = [
        {"index": i, "score": float(score)} for i, score in enumerate(importance_scores)
    ]
    return sorted(feature_scores, key=lambda x: abs(x["score"]), reverse=True)[:n]
