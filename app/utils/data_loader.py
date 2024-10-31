from pathlib import Path
import json
from typing import Dict, Any, List
import logging
from collections import defaultdict


def create_path_key(model: str, dataset: str, split: str, layer: str) -> str:
    """Create a serializable key for path lookup."""
    return f"{model}||{dataset}||{split}||{layer}"


def parse_path_key(key: str) -> Dict[str, str]:
    """Parse a path key back into its components."""
    model, dataset, split, layer = key.split("||")
    return {"model": model, "dataset": dataset, "split": split, "layer": layer}


def extract_path_components(file_path: Path) -> Dict[str, str]:
    """
    Extract components from a file path with structure like:
    dashboard_data/google_gemma-2-2b/Anthropic_election_questions/test/layer_12/16k/classifier_results.json
    """
    try:
        parts = list(file_path.parts)
        results_idx = parts.index("classifier_results.json")

        layer = parts[results_idx - 2].replace("layer_", "")
        split = parts[results_idx - 3]
        dataset = parts[results_idx - 4]
        model = parts[results_idx - 5]

        logging.info(f"Extracted path components from {file_path}:")
        logging.info(f"  Model: {model}")
        logging.info(f"  Dataset: {dataset}")
        logging.info(f"  Split: {split}")
        logging.info(f"  Layer: {layer}")

        return {"model": model, "dataset": dataset, "split": split, "layer": layer}
    except Exception as e:
        logging.error(f"Failed to extract path components from {file_path}: {str(e)}")
        return None


def load_available_options(dashboard_dir: str) -> Dict[str, Any]:
    """Load available options for model selection without loading full model data."""
    dashboard_path = Path(dashboard_dir)
    logging.info(f"Loading options from dashboard directory: {dashboard_path}")

    available_data = {
        "models": set(),
        "datasets": defaultdict(set),
        "splits": defaultdict(lambda: defaultdict(set)),
        "layers": defaultdict(set),
        "paths": {},
    }

    result_files = list(dashboard_path.rglob("classifier_results.json"))
    logging.info(f"Found {len(result_files)} result files")

    for file_path in result_files:
        try:
            if not file_path.is_file():
                logging.warning(f"File not found or not readable: {file_path}")
                continue

            components = extract_path_components(file_path)
            if not components:
                continue

            model_name = components["model"]
            dataset_name = components["dataset"]
            split = components["split"]
            layer = components["layer"]

            available_data["models"].add(model_name)
            available_data["datasets"][model_name].add(dataset_name)
            available_data["splits"][model_name][dataset_name].add(split)
            available_data["layers"][model_name].add(layer)

            path_key = create_path_key(model_name, dataset_name, split, layer)
            available_data["paths"][path_key] = str(file_path)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            continue

    if not available_data["models"]:
        logging.warning("No valid models found in the dashboard directory")

    result = {
        "models": sorted(list(available_data["models"])),
        "datasets": {
            model: sorted(list(datasets))
            for model, datasets in available_data["datasets"].items()
        },
        "splits": {
            model: {
                dataset: sorted(list(splits))
                for dataset, splits in dataset_splits.items()
            }
            for model, dataset_splits in available_data["splits"].items()
        },
        "layers": {
            model: sorted(list(layers), key=lambda x: int(x))
            for model, layers in available_data["layers"].items()
        },
        "paths": available_data["paths"],
    }

    return result


def get_model_path(
    options: Dict[str, Any], model: str, dataset: str, split: str, layer: str
) -> str:
    """Get the file path for a specific model configuration."""
    path_key = create_path_key(model, dataset, split, layer)
    path = options["paths"].get(path_key)

    if path is None:
        logging.error(
            f"No path found for: Model={model}, Dataset={dataset}, "
            f"Split={split}, Layer={layer}"
        )
        return None

    logging.info(f"Found path for key {path_key}: {path}")
    return path


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
