import json
import numpy as np
from typing import Any, Dict, List
import argparse
from datetime import datetime


# Custom JSON encoder for numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


def get_top_features(importance_scores: List[Any], n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top N most important features with their scores.
    Handles both single-class and multi-class feature importance scores.

    Args:
        importance_scores: Either a list of scores or a list of lists for multi-class
        n: Number of top features to return

    Returns:
        List of dictionaries containing feature indices and their importance scores
    """
    # Handle multi-class case (list of lists)
    if isinstance(importance_scores[0], list):
        # Aggregate feature importance across all classes by taking the mean absolute value
        aggregated_scores = np.mean(
            [np.abs(scores) for scores in importance_scores], axis=0
        )
        feature_scores = [
            {"index": i, "score": float(score)}
            for i, score in enumerate(aggregated_scores)
        ]
    # Handle single-class case
    else:
        feature_scores = [
            {"index": i, "score": float(score)}
            for i, score in enumerate(importance_scores)
        ]

    return sorted(feature_scores, key=lambda x: abs(x["score"]), reverse=True)[:n]


def prepare_dashboard_data(
    linear_results: Dict[str, Any],
    tree_results: Dict[str, Any],
    args: argparse.Namespace,
    layer: str,
    tree_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare structured dashboard data from model results."""

    dashboard_data = {
        "metadata": {
            "model": {
                "name": args.model_name,
                "layer": layer,
                "type": args.model_type,
            },
            "training": {
                "test_size": args.test_size,
                "random_state": args.random_state,
                "cv_folds": args.cv_folds if hasattr(args, "cv_folds") else 5,
            },
            "timestamp": datetime.now().isoformat(),
        },
        "models": {
            "linearProbe": {
                "performance": {
                    "accuracy": linear_results["metrics"]["accuracy"],
                    "cross_validation": {
                        "mean_accuracy": linear_results["cv_scores"]["mean"],
                        "std_accuracy": linear_results["cv_scores"]["std"],
                        "fold_scores": linear_results["cv_scores"]["scores"],
                    },
                },
                "hyperparameters": {
                    "best_params": linear_results["best_params"],
                    "search_space": linear_results.get("param_grid", {}),
                },
                "class_metrics": {
                    class_name: {
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1-score"],
                        "support": metrics["support"],
                    }
                    for class_name, metrics in linear_results["metrics"][
                        "classification_report"
                    ].items()
                    if class_name not in ["accuracy", "macro avg", "weighted avg"]
                },
                "aggregated_metrics": {
                    metric_type: {
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1-score"],
                        "support": metrics["support"],
                    }
                    for metric_type, metrics in linear_results["metrics"][
                        "classification_report"
                    ].items()
                    if metric_type in ["macro avg", "weighted avg"]
                },
                "feature_analysis": {
                    "importance_scores": linear_results["feature_importance"][
                        "coefficients"
                    ],
                    "top_features": get_top_features(
                        linear_results["feature_importance"]["coefficients"], n=10
                    ),
                },
                "roc_analysis": linear_results["metrics"].get("roc_curve", {}),
            },
            "decisionTree": {
                "performance": {
                    "accuracy": tree_results["metrics"]["accuracy"],
                    "cross_validation": {
                        "mean_accuracy": tree_results["cv_scores"]["mean"],
                        "std_accuracy": tree_results["cv_scores"]["std"],
                        "fold_scores": tree_results["cv_scores"]["scores"],
                    },
                },
                "hyperparameters": {
                    "best_params": tree_results["best_params"],
                    "search_space": tree_results.get("param_grid", {}),
                },
                "class_metrics": {
                    class_name: {
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1-score"],
                        "support": metrics["support"],
                    }
                    for class_name, metrics in tree_results["metrics"][
                        "classification_report"
                    ].items()
                    if class_name not in ["accuracy", "macro avg", "weighted avg"]
                },
                "aggregated_metrics": {
                    metric_type: {
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1-score"],
                        "support": metrics["support"],
                    }
                    for metric_type, metrics in tree_results["metrics"][
                        "classification_report"
                    ].items()
                    if metric_type in ["macro avg", "weighted avg"]
                },
                "feature_analysis": {
                    "importance_scores": tree_results["feature_importance"][
                        "importance"
                    ],
                    "top_features": get_top_features(
                        tree_results["feature_importance"]["importance"], n=10
                    ),
                },
                "roc_analysis": tree_results["metrics"].get("roc_curve", {}),
                "tree_structure": {
                    "topology": {
                        "children_left": tree_info["children_left"],
                        "children_right": tree_info["children_right"],
                        "feature_indices": tree_info["feature"],
                    },
                    "node_data": {
                        "thresholds": tree_info["threshold"],
                        "samples": tree_info["n_node_samples"],
                        "impurity": tree_info["impurity"],
                        "values": tree_info["value"],
                    },
                },
            },
        },
    }

    return dashboard_data


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
