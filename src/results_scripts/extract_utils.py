import json
import os
import logging
import csv
import pandas as pd


def extract_scores(dashboard_dir: str, output_csv: str) -> None:
    """
    Traverses the dashboard directory (dashboard_dir), extracts macro avg F1 scores
    from JSON files, and saves them to a CSV file (output_csv).

    Also handles NaN values or floats in 'binarize_value' by converting them to int
    (default = 0) when parsing.

    Args:
        dashboard_dir: Directory containing JSON files with model metrics.
        output_csv: Output CSV file path.
    Returns:
        None. Creates/updates the CSV file with extracted scores.
    """
    entries_processed = 0
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Using 'with' to open CSV in write-mode
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "timestamp",
                "model_name",
                "model_type",
                "sae_location",
                "layer",
                "width",
                "type",
                "dataset_name",
                "dataset_config_name",
                "dataset_split",
                "top_n",
                "binarize_value",
                "last_token",
                "linear_macro_f1_score",
                "linear_accuracy",
                "linear_mean_cv_accuracy",
                "linear_std_cv_accuracy",
                "decision_tree_macro_f1_score",
                "decision_tree_accuracy",
                "decision_tree_mean_cv_accuracy",
                "decision_tree_std_cv_accuracy",
            ]
        )

        # Walk through all JSON files under 'dashboard_dir'
        for root, _, files in os.walk(dashboard_dir):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)

                        metadata = data.get("metadata", {})
                        models = data.get("models", {})

                        timestamp = metadata.get("timestamp", "N/A")
                        model_name = metadata.get("model", {}).get("name", "N/A")
                        layer = metadata.get("model", {}).get("layer", "N/A")

                        args = metadata.get("args", {})
                        width = args.get("width", "N/A")
                        dataset_name = metadata.get("dataset", {}).get("name", "N/A")
                        dataset_split = args.get("dataset_split", "N/A")

                        model_type = args.get("model_type", "N/A")
                        sae_location = args.get("sae_location", "N/A")
                        dataset_config_name = args.get("dataset_config_name", "N/A")
                        top_n = args.get("top_n", "N/A")
                        binarize_value = args.get("binarize_value", "N/A")
                        last_token = args.get("last_token", "N/A")

                        hidden = metadata.get("dataset", {}).get("hidden", "N/A")
                        # Distinguish "Hidden States" vs. "SAE Features"
                        if hidden:
                            hidden_value = "Hidden States"
                        else:
                            hidden_value = "SAE Features"

                        # Robustly convert binarize_value to int, default=0 on failure
                        if binarize_value == "N/A" or pd.isna(binarize_value):
                            binarize_value = 0
                        else:
                            try:
                                binarize_value = int(binarize_value)
                            except ValueError:
                                try:
                                    binarize_value = int(float(binarize_value))
                                except Exception:
                                    binarize_value = 0

                        # Extract Linear Probe metrics
                        linear_probe = models.get("linearProbe", {})
                        linear_macro_avg = linear_probe.get(
                            "aggregated_metrics", {}
                        ).get("macro avg", {})
                        linear_f1_score = linear_macro_avg.get("f1_score", "N/A")

                        linear_performance = linear_probe.get("performance", {})
                        linear_accuracy = linear_performance.get("accuracy", "N/A")
                        linear_cv = linear_performance.get("cross_validation", {})
                        linear_mean_cv_accuracy = linear_cv.get("mean_accuracy", "N/A")
                        linear_std_cv_accuracy = linear_cv.get("std_accuracy", "N/A")

                        # Extract Decision Tree metrics
                        decision_tree = models.get("decisionTree", {})
                        decision_tree_macro_avg = decision_tree.get(
                            "aggregated_metrics", {}
                        ).get("macro avg", {})
                        decision_tree_f1_score = decision_tree_macro_avg.get(
                            "f1_score", "N/A"
                        )

                        decision_tree_performance = decision_tree.get("performance", {})
                        decision_tree_accuracy = decision_tree_performance.get(
                            "accuracy", "N/A"
                        )
                        decision_tree_cv = decision_tree_performance.get(
                            "cross_validation", {}
                        )
                        decision_tree_mean_cv_accuracy = decision_tree_cv.get(
                            "mean_accuracy", "N/A"
                        )
                        decision_tree_std_cv_accuracy = decision_tree_cv.get(
                            "std_accuracy", "N/A"
                        )

                        csv_writer.writerow(
                            [
                                timestamp,
                                model_name,
                                model_type,
                                sae_location,
                                layer,
                                width,
                                hidden_value,
                                dataset_name,
                                dataset_config_name,
                                dataset_split,
                                top_n,
                                binarize_value,
                                last_token,
                                linear_f1_score,
                                linear_accuracy,
                                linear_mean_cv_accuracy,
                                linear_std_cv_accuracy,
                                decision_tree_f1_score,
                                decision_tree_accuracy,
                                decision_tree_mean_cv_accuracy,
                                decision_tree_std_cv_accuracy,
                            ]
                        )
                        entries_processed += 1

                    except Exception as e:
                        logging.error(
                            f"Error processing {json_path}: {e}", exc_info=True
                        )

    logging.info(f"Processed {entries_processed} entries.")
    logging.info(f"Results saved to {output_csv}")
