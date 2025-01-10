#!/usr/bin/env python3

import os
import json
import csv
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import logging
from datetime import datetime
from typing import Optional, List, Tuple
from pandas.api.types import CategoricalDtype


from src.results_scripts.missing_utils import check_missing_combinations
from src.results_scripts.pooling_utils import plot_pooling_strategies
from src.results_scripts.scale_utils import plot_sae_across_models_and_widths_by_dataset
from src.results_scripts.image_utils import (
    plot_dataset_width_sae,
)


# -------------------------------------------------------------------------
# 1) CONFIGURE LOGGING
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def prepare_output_dirs(results_dir: str) -> Tuple[str, str]:
    """
    Creates 'tables' and 'figures' subdirectories under results_dir if they do not exist.
    Returns a tuple of (tables_dir, figures_dir).
    """
    tables_dir = os.path.join(results_dir, "tables")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return tables_dir, figures_dir


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


def generate_expected_results(config):
    """
    Generates a list of dictionaries representing the expected results
    based on the YAML configuration.
    """

    models = config["models"]
    datasets = config["datasets"]
    classification_params = config["classification_params"]
    settings = config["settings"]

    expected_results = []

    for model in models:
        for layer in model["layers"]:
            for width in model["widths"]:
                for dataset in datasets:
                    for top_n in classification_params["top_n_values"]:
                        for binarize in classification_params["binarize_values"]:
                            # We still iterate these 'raw' labels,
                            # but we will map them below
                            for hidden_state_type in ["baseline", "sae"]:
                                # If act_only is True, skip some classification params
                                if settings["act_only"] == True and (
                                    top_n != classification_params["top_n_values"][0]
                                    or binarize
                                    != classification_params["binarize_values"][0]
                                ):
                                    continue

                                # Map 'baseline' -> 'Hidden States'
                                # and 'sae' -> 'SAE Features'
                                if hidden_state_type == "baseline":
                                    mapped_type = "Hidden States"
                                else:
                                    mapped_type = "SAE Features"

                                expected_results.append(
                                    {
                                        "model_name": model["name"],
                                        "model_type": settings["model_type"],
                                        "sae_location": settings["sae_location"],
                                        "layer": layer,
                                        "width": width,
                                        "type": mapped_type,
                                        "dataset_name": dataset["name"],
                                        "dataset_config_name": dataset["config_name"],
                                        "dataset_split": dataset["split"],
                                        "top_n": top_n,
                                        "binarize_value": binarize,
                                        "last_token": True,
                                    }
                                )

    return expected_results


# def check_missing_combinations(
#     df: pd.DataFrame, results_dir: str, config_path: str
# ) -> None:
#     """
#     Checks for missing combinations of parameters in the DataFrame, comparing
#     against a YAML configuration file, and identifies missing results.

#     Saves a CSV file with missing combinations if any.

#     Args:
#         df: The DataFrame containing all results.
#         results_dir: Directory to save missing_combinations.csv if missing combos exist.
#         config_path: Path to the YAML configuration file.
#     Returns:
#         None
#     """
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)

#     df_expected = pd.DataFrame(generate_expected_results(config))

#     # Ensure consistent data types for merging
#     df_expected["layer"] = df_expected["layer"].astype(str)
#     df_expected["binarize_value"] = (
#         df_expected["binarize_value"].fillna(0).astype("int64")
#     )  # Fill NA with 0 and cast to int64
#     df["layer"] = df["layer"].astype(str)
#     df["binarize_value"] = (
#         df["binarize_value"].fillna(0).astype("int64")
#     )  # Fill NA with 0 and cast to int64

#     # Merge the expected results with the actual results
#     merged_df = pd.merge(
#         df_expected,
#         df,
#         on=[
#             "model_name",
#             "model_type",
#             "sae_location",
#             "layer",
#             "width",
#             "type",
#             "dataset_name",
#             "top_n",
#             "binarize_value",
#             "last_token",
#         ],
#         how="left",
#         indicator=True,
#     )

#     # Identify missing combinations
#     missing_combos = merged_df[merged_df["_merge"] == "left_only"].drop(
#         columns=["_merge"]
#     )

#     if missing_combos.empty:
#         logging.info("No missing combinations found!")
#     else:
#         logging.info(f"Found {len(missing_combos)} missing combinations.")
#         output_path = os.path.join(results_dir, "missing_combinations.csv")
#         missing_combos.to_csv(output_path, index=False)
#         logging.info(f"Missing combinations saved to: {output_path}")

#         # Summarize missing combinations
#         summary = (
#             missing_combos.groupby(["model_name", "dataset_name"])
#             .size()
#             .reset_index(name="missing_count")
#         )

#         print("\nSummary of Missing Combinations:")
#         for index, row in summary.iterrows():
#             print(
#                 f"Model: {row['model_name']}, Dataset: {row['dataset_name']} - Missing: {row['missing_count']}"
#             )

#         summary_output_path = os.path.join(
#             results_dir, "missing_combinations_summary.csv"
#         )
#         summary.to_csv(summary_output_path, index=False)
#         logging.info(f"Summary of missing combinations saved to: {summary_output_path}")


def sae_hyperparam_analysis(df: pd.DataFrame, results_dir: str) -> None:
    """
    Perform SAE hyperparameter exploration. Generates summary statistics and
    bar plots showing the impact of binarize_value and top_n.

    Args:
        df: DataFrame containing classification results.
        results_dir: Base directory to save tables and figures.
    """
    tables_dir, figures_dir = prepare_output_dirs(results_dir)

    # Filter for SAE Features
    sae_df = df[df["type"] == "SAE Features"].copy()

    # Group by some hyperparams
    summary_cols = ["binarize_value", "top_n", "layer", "width"]
    grouped = (
        sae_df.groupby(summary_cols)["linear_macro_f1_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.rename(columns={"mean": "F1_mean", "std": "F1_std"}, inplace=True)

    grouped_sorted = grouped.sort_values(by="F1_mean", ascending=False).head(15)
    grouped_sorted.to_csv(
        os.path.join(tables_dir, "SAE_hyperparam_top15.csv"), index=False
    )
    logging.info(
        f"SAE hyperparam analysis saved top-15 to {tables_dir}/SAE_hyperparam_top15.csv"
    )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sae_df, x="binarize_value", y="linear_macro_f1_score", ci="sd")
    plt.title("SAE Hyperparam: Impact of Binarization (Linear Macro F1 Score)")
    plt.xlabel("Binarize Value")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sae_binarization_impact.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sae_df, x="top_n", y="linear_macro_f1_score", ci="sd")
    plt.title("SAE Hyperparam: Impact of top_n (Linear Macro F1 Score)")
    plt.xlabel("top_n")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sae_topn_impact.png"), dpi=300)
    plt.close()

    logging.info("SAE hyperparameter analysis completed and figures generated.")


def compare_sae_to_baselines(df: pd.DataFrame, results_dir: str) -> None:
    """
    Compare best SAE hyperparameters per dataset to baseline methods (e.g., Hidden States),
    using the linear_macro_f1_score as the primary metric.  This version specifically
    addresses the prompt's concerns about potential information loss during merging and
    hyperparameter selection.

    Args:
        df: DataFrame containing classification results.
        results_dir: Base directory to save tables and figures.
    """
    tables_dir, figures_dir = prepare_output_dirs(results_dir)

    # Filter to SAE rows
    sae_df = df[df["type"] == "SAE Features"].copy()

    # **1. Identify best SAE hyperparameters per dataset per layer**
    # This ensures we don't lose layer-specific performance during hyperparameter selection
    best_sae_per_layer = sae_df.loc[
        sae_df.groupby(["dataset_name", "layer"])["linear_macro_f1_score"].idxmax()
    ].copy()

    # **2. Find the overall best layer per dataset**
    best_layer_per_dataset = best_sae_per_layer.loc[
        best_sae_per_layer.groupby("dataset_name")["linear_macro_f1_score"].idxmax()
    ][["dataset_name", "layer"]].copy()

    # **3. Merge to get the best hyperparameters for the best layer**
    best_sae = pd.merge(
        best_layer_per_dataset,
        best_sae_per_layer,
        on=["dataset_name", "layer"],
        how="left",
    )

    # Baseline methods (Hidden States, TFIDF, etc.)
    baseline_df = df[df["type"].isin(["Hidden States", "TFIDF"])].copy()

    # **4. Find best layer for Hidden States**
    best_hidden_states = (
        baseline_df[baseline_df["type"] == "Hidden States"]
        .loc[
            baseline_df[baseline_df["type"] == "Hidden States"]
            .groupby("dataset_name")["linear_macro_f1_score"]
            .idxmax()
        ]
        .copy()
    )

    # **5. Recombine best_sae, best_hidden_states, and tfidf for comparison**
    compare_df = pd.concat(
        [best_sae, best_hidden_states, baseline_df[baseline_df["type"] == "TFIDF"]],
        ignore_index=True,
    )

    best_sae_table = best_sae[
        [
            "dataset_name",
            "layer",
            "width",
            "top_n",
            "binarize_value",
            "linear_macro_f1_score",
        ]
    ].sort_values("dataset_name")
    best_sae_table.to_csv(os.path.join(tables_dir, "best_sae_table.csv"), index=False)

    compare_df_table = compare_df[
        ["dataset_name", "type", "layer", "linear_macro_f1_score"]
    ].sort_values(["dataset_name", "type"])
    compare_df_table.to_csv(
        os.path.join(tables_dir, "compare_sae_baselines.csv"), index=False
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=compare_df,
        x="dataset_name",
        y="linear_macro_f1_score",
        hue="type",
        errorbar="sd",
    )
    plt.title("Best SAE Hyperparameters vs. Baselines (Linear Macro F1 Score)")
    plt.xlabel("Dataset Name")
    plt.ylabel("F1 Score")
    plt.legend(title="Feature Type", loc="best")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sae_vs_baseline_bar.png"), dpi=300)
    plt.close()

    # **6. Merge best hyperparams onto all layers for SAE, keeping layer information**
    # Here we merge based on dataset_name and the hyperparameter set
    # We DO NOT merge on 'layer' to retain all layer information
    sae_hyperparams = best_sae[
        ["dataset_name", "binarize_value", "top_n", "width"]
    ].drop_duplicates()
    sae_df_merged = pd.merge(
        sae_df,
        sae_hyperparams,
        on=["dataset_name", "binarize_value", "top_n", "width"],
        how="inner",
    )

    # **7. Layer-wise Comparison**

    # Correctly handle the 'layer' column for Hidden States if it's missing
    if "layer" not in baseline_df.columns or baseline_df["layer"].isnull().all():
        # If 'layer' is entirely missing or NaN for all Hidden States, add a placeholder
        baseline_df.loc[baseline_df["type"] == "Hidden States", "layer"] = (
            best_hidden_states["layer"].unique()[0]
            if "layer" in best_hidden_states.columns
            else -1
        )

    # Ensure 'layer' is of a compatible type for merging
    sae_df_merged["layer"] = sae_df_merged["layer"].astype(str)
    baseline_df["layer"] = baseline_df["layer"].astype(str)

    layer_compare_df = pd.concat([sae_df_merged, baseline_df], ignore_index=True)

    layer_compare_df_table = layer_compare_df[
        [
            "dataset_name",
            "type",
            "layer",
            "binarize_value",
            "top_n",
            "width",
            "linear_macro_f1_score",
        ]
    ].sort_values(["dataset_name", "type", "layer"])
    layer_compare_df_table.to_csv(
        os.path.join(tables_dir, "layer_compare_sae_baselines.csv"), index=False
    )

    g = sns.relplot(
        data=layer_compare_df,
        x="layer",
        y="linear_macro_f1_score",
        hue="type",
        kind="line",
        col="dataset_name",
        col_wrap=3,
        markers=True,
        dashes=False,
        errorbar="sd",
        height=4,
        aspect=1.2,
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Layer", "Linear Macro F1 Score")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Layer Analysis: Best SAE Hyperparams vs. Baselines", y=1.05)

    output_path = os.path.join(figures_dir, "layer_analysis_sae_vs_baselines.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info("Comparison of SAE vs. baselines completed.")


def plot_multilingual_results(df: pd.DataFrame, results_dir: str) -> None:
    """
    Plot performance for specific multilingual datasets and their unique languages.
    Excludes 'test' splits for textdetox and excludes NaNs for cardiffnlp.

    Args:
        df: DataFrame containing classification results.
        results_dir: Base directory to save figures.
    """
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    multi_datasets = [
        "textdetox/multilingual_toxicity_dataset",
        "cardiffnlp/tweet_sentiment_multilingual",
    ]
    multi_df = df[df["dataset_name"].isin(multi_datasets)].copy()
    logging.info("[DEBUG] After filtering to multilingual datasets:")
    logging.info(f"  Found {len(multi_df)} rows matching {multi_datasets}")

    if multi_df.empty:
        logging.info("  No data found for multilingual datasets. Skipping plot.")
        return

    for dset in multi_datasets:
        subset_len = len(multi_df[multi_df["dataset_name"] == dset])
        logging.debug(f"  - {dset} has {subset_len} rows.")

    def extract_language(row):
        """
        Extract the language depending on which dataset it is from.
        Returns None if it doesn't match expected criteria (test split, missing config).
        """
        dname = row["dataset_name"]
        if dname == "textdetox/multilingual_toxicity_dataset":
            split_val = row["dataset_split"]
            if split_val == "test":
                return None
            else:
                return split_val
        elif dname == "cardiffnlp/tweet_sentiment_multilingual":
            config_val = row["dataset_config_name"]
            if pd.isna(config_val):
                return None
            else:
                return config_val
        return None

    multi_df["language"] = multi_df.apply(extract_language, axis=1)

    n_none = multi_df["language"].isna().sum()
    logging.debug(f"[DEBUG] Rows with language=None: {n_none}")

    initial_len = len(multi_df)
    multi_df.dropna(subset=["language"], inplace=True)
    after_drop_len = len(multi_df)
    logging.debug(
        f"[DEBUG] Dropped {initial_len - after_drop_len} rows where language is None."
    )
    logging.info(f"  Remaining rows in multilingual df: {after_drop_len}")

    if multi_df.empty:
        logging.info("No valid rows left for multilingual plotting. Exiting.")
        return

    logging.debug(
        "[DEBUG] Unique dataset_name after cleaning: %s",
        multi_df["dataset_name"].unique(),
    )
    logging.debug(
        "[DEBUG] Unique language after cleaning: %s", multi_df["language"].unique()
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Multilingual Results: Performance Across Languages")

    for i, dataset_name in enumerate(multi_datasets):
        dataset_df = multi_df[multi_df["dataset_name"] == dataset_name]
        try:
            sns.barplot(
                data=dataset_df,
                x="language",
                y="linear_macro_f1_score",
                hue="type",
                palette="rocket",
                alpha=0.85,
                ci="sd",
                ax=axes[i],
            )
            axes[i].set_title(dataset_name)
            axes[i].set_xlabel("Language")
            if i == 0:
                axes[i].set_ylabel("Linear Macro F1 Score")
            else:
                axes[i].set_ylabel("")
            axes[i].legend(title="Type", loc="upper right", bbox_to_anchor=(1.35, 0.8))
        except ValueError as ve:
            logging.error(
                f"[ERROR] Plotting for {dataset_name} failed with ValueError: {ve}",
                exc_info=True,
            )

    plt.tight_layout()
    output_path = os.path.join(figures_dir, "multilingual_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"[DEBUG] Multilingual performance figure saved to {output_path}")


def clean_dataset_name(dataset_name):
    """Cleans the dataset name for saving figures."""
    return dataset_name.replace("/", "_").replace(" ", "_")


def evaluate_action_prediction(df: pd.DataFrame, results_dir: str) -> None:
    """
    Evaluates action prediction performance of SAEs on multiple MCQ-like datasets.
    Generates subplots for each original dataset, showing performance across layers.

    Args:
        df: DataFrame containing evaluation results.
        results_dir: Base directory to save figures.
    """
    figures_dir = os.path.join(results_dir, "figures", "action_prediction")
    os.makedirs(figures_dir, exist_ok=True)

    datasets_of_interest = [
        "gallifantjack/pminervini_NQ_Swap_sub_answer_question_openai_google_gemma_2_9b_it",
        "AIM-Harvard/google_gemma_2_9b_it_pubmed_qa",
        "gallifantjack/pminervini_NQ_Swap_org_answer_question_openai_google_gemma_2_9b_it",
        "AIM-Harvard/google_gemma_2_2b_pubmed_qa",
        "gallifantjack/pminervini_NQ_Swap_org_answer_None_openai_google_gemma_2_9b_it",
        "AIM-Harvard/google_gemma_2_9b_pubmed_qa",
    ]
    df = df[df["dataset_name"].isin(datasets_of_interest)]
    df = df[df["type"] == "SAE Features"]

    def extract_original_dataset(dataset_name: str) -> str:
        if "pubmed_qa" in dataset_name:
            return "pubmed_qa"
        elif "pminervini_NQ_Swap" in dataset_name:
            return "pminervini_NQ_Swap"
        else:
            return "unknown"

    df["original_dataset"] = df["dataset_name"].apply(extract_original_dataset)

    unique_models = df["model_name"].unique()
    unique_original_datasets = df["original_dataset"].unique()

    sns.set_theme(style="whitegrid", context="talk")

    for original_dataset in unique_original_datasets:
        dataset_df = df[df["original_dataset"] == original_dataset]
        dataset_df["setting"] = dataset_df.apply(
            lambda r: f"top_n: {r.top_n}, binarize: {r.binarize_value}", axis=1
        )
        unique_settings = dataset_df["setting"].unique()

        num_settings = len(unique_settings)
        num_models = len(unique_models)
        fig, axes = plt.subplots(
            num_models,
            num_settings,
            figsize=(4 * num_settings, 4 * num_models),
            sharey=True,
        )

        # If only one setting or model, ensure 'axes' is 2D
        if num_settings == 1 and num_models == 1:
            axes = [[axes]]
        elif num_settings == 1:
            axes = [axes]
        elif num_models == 1:
            axes = [[ax] for ax in axes]

        for r, model in enumerate(unique_models):
            for c, setting in enumerate(unique_settings):
                model_setting_df = dataset_df[
                    (dataset_df["model_name"] == model)
                    & (dataset_df["setting"] == setting)
                ]
                ax = axes[r][c]

                if not model_setting_df.empty:
                    cleaned_model_name = model.replace("/", "_").replace("google_", "")

                    sns.lineplot(
                        data=model_setting_df,
                        x="layer",
                        y="linear_macro_f1_score",
                        label=cleaned_model_name,
                        marker="o",
                        ax=ax,
                    )
                    ax.set_title(f"{setting}", fontsize=8)
                    ax.set_xlabel("Layer", fontsize=8)
                    if c == 0:
                        ax.set_ylabel(
                            f"{cleaned_model_name}\nLinear Macro F1 Score", fontsize=8
                        )
                    else:
                        ax.set_ylabel("")
                    ax.tick_params(axis="both", which="major", labelsize=6)
                    ax.grid(True)
                    ax.get_legend().remove()

        # Add a single legend
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper right", bbox_to_anchor=(1.15, 0.95), fontsize=6
        )

        plt.tight_layout()

        output_path = os.path.join(
            figures_dir, f"action_prediction_{original_dataset}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(
            f"[DEBUG] Action prediction performance figure for {original_dataset} saved to {output_path}"
        )


def main() -> None:
    """
    Main entry point. Loads YAML config, extracts scores if needed, and runs selected analyses.
    """
    try:
        with open("config_jg.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("config_jg.yaml not found. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading config_jg.yaml: {e}", exc_info=True)
        return

    classifications_dir = os.path.join("output", "classifications")
    results_dir = os.path.join("output", "results")
    os.makedirs(results_dir, exist_ok=True)

    output_csv = os.path.join(results_dir, "classify_scores.csv")

    if not os.path.exists(output_csv):
        logging.info("No classify_scores.csv found, extracting new scores...")
        extract_scores(classifications_dir, output_csv)
    else:
        logging.info(f"{output_csv} already exists. Skipping extraction.")

    df = pd.read_csv(output_csv)
    logging.info("\nDataFrame Info:")
    logging.info(df.info())

    logging.info(f"Dataset Name Unique Values: {df['dataset_name'].unique()}")

    check_missing_combinations(df, results_dir)
    # sae_hyperparam_analysis(df, results_dir)
    # compare_sae_to_baselines(df, results_dir)
    # plot_multilingual_results(df, results_dir)

    # 1. Pooling strategies
    pooling_datasets_list = [
        "AIM-Harvard/reject_prompts",
        "jackhhao/jailbreak-classification",
        "willcb/massive-intent",
        "willcb/massive-scenario",
        "legacy-datasets/banking77",
        "SetFit/tweet_eval_stance_abortion",
    ]

    pooling_widths_of_interest = ["16k", "65k"]

    plot_pooling_strategies(
        df=df,
        results_dir=results_dir,
        pooling_datasets_list=pooling_datasets_list,
        pooling_widths_of_interest=pooling_widths_of_interest,
        score_col="linear_macro_f1_score",
    )

    # 2. Scale across size
    scale_datasets_of_interest = [
        "Anthropic/election_questions",
        "dair-ai/emotion",
        "willcb/massive-scenario",
    ]

    scale_models_of_interest = ["google/gemma-2-9b", "google/gemma-2-9b-it"]
    scale_widths_of_interest = ["16k", "131k", "1m"]

    # 2.1 for each dataset plot performance across layers across model size and sae widths
    plot_sae_across_models_and_widths_by_dataset(
        df,
        results_dir,
        scale_datasets_of_interest,
        scale_models_of_interest,
        scale_widths_of_interest,
    )

    # 3. Image Results
    image_datasets_of_interest = [
        "nelorth/oxford-flowers",
        "rajistics/indian_food_images",
        "renumics/cifar100-enriched",
    ]
    image_widths_of_interest = ["16k"]

    # 3.1 Per-dataset + per-width SAE performance plots
    plot_dataset_width_sae(
        df, results_dir, image_datasets_of_interest, image_widths_of_interest
    )

    # evaluate_action_prediction(df, results_dir)


if __name__ == "__main__":
    main()
