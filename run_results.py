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


from src.results_scripts.extract_utils import extract_scores
from src.results_scripts.missing_utils import check_missing_combinations
from src.results_scripts.pooling_utils import plot_pooling_strategies
from src.results_scripts.scale_utils import plot_sae_across_models_and_widths_by_dataset
from src.results_scripts.image_utils import (
    plot_dataset_width_sae,
)
from src.results_scripts.action_utils import (
    plot_action_across_models_and_widths_by_dataset,
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


def clean_dataset_name(dataset_name):
    """Cleans the dataset name for saving figures."""
    return dataset_name.replace("/", "_").replace(" ", "_")


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

    # 4. Action prediction
    action_models_of_interest = [
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-9b-it",
    ]
    action_widths_of_interest = ["16k", "131k", "1m"]

    plot_action_across_models_and_widths_by_dataset(
        df, results_dir, action_models_of_interest, action_widths_of_interest
    )


if __name__ == "__main__":
    main()
