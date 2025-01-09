#!/usr/bin/env python3

import os
import json
import csv
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from datetime import datetime


def extract_scores(dashboard_dir, output_csv):
    """
    Traverse the dashboard directory, extract macro avg F1 scores from JSON files,
    and save them to a CSV file. Also handles NaN values or floats in 'binarize_value'
    by converting them to int (default = 0) when parsing.
    """
    entries_processed = 0
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

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
                        if hidden:
                            hidden_value = "Hidden States"
                        else:
                            hidden_value = "SAE Features"

                        # Robustly convert binarize_value to int, defaulting to 0 on failure
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
                        print(f"Error processing {json_path}: {e}")

    print(f"Processed {entries_processed} entries.")
    print(f"Results saved to {output_csv}")


def check_missing_combinations(df, results_dir):
    """
    Check for missing combinations of parameters in the CSV.

    1) Gather all unique values for the relevant columns.
    2) Compute the Cartesian product of these unique values.
    3) Compare with actual combinations in the dataframe.
    4) Print and save missing combinations to CSV.
    """

    columns = [
        "model_name",
        "model_type",
        "sae_location",
        "layer",
        "width",
        "type",
        "dataset_name",
        "top_n",
        "binarize_value",
        "last_token",
    ]

    # 1) Get unique values
    unique_values = {col: df[col].unique() for col in columns}

    # 2) Cartesian product of all unique values
    all_combos = list(itertools.product(*unique_values.values()))

    # 3) Existing combinations in the dataframe
    existing_combos = set(tuple(row) for row in df[columns].to_numpy())

    # 4) Determine which combos are missing
    missing_combos = [combo for combo in all_combos if combo not in existing_combos]

    # Print and save the missing combos
    if len(missing_combos) == 0:
        print("No missing combinations found!")
    else:
        print(f"Found {len(missing_combos)} missing combinations.")
        for mc in missing_combos[:10]:  # Just print first 10 for brevity
            print(mc)

        missing_df = pd.DataFrame(missing_combos, columns=columns)
        output_path = os.path.join(results_dir, "missing_combinations.csv")
        missing_df.to_csv(output_path, index=False)
        print(f"Missing combinations saved to: {output_path}")


def sae_hyperparam_analysis(df, results_dir):
    """
    Perform SAE hyperparameter exploration:
    - Filter for classification datasets of interest if needed.
    - Compare different binarize_value, top_n, etc.
    - Produce grouped summary stats and save results/figures.
    """
    # Prepare subfolders for outputs
    tables_dir = os.path.join(results_dir, "tables")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Filter for SAE Features
    sae_df = df[df["type"] == "SAE Features"].copy()

    # Example grouping by hyperparameters
    summary_cols = ["binarize_value", "top_n", "layer", "width"]
    grouped = (
        sae_df.groupby(summary_cols)["linear_macro_f1_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.rename(columns={"mean": "F1_mean", "std": "F1_std"}, inplace=True)

    # Sort and save top-15 as a CSV table
    grouped_sorted = grouped.sort_values(by="F1_mean", ascending=False).head(15)
    grouped_sorted.to_csv(
        os.path.join(tables_dir, "SAE_hyperparam_top15.csv"), index=False
    )

    # Seaborn styling
    sns.set_theme(style="whitegrid")

    # Figure: Impact of Binarization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sae_df, x="binarize_value", y="linear_macro_f1_score", ci="sd")
    plt.title("SAE Hyperparam: Impact of Binarization (Linear Macro F1 Score)")
    plt.xlabel("Binarize Value")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sae_binarization_impact.png"), dpi=300)
    plt.close()

    # Figure: Impact of top_n
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sae_df, x="top_n", y="linear_macro_f1_score", ci="sd")
    plt.title("SAE Hyperparam: Impact of top_n (Linear Macro F1 Score)")
    plt.xlabel("top_n")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sae_topn_impact.png"), dpi=300)
    plt.close()


def compare_sae_to_baselines(df, results_dir):
    """
    Compare best SAE hyperparameters to baseline methods (Hidden States, TFIDF, etc.)
    using linear_macro_f1_score as the primary metric.
    - Saves summary tables and figures in subfolders under results_dir.
    """
    # Prepare subfolders for outputs
    tables_dir = os.path.join(results_dir, "tables")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Filter to SAE rows
    sae_df = df[df["type"] == "SAE Features"].copy()
    # Identify best SAE hyperparameters per dataset
    best_sae = sae_df.loc[
        sae_df.groupby("dataset_name")["linear_macro_f1_score"].idxmax()
    ].copy()

    # Baseline methods
    baseline_df = df[df["type"].isin(["Hidden States", "TFIDF"])].copy()
    compare_df = pd.concat([best_sae, baseline_df], ignore_index=True)

    # Save best SAE table
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

    # Save comparison (SAE + Baselines)
    compare_df_table = compare_df[
        ["dataset_name", "type", "layer", "linear_macro_f1_score"]
    ].sort_values(["dataset_name", "type"])
    compare_df_table.to_csv(
        os.path.join(tables_dir, "compare_sae_baselines.csv"), index=False
    )

    # Bar plot: Best SAE vs. Baselines by dataset
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=compare_df,
        x="dataset_name",
        y="linear_macro_f1_score",
        hue="type",
        ci="sd",
    )
    plt.title("Best SAE Hyperparameters vs. Baselines (Linear Macro F1 Score)")
    plt.xlabel("Dataset Name")
    plt.ylabel("F1 Score")
    plt.legend(title="Feature Type", loc="best")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sae_vs_baseline_bar.png"), dpi=300)
    plt.close()

    # Merge best hyperparams onto all layers for SAE
    sae_hyperparams = best_sae[
        ["dataset_name", "binarize_value", "top_n", "width"]
    ].drop_duplicates()
    sae_df_merged = pd.merge(
        sae_df,
        sae_hyperparams,
        on=["dataset_name", "binarize_value", "top_n", "width"],
        how="inner",
    )
    layer_compare_df = pd.concat([sae_df_merged, baseline_df], ignore_index=True)

    # Save layer-level comparison
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

    # Line plot: per dataset_name
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
        ci="sd",
        height=4,
        aspect=1.2,
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Layer", "Linear Macro F1 Score")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Layer Analysis: Best SAE Hyperparams vs. Baselines", y=1.05)
    # Save figure from the FacetGrid
    plt.savefig(
        os.path.join(figures_dir, "layer_analysis_sae_vs_baselines.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_multilingual_results(df, results_dir):
    """
    Plot performance for the two multilingual datasets, each with its unique languages:

    - textdetox/multilingual_toxicity_dataset -> language in 'dataset_split'
      (exclude 'test' if needed).
    - cardiffnlp/tweet_sentiment_multilingual -> language in 'dataset_config_name'
      (exclude NaN).

    Saves plots to: results_dir/figures/multilingual_performance.png.
    """

    # Prepare figure output directory
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Datasets of interest
    multi_datasets = [
        "textdetox/multilingual_toxicity_dataset",
        "cardiffnlp/tweet_sentiment_multilingual",
    ]

    # Filter the main DF to just these datasets
    multi_df = df[df["dataset_name"].isin(multi_datasets)].copy()
    print("[DEBUG] After filtering to multilingual datasets:")
    print(f"  Found {len(multi_df)} rows matching {multi_datasets}")
    if multi_df.empty:
        print("  No data found for multilingual datasets. Skipping plot.")
        return

    # Further debug: how many rows per dataset
    for dset in multi_datasets:
        subset_len = len(multi_df[multi_df["dataset_name"] == dset])
        print(f"  - {dset} has {subset_len} rows.")

    # Print unique config and splits in the filtered DF
    unique_config_names = multi_df["dataset_config_name"].unique()
    unique_splits = multi_df["dataset_split"].unique()
    print("[DEBUG] Unique Config Names:", unique_config_names)
    print("[DEBUG] Unique Splits:", unique_splits)

    def extract_language(row):
        """
        Extract the language depending on which dataset it is.
        """
        dname = row["dataset_name"]

        # For textdetox => language is from dataset_split, ignoring 'test'
        if dname == "textdetox/multilingual_toxicity_dataset":
            split_val = row["dataset_split"]
            # If you don't want to exclude 'test', remove this check:
            if split_val == "test":
                return None
            else:
                return split_val

        # For cardiffnlp => language is from dataset_config_name, ignoring NaN
        elif dname == "cardiffnlp/tweet_sentiment_multilingual":
            config_val = row["dataset_config_name"]
            if pd.isna(config_val):
                return None
            else:
                return config_val

        return None  # Default/fallback

    # Apply the logic to generate a "language" column
    multi_df["language"] = multi_df.apply(extract_language, axis=1)

    # How many got assigned language=None?
    n_none = multi_df["language"].isna().sum()
    print(f"[DEBUG] Rows with language=None: {n_none}")

    # Drop rows with None language
    initial_len = len(multi_df)
    multi_df.dropna(subset=["language"], inplace=True)
    after_drop_len = len(multi_df)
    print(
        f"[DEBUG] Dropped {initial_len - after_drop_len} rows where language is None."
    )
    print(f"  Remaining rows in multilingual df: {after_drop_len}")

    if multi_df.empty:
        print("No valid rows left for multilingual plotting. Exiting.")
        return

    # Print some final debug info
    print(
        "[DEBUG] Unique dataset_name after cleaning:", multi_df["dataset_name"].unique()
    )
    print("[DEBUG] Unique language after cleaning:", multi_df["language"].unique())

    # --- Changes for side-by-side subplots start here ---

    # Create a figure with two subplots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True
    )  # sharey for same y-axis scale
    fig.suptitle("Multilingual Results: Performance Across Languages")

    # Plot for each dataset
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
            if i == 0:  # Set y-label only for the first subplot
                axes[i].set_ylabel("Linear Macro F1 Score")
            axes[i].legend(
                title="Type", loc="upper right", bbox_to_anchor=(1.35, 0.8)
            )  # moves legend outside of the plot
        except ValueError as ve:
            print(f"[ERROR] Plotting for {dataset_name} failed with ValueError:")
            print(f"    {ve}")
            print("    Possibly no valid data. Check the dataframe content.")

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(figures_dir, "multilingual_performance.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DEBUG] Multilingual performance figure saved to {output_path}")


def multimodal_eval(df, results_dir):
    """
    Evaluates SAE performance across different settings and layers for multimodal models,
    and generates plots, saving them into a 'multimodal' subfolder within the 'figures' directory.

    Args:
        df: The main dataframe containing all the evaluation results.
        results_dir: The directory to save the generated plots to.
    """

    # Prepare figure output directory
    figures_dir = os.path.join(results_dir, "figures", "multimodal")
    os.makedirs(figures_dir, exist_ok=True)

    # --- Filter data to datasets and model of interest ---
    dataset_of_interest = [
        "nelorth/oxford-flowers",
        "rajistics/indian_food_images",
        "renumics/cifar100-enriched",
    ]
    print(df["dataset_name"].unique())
    df = df[df["dataset_name"].isin(dataset_of_interest)]
    df = df[df["model_name"] == "google/paligemma2-3b-pt-448"]

    # Custom OpenAI-inspired color palette
    openai_palette = [
        "#0DAB76",
        "#33A1DE",
        "#FF6F61",
        "#6B5B95",
        "#88B04B",
        "#F7CAC9",
        "#92A8D1",
    ]

    # Set theme: whitegrid for a clean background, 'talk' context for emphasis, and our custom palette
    sns.set_theme(style="whitegrid", context="talk", palette=openai_palette)

    # Filter for SAE Features only
    sae_df = df[df["type"] == "SAE Features"]

    # Group by dataset, layer, top_n, binarize and compute mean score
    grouped = (
        sae_df.groupby(["dataset_name", "layer", "top_n", "binarize_value"])[
            "linear_macro_f1_score"
        ]
        .mean()
        .reset_index()
    )

    # Create a combined setting description
    grouped["setting"] = grouped.apply(
        lambda r: f"top_n: {r.top_n}, binarize: {r.binarize_value}", axis=1
    )

    # Baseline values for each dataset
    vision_baselines = {
        "nelorth/oxford-flowers": 0.98,
        "rajistics/indian_food_images": 0.98,
        "renumics/cifar100-enriched": 0.79,
    }
    residual_baselines = {
        "nelorth/oxford-flowers": 0.95,
        "rajistics/indian_food_images": 0.90,
        "renumics/cifar100-enriched": 0.75,
    }

    datasets = grouped["dataset_name"].unique()

    for dataset in datasets:
        dataset_data = grouped[grouped["dataset_name"] == dataset]

        # High resolution figure suitable for conference (300 dpi)
        plt.figure(figsize=(10, 7), dpi=300)

        # Plot lines for each combination of top_n and binarization
        ax = sns.lineplot(
            data=dataset_data,
            x="layer",
            y="linear_macro_f1_score",
            hue="setting",
            style="setting",
            markers=True,
            dashes=False,
            linewidth=2.5,
            markersize=8,
        )

        # Add baseline lines
        vb = vision_baselines.get(dataset)
        rb = residual_baselines.get(dataset)

        if vb is not None:
            plt.axhline(
                y=vb,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Vision Tower Best ({vb:.2f})",
            )

        if rb is not None:
            plt.axhline(
                y=rb,
                color="#FF6F61",
                linestyle=":",
                linewidth=2,
                label=f"Residual Stream Probing Best ({rb:.2f})",
            )

        # Titles and labels with larger fonts for conference clarity
        # Clean dataset name for plot title (replace slashes with underscores)
        cleaned_dataset_name = dataset.replace("/", "_")
        plt.title(
            f"Linear Macro F1 Score across Layers for {cleaned_dataset_name}",
            fontsize=22,
            weight="bold",
        )
        plt.xlabel("Layer", fontsize=18)
        plt.ylabel("Linear Macro F1 Score", fontsize=18)

        # Customize tick parameters for readability
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.3)

        # Manage legend outside plot area for clarity
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            title="Setting",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=14,
            title_fontsize=16,
        )

        sns.despine(trim=True)  # Remove top and right spines for a cleaner look
        plt.tight_layout()

        # Save the figure with cleaned dataset name in the path
        output_path = os.path.join(
            figures_dir, f"{cleaned_dataset_name}_sae_performance.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[DEBUG] SAE performance figure saved to {output_path}")


def main():
    # Load configuration
    with open("config_jg.yaml", "r") as f:
        config = yaml.safe_load(f)
    dashboard_dir = config["settings"]["base_classify_dir"]

    # Check if classify_scores.csv already exists
    output_csv = os.path.join(dashboard_dir, "classify_scores.csv")
    if not os.path.exists(output_csv):
        extract_scores(dashboard_dir, output_csv)
    else:
        print(f"{output_csv} already exists. Skipping extraction.")

    # Make results folder: output/results
    results_dir = os.path.join("output", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load into DataFrame
    df = pd.read_csv(output_csv)
    df_original = df.copy()

    print("\nDataFrame Info:")
    print(df.info())

    # Check for missing combinations
    check_missing_combinations(df, results_dir)

    # SAE Hyperparameter Analysis
    sae_hyperparam_analysis(df, results_dir)

    # Compare Best SAE to Baselines
    compare_sae_to_baselines(df, results_dir)

    # Multilingual eval
    plot_multilingual_results(df, results_dir)

    # multimodal eval
    multimodal_eval(df, results_dir)


if __name__ == "__main__":
    main()


## TODO
# shans multilingual heatmap transfer/ line plot
# shan upload his results
