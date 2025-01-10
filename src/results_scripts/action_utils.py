# action_utils.py

import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas.api.types import CategoricalDtype

###############################################################################
# Reuse or re-define these utility functions if they're not already imported. #
###############################################################################


def layer_group(layer: int) -> str:
    """Maps layer index to 'early', 'middle', or 'late'."""
    if layer < 10:
        return "early"
    elif layer == 19 or layer > 20:
        return "late"
    else:
        return "middle"


def clean_dataset_name(dataset_name: str) -> str:
    """Cleans dataset name for filenames."""
    return dataset_name.replace("/", "_").replace(" ", "_")


###############################################################################
# Main plotting function for action-prediction datasets.
###############################################################################


def plot_action_across_models_and_widths_by_dataset(
    df: pd.DataFrame,
    results_dir: str,
    models_of_interest: list,
    widths_of_interest: list,
) -> None:
    """
    Similar to plot_sae_across_models_and_widths_by_dataset, but for action-
    prediction datasets. We focus on the runs listed below. For each dataset in
    this set, create a figure showing performance (e.g., linear_macro_f1_score)
    for every (model, width) combination across early/middle/late layers.

    Args:
        df: DataFrame containing columns like:
            'dataset_name', 'model_name', 'width', 'layer', 'type',
            'linear_macro_f1_score', etc.
        results_dir: Base directory for saving figures.
        models_of_interest: Which models to include in each dataset's figure.
        widths_of_interest: Which widths to include in each dataset's figure.
    """

    ###########################################################################
    # 1) Specify the action-prediction datasets (runs) we want to plot.
    ###########################################################################
    runs_of_interest = [
        # pminervini NQ Swap
        "gallifantjack/pminervini_NQ_Swap_org_answer_None_openai_google_gemma_2_9b_it",
        "gallifantjack/pminervini_NQ_Swap_org_answer_question_openai_google_gemma_2_9b_it",
        "gallifantjack/pminervini_NQ_Swap_sub_answer_question_openai_google_gemma_2_9b_it",
        # google_gemma_2_xb_piqa
        "AIM-Harvard/google_gemma_2_2b_piqa",
        "AIM-Harvard/google_gemma_2_9b_piqa",
        "AIM-Harvard/google_gemma_2_9b_it_piqa",
        # google_gemma_2_xb_boolq
        "AIM-Harvard/google_gemma_2_2b_boolq",
        "AIM-Harvard/google_gemma_2_9b_boolq",
        "AIM-Harvard/google_gemma_2_9b_it_boolq",
        # google_gemma_2_xb_pubmed_qa
        "AIM-Harvard/google_gemma_2_2b_pubmed_qa",
        "AIM-Harvard/google_gemma_2_9b_pubmed_qa",
        "AIM-Harvard/google_gemma_2_9b_it_pubmed_qa",
    ]

    # Create subdirectory for these per-dataset plots
    fig_dir = os.path.join(results_dir, "figures", "action_scale_model_width")
    os.makedirs(fig_dir, exist_ok=True)

    # Define an OpenAI-inspired color palette and set up Seaborn
    openai_palette = [
        "#0DAB76",  # green
        "#33A1DE",  # blue
        "#FF6F61",  # red
        "#6B5B95",  # purple
        "#88B04B",  # lime-ish
        "#F7CAC9",  # pink
        "#92A8D1",  # light periwinkle
    ]
    sns.set_theme(style="whitegrid", context="talk", palette=openai_palette)

    # Optionally, map each model to a color
    model_colors = {
        model: openai_palette[i % len(openai_palette)]
        for i, model in enumerate(models_of_interest)
    }

    # Optionally, map each width to a distinct linestyle or marker
    width_linestyles = {
        "16k": (0, (5, 2)),  # dashed
        "131k": "-",  # solid
        "1m": (0, (1, 1)),  # dotted
    }
    width_markers = {
        "16k": "o",
        "131k": "s",
        "1m": "^",
    }

    ###########################################################################
    # 2) Filter the DataFrame for only the runs we're interested in.
    #    (Assuming 'dataset_name' holds these run strings.)
    ###########################################################################
    df_action = df[df["dataset_name"].isin(runs_of_interest)].copy()

    # Force `layer_group` in the DataFrame
    df_action["layer_group"] = df_action["layer"].astype(int).apply(layer_group)

    ###########################################################################
    # 3) Group (and average) the data as appropriate.
    #    We assume we're plotting "linear_macro_f1_score" for type="SAE Features"
    #    or some "action" type. If needed, adjust the filtering below.
    ###########################################################################
    grouped_df = (
        df_action[df_action["type"] == "SAE Features"]  # or adjust if needed
        .groupby(["dataset_name", "model_name", "width", "layer_group"])[
            "linear_macro_f1_score"
        ]
        .mean()
        .reset_index()
    )

    # Ensure consistent category ordering: early, middle, late
    layer_cat_type = CategoricalDtype(
        categories=["early", "middle", "late"], ordered=True
    )
    grouped_df["layer_group"] = grouped_df["layer_group"].astype(layer_cat_type)

    ###########################################################################
    # 4) Iterate over the runs_of_interest (datasets) and plot each one.
    ###########################################################################
    for ds in runs_of_interest:
        # Subset to the one dataset, plus only the models/widths we want
        subset = grouped_df[
            (grouped_df["dataset_name"] == ds)
            & (grouped_df["model_name"].isin(models_of_interest))
            & (grouped_df["width"].isin(widths_of_interest))
        ]

        if subset.empty:
            logging.warning(f"[WARN] No data found for dataset='{ds}'. Skipping plot.")
            continue

        # Create figure
        plt.figure(figsize=(8, 5), dpi=300)

        # Plot a line for each (model, width) combination
        for model in models_of_interest:
            for width_val in widths_of_interest:
                line_data = subset[
                    (subset["model_name"] == model) & (subset["width"] == width_val)
                ].copy()
                if line_data.empty:
                    continue

                # Sort by layer group or reindex by [early, middle, late]
                line_data = (
                    line_data.set_index("layer_group")
                    .reindex(["early", "middle", "late"])
                    .reset_index()
                )

                plt.plot(
                    line_data["layer_group"],
                    line_data["linear_macro_f1_score"],
                    color=model_colors[model],
                    linestyle=width_linestyles.get(width_val, "-"),
                    marker=width_markers.get(width_val, "o"),
                    label=None,  # We'll create a custom legend below
                    linewidth=2,
                )

        # Title & axis labels
        plt.title(
            f"Action Prediction Performance by Model/Width\nDataset: {ds}",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("Layer Group", fontsize=12)
        plt.ylabel("Avg Linear Macro F1 Score", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.3)
        sns.despine(trim=True)

        # Build a custom legend combining model & width
        legend_handles = []
        for model in models_of_interest:
            for width_val in widths_of_interest:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=model_colors[model],
                        marker=width_markers.get(width_val, "o"),
                        linestyle=width_linestyles.get(width_val, "-"),
                        label=f"{model.split('/')[-1]} ({width_val})",
                    )
                )

        plt.legend(
            handles=legend_handles,
            title="Model/Width",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=9,
        )

        plt.tight_layout()

        # Save the figure
        ds_cleaned = clean_dataset_name(ds)
        out_file = f"{ds_cleaned}_action_model_width.png"
        out_path = os.path.join(fig_dir, out_file)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"[INFO] Saved action figure: {out_path}")
