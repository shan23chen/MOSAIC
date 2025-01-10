import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from matplotlib.lines import Line2D


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def clean_dataset_name(dataset_name: str) -> str:
    """
    Cleans the dataset name for safe file-saving.
    """
    return dataset_name.replace("/", "_").replace(" ", "_")


def layer_group(layer) -> str:
    """
    Maps layer index to a string indicating 'early', 'middle', or 'late'.
    """
    if layer < 10:
        return "early"
    elif layer == 19 or layer > 20:
        return "late"
    else:
        return "middle"


# -------------------------------------------------------------------
# 1) PLOT FOR A SPECIFIC WIDTH AND DATASET
# -------------------------------------------------------------------
def plot_dataset_width_sae(
    df: pd.DataFrame,
    results_dir: str,
    datasets_of_interest: list,
    widths_of_interest: list,
) -> None:
    """
    For each dataset in `datasets_of_interest` and each width in `widths_of_interest`,
    plots the layer-group performance (SAE features) across top_n and binarize_value.

    * A variation of your existing `multimodal_eval`, but extended to loop over widths.

    Args:
        df: DataFrame with columns like 'dataset_name', 'model_name', 'width', 'layer',
            'type', 'top_n', 'binarize_value', 'linear_macro_f1_score', etc.
        results_dir: Base directory to save the generated plots.
        datasets_of_interest: List of dataset names to include in the loops.
        widths_of_interest: List of widths to include in the loops.
    """

    # Create output subdir for these plots
    figures_dir = os.path.join(results_dir, "figures", "multimodal_dataset_width")
    os.makedirs(figures_dir, exist_ok=True)

    # Set up a color palette
    openai_palette = [
        "#0DAB76",
        "#33A1DE",
        "#FF6F61",
        "#6B5B95",
        "#88B04B",
        "#F7CAC9",
        "#92A8D1",
    ]
    sns.set_theme(style="whitegrid", context="talk", palette=openai_palette)

    # Hard-coded vision baselines
    vision_baselines = {
        "nelorth/oxford-flowers": 0.98,
        "rajistics/indian_food_images": 0.98,
        "renumics/cifar100-enriched": 0.79,
    }

    # Filter down to the relevant model (as in your original code)
    df = df.copy()  # to avoid modifying the caller's DataFrame
    df = df[df["model_name"] == "google/paligemma2-3b-pt-224"]
    df = df.fillna("no")  # Fill NaNs with 'no'
    df["layer_group"] = df["layer"].astype(int).apply(layer_group)

    # For each (dataset, width), we generate a plot
    for dataset_name in datasets_of_interest:
        for width_val in widths_of_interest:
            subset = df[
                (df["dataset_name"] == dataset_name) & (df["width"] == width_val)
            ]

            # If there's no data, skip
            if subset.empty:
                logging.warning(
                    f"[WARN] No data found for dataset={dataset_name}, width={width_val}. Skipping."
                )
                continue

            # 1) Hidden States baseline: find best top_n/binarize combo
            hidden_states_df = subset[subset["type"] == "Hidden States"]
            grouped_hidden = (
                hidden_states_df.groupby(["dataset_name", "top_n", "binarize_value"])[
                    "linear_macro_f1_score"
                ]
                .mean()
                .reset_index()
            )
            if not grouped_hidden.empty:
                best_combos_hidden = grouped_hidden.loc[
                    grouped_hidden.groupby("dataset_name")[
                        "linear_macro_f1_score"
                    ].idxmax()
                ]
                # Create a dictionary for baseline lines
                residual_baselines = {
                    ds: score
                    for ds, score in zip(
                        best_combos_hidden["dataset_name"],
                        best_combos_hidden["linear_macro_f1_score"],
                    )
                }
            else:
                residual_baselines = {}

            # 2) SAE Features analysis
            sae_df = subset[subset["type"] == "SAE Features"]
            grouped_sae = (
                sae_df.groupby(["dataset_name", "layer", "top_n", "binarize_value"])[
                    "linear_macro_f1_score"
                ]
                .mean()
                .reset_index()
            )
            if grouped_sae.empty:
                logging.warning(
                    f"[WARN] No SAE Features data for dataset={dataset_name}, width={width_val}. Skipping plot."
                )
                continue

            # For plotting, define a "setting" column
            grouped_sae["setting"] = grouped_sae.apply(
                lambda r: f"top_n: {r.top_n}, binarize: {r.binarize_value}", axis=1
            )

            # Force 'layer_group' again on grouped_sae
            grouped_sae["layer_group"] = (
                grouped_sae["layer"].astype(int).apply(layer_group)
            )
            cat_type = CategoricalDtype(
                categories=["early", "middle", "late"], ordered=True
            )
            grouped_sae["layer_group"] = grouped_sae["layer_group"].astype(cat_type)

            # Plot
            plt.figure(figsize=(10, 7), dpi=300)
            ax = sns.lineplot(
                data=grouped_sae,
                x="layer_group",
                y="linear_macro_f1_score",
                hue="setting",
                style="setting",
                markers=True,
                dashes=False,
                linewidth=2.5,
                markersize=8,
            )

            # Add baseline lines
            vb = vision_baselines.get(dataset_name, None)
            rb = residual_baselines.get(dataset_name, None)
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
                    label=f"Residual Stream Best ({rb:.2f})",
                )

            plt.title(
                f"{dataset_name} (Width = {width_val})\nSAE Performance by Layer Group",
                fontsize=18,
                weight="bold",
            )
            plt.xlabel("Layer Group", fontsize=14)
            plt.ylabel("Linear Macro F1 Score", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.3)

            # Move legend outside
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(
                by_label.values(),
                by_label.keys(),
                title="Setting",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=10,
                title_fontsize=12,
            )

            sns.despine(trim=True)
            plt.tight_layout()

            # Save figure
            cleaned_ds_name = clean_dataset_name(dataset_name)
            fname = f"{cleaned_ds_name}_width_{width_val}_sae_performance.png"
            output_path = os.path.join(figures_dir, fname)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logging.info(f"[DEBUG] Plot saved -> {output_path}")
