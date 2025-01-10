import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas.api.types import CategoricalDtype


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


def plot_sae_across_models_and_widths_by_dataset(
    df: pd.DataFrame,
    results_dir: str,
    datasets_of_interest: list,
    models_of_interest: list,
    widths_of_interest: list,
) -> None:
    """
    For each dataset in `datasets_of_interest`, create a figure that shows
    SAE performance for every (model, width) combination across the layer groups
    (early, middle, late).

    This yields one plot per dataset, each containing multiple lines
    (one for each (model, width) combo).

    Args:
        df: DataFrame containing columns like:
            'dataset_name', 'model_name', 'width', 'layer', 'type', 'linear_macro_f1_score', etc.
        results_dir: Base directory for saving figures.
        datasets_of_interest: The specific datasets to plot separately.
        models_of_interest: Which models to include in each dataset's figure.
        widths_of_interest: Which widths to include in each dataset's figure.
    """

    # Create subdirectory for these per-dataset plots
    fig_dir = os.path.join(results_dir, "figures", "scale_model_width")
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

    # Force `layer_group` in the DataFrame
    df = df.copy()
    df["layer_group"] = df["layer"].astype(int).apply(layer_group)
    # If you'll average across top_n, binarize_value, etc. (adjust as needed)
    grouped_df = (
        df[df["type"] == "SAE Features"]  # only SAE Features
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

    # Loop over each dataset to produce one figure
    for ds in datasets_of_interest:
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
                    label=None,  # We'll create a custom legend
                    linewidth=2,
                )

        # Title & axis labels
        plt.title(
            f"SAE Performance by Model/Width\nDataset: {ds}", fontsize=16, weight="bold"
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
        out_file = f"{ds_cleaned}_model_width_sae.png"
        out_path = os.path.join(fig_dir, out_file)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"[INFO] Saved figure: {out_path}")
