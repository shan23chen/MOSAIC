import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_layer_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unique_models = df["model_name"].unique()
    model_layer_map = {}

    for model in unique_models:
        layers_for_model = sorted(df.loc[df["model_name"] == model, "layer"].unique())
        if len(layers_for_model) != 3:
            logging.warning(
                f"[WARN] Model '{model}' does not have exactly 3 layers. Found {len(layers_for_model)}. "
                "Adjust logic if needed."
            )
            continue

        layer_group_names = ["early", "middle", "late"]
        model_layer_map[model] = {
            layer: layer_group_names[idx] for idx, layer in enumerate(layers_for_model)
        }

    def get_layer_group(row):
        m = row["model_name"]
        l = row["layer"]
        if m in model_layer_map and l in model_layer_map[m]:
            return model_layer_map[m][l]
        return None

    df["layer_group"] = df.apply(get_layer_group, axis=1)
    return df


def plot_pooling_strategies(
    df: pd.DataFrame,
    results_dir: str,
    pooling_datasets_list: list,
    pooling_widths_of_interest: list,
    score_col: str = "linear_macro_f1_score",
) -> None:

    df = compute_layer_groups(df)

    pooling_fig_dir = os.path.join(results_dir, "figures", "pooling_strategies")
    os.makedirs(pooling_fig_dir, exist_ok=True)

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

    df_filtered = df[
        df["dataset_name"].isin(pooling_datasets_list)
        & df["width"].isin(pooling_widths_of_interest)
    ].copy()

    for ds_name in pooling_datasets_list:
        for w in pooling_widths_of_interest:
            subset = df_filtered[
                (df_filtered["dataset_name"] == ds_name) & (df_filtered["width"] == w)
            ]
            if subset.empty:
                logging.info(
                    f"[INFO] No data for dataset={ds_name}, width={w}. Skipping."
                )
                continue

            width_dir = os.path.join(pooling_fig_dir, w)
            os.makedirs(width_dir, exist_ok=True)

            # A) top_n plot
            grouped_topn = (
                subset.groupby(["model_name", "layer_group", "top_n"])[score_col]
                .mean()
                .reset_index()
            )

            g = sns.catplot(
                data=grouped_topn,
                x="top_n",
                y=score_col,
                hue="model_name",
                col="layer_group",
                col_order=["early", "middle", "late"],  # Force correct order
                kind="bar",
                height=5,
                aspect=1.0,
                sharey=True,
            )
            g.set(ylim=(0.5, 1.0))
            g.set_titles(template="{col_name}")
            g.set_axis_labels("top_n", score_col)

            title_text = f"Dataset: {ds_name} | Width: {w}\n(top_n vs. {score_col})"
            g.fig.suptitle(title_text, y=1.05, fontsize=16, weight="bold")
            g.tight_layout()

            ds_clean = ds_name.replace("/", "_").replace(" ", "_")
            fname_topn = f"{ds_clean}_topn_bar.png"
            outpath_topn = os.path.join(width_dir, fname_topn)
            g.savefig(outpath_topn, dpi=300, bbox_inches="tight")
            plt.close(g.fig)
            logging.info(f"[INFO] Saved top_n bar -> {outpath_topn}")

            # B) binarize_value plot
            grouped_bin = (
                subset.groupby(["model_name", "layer_group", "binarize_value"])[
                    score_col
                ]
                .mean()
                .reset_index()
            )

            g2 = sns.catplot(
                data=grouped_bin,
                x="binarize_value",
                y=score_col,
                hue="model_name",
                col="layer_group",
                col_order=["early", "middle", "late"],  # Force correct order
                kind="bar",
                height=5,
                aspect=1.0,
                sharey=True,
            )
            g2.set(ylim=(0.5, 1.0))
            g2.set_titles(template="{col_name}")
            g2.set_axis_labels("binarize_value", score_col)

            title_text = (
                f"Dataset: {ds_name} | Width: {w}\n(binarize_value vs. {score_col})"
            )
            g2.fig.suptitle(title_text, y=1.05, fontsize=16, weight="bold")
            g2.tight_layout()

            fname_bin = f"{ds_clean}_binarize_bar.png"
            outpath_bin = os.path.join(width_dir, fname_bin)
            g2.savefig(outpath_bin, dpi=300, bbox_inches="tight")
            plt.close(g2.fig)
            logging.info(f"[INFO] Saved binarize_value bar -> {outpath_bin}")
