# missing_utils.py

import os
import pandas as pd


############################################################
# 1) DEFINE DATASETS
############################################################

# PART 1: Text Datasets
TEXT_DATASETS = [
    "Anthropic/election_questions",
    "AIM-Harvard/reject_prompts",
    "jackhhao/jailbreak-classification",
    "willcb/massive-intent",
    "willcb/massive-scenario",
    "legacy-datasets/banking77",  # stripped 'https://huggingface.co/datasets/'
    "SetFit/tweet_eval_stance_abortion",
    "baseline: tfidf, last token prob",  # Non-HF "baseline" placeholder
]

# Additional “image” or multi-modal or specialized datasets
IMAGE_DATASETS = [
    "renumics/cifar100-enriched",
    "rajistics/indian_food_images",
    "nelorth/oxford-flowers",
]

# PART 2: Multi-lingual Datasets
MULTILINGUAL_DATASETS = [
    "textdetox/multilingual_toxicity_dataset",
    "cardiffnlp/tweet_sentiment_multilingual",
]

# PART 3: Behavioral Datasets
BEHAVIORAL_DATASETS = [
    "pminervini/NQ-Swap-original",
    "pminervini/NQ-Swap-subbed-context",
    "pminervini/NQ-Swap-no-context",
    "inspect_evals/pubmedqa",
    "inspect_evals/piqa",
    "inspect_evals/boolq",
]


############################################################
# 2) DEFINE MODELS & LAYERS & WIDTHS
############################################################

# Below is an example structure that includes different
# Gemma model sizes with specific layers and widths.

MODELS_SPECS = [
    {
        "model_name": "Gemma 2 2B",
        "layers": [5, 12, 19],
        "widths": ["2^14", "2^16", "2^20"],
    },
    {
        "model_name": "Gemma 2 9B",
        "layers": [9, 20, 31],
        "widths": ["2^14", "2^17", "2^20"],
    },
    {
        "model_name": "Gemma 2 9B IT",
        "layers": [9, 20, 31],  # or [9, 31], depending on your partial combos
        "widths": ["2^17"],
    },
    {"model_name": "Gemma 2 27B", "layers": [22], "widths": ["2^17"]},
    # Additional "2,4" combos (from your notes):
    # (2,4) combos might be subsets or special cases—feel free to encode them
    # as separate or folded into the same list:
    {"model_name": "Gemma 2 2B (2,4)", "layers": [12], "widths": ["2^17"]},
    {"model_name": "Gemma 2 9B (2,4)", "layers": [20], "widths": ["2^17"]},
    {"model_name": "Gemma 2 27B (2,4)", "layers": [22], "widths": ["2^17"]},
    {"model_name": "Gemma 2 9B IT (2,4)", "layers": [9, 20, 31], "widths": ["2^17"]},
]

# If you want a separate "Image" model specification, or
# if you simply combine them with the same Gemma specs, that’s up to you.
# For demonstration, we’ll just reuse the same model specs for images,
# but in practice you might define separate model specs or rename them.


############################################################
# 3) GENERATE EXPECTED COMBINATIONS
############################################################


def generate_expected_combinations():
    """
    Generate all combos from:
      - Text + Image + Multilingual + Behavioral datasets
      - Each model spec
      - Each layer, width
      - Potential flags (binarize_value, last_token, etc.)

    Returns:
      A list of dictionaries (one per combination).
    """

    # Example: If you'd like to produce separate "type" fields for
    #    - Hidden States
    #    - SAE Features
    # or something else, you can do so here.
    # We'll show a simple approach:
    hidden_types = ["Hidden States", "SAE Features"]

    # Potential top_n or binarize_value arrays
    top_n_values = [1, 4]  # from your notes: "Multi-lingual at least do (1,4)"
    binarize_values = [0]  # or [0, 1] if you test different binarization
    last_token = [True]  # if you always set last_token=True

    all_datasets = (
        TEXT_DATASETS + IMAGE_DATASETS + MULTILINGUAL_DATASETS + BEHAVIORAL_DATASETS
    )

    expected = []
    for model_dict in MODELS_SPECS:
        model_name = model_dict["model_name"]
        for layer in model_dict["layers"]:
            for width in model_dict["widths"]:
                for dataset_name in all_datasets:
                    for hidden_type in hidden_types:
                        for topn in top_n_values:
                            for bin_val in binarize_values:
                                for lt in last_token:
                                    combo = {
                                        "model_name": model_name,
                                        "layer": str(layer),
                                        "width": width,
                                        "type": hidden_type,
                                        "dataset_name": dataset_name,
                                        "top_n": topn,
                                        "binarize_value": bin_val,
                                        "last_token": lt,
                                    }
                                    expected.append(combo)
    return expected


############################################################
# 4) CHECK MISSING COMBINATIONS
############################################################


def check_missing_combinations(df_results: pd.DataFrame, output_dir: str) -> None:
    """
    Given a DataFrame of actual results (df_results),
    compare it to all expected combos from generate_expected_combinations().

    Saves missing combos as `missing_combinations.csv` inside `output_dir`.
    Prints a summary to the console.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert expected combos -> DataFrame
    df_expected = pd.DataFrame(generate_expected_combinations())

    # For consistent merging, make sure fields match / are same dtype
    str_cols = ["model_name", "layer", "width", "type", "dataset_name"]
    for c in str_cols:
        df_expected[c] = df_expected[c].astype(str)
        if c in df_results.columns:
            df_results[c] = df_results[c].astype(str)

    # Binarize and top_n typically numeric
    df_expected["binarize_value"] = df_expected["binarize_value"].astype(int)
    if "binarize_value" in df_results.columns:
        df_results["binarize_value"] = (
            df_results["binarize_value"].fillna(0).astype(int)
        )

    if "top_n" in df_results.columns:
        df_results["top_n"] = df_results["top_n"].fillna(0).astype(int)
        df_expected["top_n"] = df_expected["top_n"].astype(int)

    # last_token typically boolean, or can convert to string if needed
    df_expected["last_token"] = df_expected["last_token"].astype(str)
    if "last_token" in df_results.columns:
        df_results["last_token"] = df_results["last_token"].astype(str)

    # Merge on the relevant columns
    merge_cols = [
        "model_name",
        "layer",
        "width",
        "type",
        "dataset_name",
        "top_n",
        "binarize_value",
        "last_token",
    ]

    # Filter actual results to only the columns we need for checking
    # (avoid KeyError if your df doesn't have all the same columns)
    df_mergeable = df_results.copy()
    for col in merge_cols:
        if col not in df_mergeable.columns:
            df_mergeable[col] = None  # or the correct default

    merged_df = pd.merge(
        df_expected, df_mergeable, on=merge_cols, how="left", indicator=True
    )

    # Identify missing combos
    missing_df = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])

    if missing_df.empty:
        print("No missing combinations found!")
    else:
        print(f"Found {len(missing_df)} missing combinations.")
        missing_path = os.path.join(output_dir, "missing_combinations.csv")
        missing_df.to_csv(missing_path, index=False)
        print(f"Missing combos saved to: {missing_path}")

        # Summarize missing combos by (model, dataset) or however you'd like
        summary = (
            missing_df.groupby(["model_name", "dataset_name"])
            .size()
            .reset_index(name="missing_count")
        )

        print("\nSummary of Missing Combinations:")
        for _, row in summary.iterrows():
            print(
                f"Model: {row['model_name']}, Dataset: {row['dataset_name']} "
                f"- Missing: {row['missing_count']}"
            )

        summary_path = os.path.join(output_dir, "missing_combinations_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Summary of missing combos saved to: {summary_path}")
