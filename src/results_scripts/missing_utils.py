import os
import pandas as pd

############################################################
# 1) DEFINE DATASET CONFIGURATIONS
############################################################

DATASET_CONFIGS = [
    {
        "dataset_name": "Anthropic/election_questions",
        "dataset_config_name": "",
        "dataset_split": "test",
    },
    {
        "dataset_name": "willcb/massive-scenario",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "willcb/massive-intent",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "legacy-datasets/banking77",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "es",
    },
    {
        "dataset_name": "AIM-Harvard/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "de",
    },
    {
        "dataset_name": "AIM-Harvard/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "ru",
    },
    {
        "dataset_name": "AIM-Harvard/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "zh",
    },
    {
        "dataset_name": "AIM-Harvard/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "en",
    },
    {
        "dataset_name": "textdetox/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "es",
    },
    {
        "dataset_name": "textdetox/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "de",
    },
    {
        "dataset_name": "textdetox/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "ru",
    },
    {
        "dataset_name": "textdetox/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "zh",
    },
    {
        "dataset_name": "textdetox/multilingual_toxicity_dataset",
        "dataset_config_name": "",
        "dataset_split": "en",
    },
    {
        "dataset_name": "SetFit/tweet_eval_stance_abortion",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/reject_prompts",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "jackhhao/jailbreak-classification",
        "dataset_config_name": "",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "english",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "spanish",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "french",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "german",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "portuguese",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "italian",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "arabic",
        "dataset_split": "test",
    },
    {
        "dataset_name": "cardiffnlp/tweet_sentiment_multilingual",
        "dataset_config_name": "hindi",
        "dataset_split": "test",
    },
    {
        "dataset_name": "gallifantjack/pminervini_NQ_Swap_org_answer_None_openai_google_gemma_2_9b_it",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "gallifantjack/pminervini_NQ_Swap_org_answer_question_openai_google_gemma_2_9b_it",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "gallifantjack/pminervini_NQ_Swap_sub_answer_question_openai_google_gemma_2_9b_it",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_2b_piqa",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_9b_piqa",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_9b_it_piqa",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_2b_boolq",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_9b_boolq",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_9b_it_boolq",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_2b_pubmed_qa",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_9b_pubmed_qa",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
    {
        "dataset_name": "AIM-Harvard/google_gemma_2_9b_it_pubmed_qa",
        "dataset_config_name": "",
        "dataset_split": "train",
    },
]

############################################################
# 2) DEFINE MODELS & LAYERS & WIDTHS
############################################################

MODELS_SPECS = [
    {
        "model_name": "google/gemma-2-2b",
        "layers": [5, 12, 19],
        "widths": ["16k", "65k", "1m"],
    },
    {
        "model_name": "google/gemma-2-9b",
        "layers": [9, 20, 31],
        "widths": ["16k", "131k", "1m"],
    },
    {
        "model_name": "google/gemma-2-9b-IT",
        "layers": [9, 20, 31],
        "widths": ["131k"],
    },
]

############################################################
# 3) GENERATE EXPECTED COMBINATIONS
############################################################


def generate_expected_combinations():
    """
    Generate all combos from:
      - Each dataset (with name/config_name/split)
      - Each model spec (with layers, widths)
      - top_n, binarize_value, etc.
    """
    hidden_types = ["Hidden States", "SAE Features"]
    top_n_values = [0, 20, 50, -1]
    binarize_values = [0, 1]

    expected = []

    for model_dict in MODELS_SPECS:
        model_name = model_dict["model_name"]
        for layer in model_dict["layers"]:
            for width in model_dict["widths"]:
                for ds_cfg in DATASET_CONFIGS:
                    ds_name = ds_cfg["dataset_name"]
                    ds_cfg_name = ds_cfg["dataset_config_name"]
                    ds_split = ds_cfg["dataset_split"]

                    for hidden_type in hidden_types:
                        for topn in top_n_values:
                            for bin_val in binarize_values:
                                combo = {
                                    "model_name": model_name,
                                    "layer": str(layer),
                                    "width": width,
                                    "type": hidden_type,
                                    "dataset_name": ds_name,
                                    "dataset_config_name": ds_cfg_name,
                                    "dataset_split": ds_split,
                                    "top_n": topn,
                                    "binarize_value": bin_val,
                                }
                                expected.append(combo)
    return expected


############################################################
# 4) CHECK MISSING COMBINATIONS
############################################################


def check_missing_combinations(df_results: pd.DataFrame, output_dir: str) -> None:
    """
    Compare df_results to all expected combos from generate_expected_combinations().
    Save missing combos in a CSV, print summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Columns to merge on
    merge_cols = [
        "model_name",
        "layer",
        "width",
        "type",
        "dataset_name",
        "dataset_config_name",
        "dataset_split",
        "top_n",
        "binarize_value",
    ]

    # --------------------------------------------------
    # 1) Print unique values and dtypes BEFORE alignment
    # --------------------------------------------------

    # Generate the expected DataFrame
    df_expected = pd.DataFrame(generate_expected_combinations())

    # --------------------------------------------------
    # 2) Enforce consistent dtypes between df_expected & df_results
    # --------------------------------------------------
    # Convert these string-based columns to str
    for col in [
        "model_name",
        "layer",
        "width",
        "type",
        "dataset_name",
        "dataset_config_name",
        "dataset_split",
    ]:
        df_expected[col] = df_expected[col].astype(str)
        if col in df_results.columns:
            df_results[col] = df_results[col].astype(str)

    # Convert numeric columns
    df_expected["top_n"] = df_expected["top_n"].astype(int)
    if "top_n" in df_results.columns:
        df_results["top_n"] = df_results["top_n"].fillna(0).astype(int)

    df_expected["binarize_value"] = df_expected["binarize_value"].astype(int)
    if "binarize_value" in df_results.columns:
        df_results["binarize_value"] = (
            df_results["binarize_value"].fillna(0).astype(int)
        )

    # Replace NaN in dataset_config_name with empty string in df_results
    df_results["dataset_config_name"] = df_results["dataset_config_name"].fillna("")

    # Then enforce string dtype
    df_results["dataset_config_name"] = df_results["dataset_config_name"].astype(str)

    # --------------------------------------------------
    # 3) Ensure df_results has all required columns
    # --------------------------------------------------
    df_mergeable = df_results.copy()
    for col in merge_cols:
        if col not in df_mergeable.columns:
            df_mergeable[col] = None

    # --------------------------------------------------
    # 4) Merge and detect missing combos
    # --------------------------------------------------
    merged_df = pd.merge(
        df_expected, df_mergeable, on=merge_cols, how="left", indicator=True
    )

    missing_df = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])
    if missing_df.empty:
        print("No missing combinations found!")
    else:
        print(f"\nFound {len(missing_df)} missing combinations.")
        missing_path = os.path.join(output_dir, "missing_combinations.csv")
        missing_df.to_csv(missing_path, index=False)
        print(f"Missing combos saved to: {missing_path}")

        # Summarize by a subset of columns
        summary = (
            missing_df.groupby(["model_name", "dataset_name", "dataset_split"])
            .size()
            .reset_index(name="missing_count")
        )

        print("\nSummary of Missing Combinations:")
        for _, row in summary.iterrows():
            print(
                f"Model: {row['model_name']} | "
                f"Dataset: {row['dataset_name']} | "
                f"Split: {row['dataset_split']} | "
                f"Missing: {row['missing_count']}"
            )

        summary_path = os.path.join(output_dir, "missing_combinations_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Summary of missing combos saved to: {summary_path}")
