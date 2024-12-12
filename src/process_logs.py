from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_samples
from collections import defaultdict
import json
import os
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import re


def clean_string(s):
    """Clean string by removing special chars and converting to lowercase."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s).lower()).strip("_")


def process_eval_logs():
    logs = list_eval_logs()
    print(f"Found {len(logs)} evaluation logs.")
    results = defaultdict(lambda: {"task_info": [], "samples": []})
    os.makedirs("per_run_results", exist_ok=True)

    for log_file in logs:
        print(f"Processing log: {log_file}")
        log = read_eval_log(log_file, header_only=True)

        if log.status == "success":
            task_id = log.eval.run_id
            model_name = log.eval.model
            model_url = log.eval.model_base_url
            task_name = log.eval.dataset.name
            dataset_name = log.eval.task_args.get("dataset_name", "N/A")
            if dataset_name == "N/A":
                dataset_name = task_name
            input_column = log.eval.task_args.get("input_column", "N/A")
            label_column = log.eval.task_args.get("label_column", "N/A")
            split = log.eval.task_args.get("split", "N/A")
            dataset_samples = log.eval.dataset.samples if log.eval.dataset else 0

            # Create a composite key (dataset-model)
            dict_key = f"{clean_string(dataset_name)}_{clean_string(model_name)}"
            filename = os.path.join("per_run_results", f"{dict_key}.json")

            # If file already exists, skip processing
            if os.path.exists(filename):
                print(f"Results for {dict_key} already exist at {filename}, skipping.")
                continue

            # Collect scores
            for score in log.results.scores:
                score_name = score.name
                accuracy = (
                    score.metrics.get("accuracy", {}).value
                    if "accuracy" in score.metrics
                    else None
                )
                results[dict_key]["task_info"].append(
                    {
                        "task_id": task_id,
                        "model_name": model_name,
                        "model_url": model_url,
                        "dataset_name": dataset_name,
                        "input_column": input_column,
                        "label_column": label_column,
                        "split": split,
                        "samples": dataset_samples,
                        "score_name": score_name,
                        "accuracy": accuracy,
                    }
                )

            # Collect samples
            for sample in read_eval_log_samples(log_file):
                sample_dict = {
                    "id": sample.id,
                    "user": sample.messages[0].content if sample.messages else None,
                    "assistant": (
                        sample.messages[1].content if len(sample.messages) > 1 else None
                    ),
                    "target": sample.target,
                    "score": sample.score.value if sample.score else None,
                    "explanation": sample.score.explanation if sample.score else None,
                }
                results[dict_key]["samples"].append(sample_dict)

            # Save to per-run JSON
            with open(filename, "w") as f:
                json.dump(results[dict_key], f, indent=4)
            print(f"Saved results for {dict_key} to {filename}")

    return results


def create_dataset_card(dataset_name, label_column):
    """Create a dataset card for the Hugging Face repository."""
    readme_content = f"""---
language: en
license: mit
pretty_name: {dataset_name}
size_categories:
- n<1K
task_categories:
- text-classification
- other
task_ids:
- text-classification
- medical-diagnosis
---

# {dataset_name}

## Dataset Description

This dataset contains evaluation results for {dataset_name} with label column {label_column}, with various model performance metrics and samples.

### Dataset Summary

The dataset contains original samples from the evaluation process, along with metadata like model names, input columns, and scores. This helps with understanding model performance across different tasks and datasets.

### Features

- id: Unique identifier for the sample.
- user: User query/content.
- assistant: Assistant response.
- target: The expected output.
- score: Score of the assistant's response.
- explanation: Explanation of the score.
- input_column: Input column used in the original dataset.
- label_column: Label column in the original dataset.
- model_name: Name of the model used in evaluation.
- dataset_name: Name of the original dataset used.

### Usage

This dataset can be used for:
- Evaluating model robustness across various tasks.
- Assessing potential biases in model responses.
- Model performance monitoring and analysis.

### Citation

If you use this dataset, please cite:
@misc{{eval_dataset_{dataset_name}, title={{Evaluation Dataset for {dataset_name}}}, author={{Gallifant, Jack}}, year={{2024}}, publisher={{Hugging Face}} }}
"""
    return readme_content


def upload_eval_dataset_to_huggingface(
    df, dataset_name, label_column, repo_id="gallifantjack/task_dataset"
):
    """
    Upload the evaluation dataset to Hugging Face.

    Args:
        df: DataFrame containing the dataset.
        dataset_name: Name of the original dataset.
        label_column: Label column in the dataset.
        repo_id: Hugging Face repository ID.
    """
    # Create HF dataset
    dataset = Dataset.from_pandas(df)

    # Create dataset dictionary with a single split
    dataset_dict = DatasetDict({"train": dataset})

    # Initialize Hugging Face API
    api = HfApi()

    try:
        # Create or update the repository
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        # Push the dataset
        dataset_dict.push_to_hub(repo_id, private=False)

        # Create README content
        readme_content = create_dataset_card(dataset_name, label_column)

        # Upload the README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"Successfully uploaded dataset to {repo_id}")

    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")


if __name__ == "__main__":
    # Process logs and create per-run JSON files (skip if they already exist)
    results = process_eval_logs()

    if results:
        for dict_key, data in results.items():
            # Convert samples to DataFrame
            samples_df = pd.DataFrame(data["samples"])
            if samples_df.empty:
                print(f"No samples found for {dict_key}, skipping upload.")
                continue

            # Extract dataset_name and label_column from the first task_info entry
            if data["task_info"]:
                dataset_name = data["task_info"][0]["dataset_name"]
                label_column = data["task_info"][0]["label_column"]
            else:
                dataset_name = "unknown_dataset"
                label_column = "unknown_label"

            # Clean dataset and label names for repo_id
            dataset_name = dataset_name.replace("/", "_").replace("-", "_")
            label_column = label_column.replace("/", "_").replace("-", "_")

            repo_id = f"gallifantjack/{dataset_name}_{label_column}"

            # Upload the extracted dataset
            upload_eval_dataset_to_huggingface(
                samples_df, dataset_name, label_column, repo_id=repo_id
            )
    else:
        print("No results to upload.")
