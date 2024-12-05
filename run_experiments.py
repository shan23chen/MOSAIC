#!/usr/bin/env python3

import yaml
import os
import subprocess
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run extraction and classification experiments."
)
parser.add_argument(
    "--extract-only",
    action="store_true",
    help="Run extraction only.",
)
parser.add_argument(
    "--classify-only",
    action="store_true",
    help="Run classification only.",
)
args = parser.parse_args()

# Load configuration from YAML file
with open("configc.yaml", "r") as f:
    config = yaml.safe_load(f)

settings = config["settings"]
models = config["models"]
datasets = config["datasets"]
classification_params = config["classification_params"]

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = settings["cuda_visible_devices"]

# Base directories
BASE_SAVE_DIR = settings["base_save_dir"]
BASE_CLASSIFY_DIR = settings["base_classify_dir"]


def run_extraction(
    model_name, model_type, sae_location, layers, width, dataset, act_only, batch_size
):
    print("===============================================")
    print("Starting token extraction with configuration:")
    print(f"Model: {model_name}")
    print(f"Layers: {layers}")
    print(f"Width: {width}")
    print(f"Dataset: {dataset['name']}")
    print(f"Split: {dataset['split']}")
    print(f"Text Field: {dataset['text_field']}")
    print(f"Label Field: {dataset['label_field']}")
    print("===============================================")

    cmd = [
        "python",
        "src/step1_extract_all.py",
        "--model-name",
        model_name,
        "--model-type",
        model_type,
        "--sae-location",
        sae_location,
        "--layer",
        ",".join(map(str, layers)),
        "--save-dir",
        BASE_SAVE_DIR,
        "--dataset-name",
        dataset["name"],
        "--dataset-config-name",
        dataset.get("config_name", "None"),
        "--dataset-split",
        dataset["split"],
        "--text-field",
        dataset["text_field"],
        "--batch-size",
        str(batch_size),
        "--image-field",
        str(dataset["image_field"]) if "image_field" in dataset else "NA",
        "--label-field",
        dataset["label_field"],
        "--act-only",
        str(act_only),
        "--width",
        width,
        "--all-tokens",
        "True",
    ]

    try:
        subprocess.check_call(cmd)
        print(
            f"Successfully completed extraction for {dataset['name']} with {model_name} (width={width})"
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Error during extraction for {dataset['name']} with {model_name} (width={width})"
        )
        print(f"Error code: {e.returncode}")


def run_classification(
    model_name,
    model_type,
    sae_location,
    layers,
    width,
    dataset,
    top_n,
    binarize_value,
    test_size,
    tree_depth,
):
    print("===============================================")
    print("Starting classification with configuration:")
    print(f"Model: {model_name}")
    print(f"Layers: {layers}")
    print(f"Width: {width}")
    print(f"Dataset: {dataset['name']}")
    print(f"Split: {dataset['split']}")
    print(f"Top N: {top_n}")
    print(f"Binarize Value: {binarize_value}")
    print("===============================================")

    cmd = [
        "python",
        "src/step2_dataset_classify.py",
        "--input-dir",
        BASE_SAVE_DIR,
        "--dashboard-dir",
        BASE_CLASSIFY_DIR,
        "--model-name",
        model_name,
        "--dataset-name",
        dataset["name"],
        "--dataset-config-name",
        dataset.get("config_name", "None"),
        "--model-type",
        model_type,
        "--dataset-split",
        dataset["split"],
        "--layer",
        ",".join(map(str, layers)),
        "--sae-location",
        sae_location,
        "--width",
        width,
        "--top-n",
        str(top_n),
        "--test-size",
        str(test_size),
        "--tree-depth",
        str(tree_depth),
        "--save-plots",
    ]

    if binarize_value is not None:
        cmd.extend(["--binarize-value", str(binarize_value)])

    try:
        subprocess.check_call(cmd)
        print(
            f"Successfully completed classification for {dataset['name']} with {model_name} (width={width})"
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Error during classification for {dataset['name']} with {model_name} (width={width})"
        )
        print(f"Error code: {e.returncode}")


def main():
    for model in models:
        model_name = model["name"]
        layers = model["layers"]
        model_type = settings["model_type"]
        sae_location = settings["sae_location"]
        act_only = settings["act_only"]
        batch_size = settings["batch_size"]
        test_size = settings["test_size"]
        tree_depth = settings["tree_depth"]

        for width in model["widths"]:
            for dataset in datasets:
                if not args.classify_only:
                    # Run token extraction
                    run_extraction(
                        model_name,
                        model_type,
                        sae_location,
                        layers,
                        width,
                        dataset,
                        act_only,
                        batch_size,
                    )

                if not args.extract_only:
                    # Run classification with various top_n and binarize_value settings
                    for top_n in classification_params["top_n_values"]:
                        for binarize_value in classification_params["binarize_values"]:
                            run_classification(
                                model_name,
                                model_type,
                                sae_location,
                                layers,
                                width,
                                dataset,
                                top_n,
                                binarize_value,
                                test_size,
                                tree_depth,
                            )
                    # Run classification for extra_top_n and extra_binarize_value
                    run_classification(
                        model_name,
                        model_type,
                        sae_location,
                        layers,
                        width,
                        dataset,
                        classification_params["extra_top_n"],
                        classification_params["extra_binarize_value"],
                        test_size,
                        tree_depth,
                    )

    print(f"All processes completed at: {subprocess.getoutput('date')}")


if __name__ == "__main__":
    main()
