import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
import logging
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import json
from datetime import datetime
import os
import dash
from dash import dcc, html
import webbrowser
import threading
import time


from classifiers import (
    prepare_hidden_states_for_linear_probe,
    train_linear_probe,
    train_decision_tree,
    save_classification_results,
    plot_decision_tree,
    get_model_info,
)
from dashboard import run_dash_server, get_tree_info


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process SAE outputs and run classification"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing run.py outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (e.g., google/gemma-2b-it)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vlm", "llm"],
        required=True,
        help="Model type (vlm or llm)",
    )
    parser.add_argument(
        "--layer", type=str, default="12", help="Layer number to process"
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        required=True,
        help="SAE release name (e.g., gemma-2b)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top values for feature extraction",
    )
    parser.add_argument("--last-token", action="store_true", help="Use only last token")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_features",
        help="Output directory for processed features",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--tree-depth", type=int, default=5, help="Maximum depth for decision tree"
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save visualization plots"
    )

    return parser.parse_args()


def get_metadata_path(input_dir, model_name, layer):
    """
    Construct metadata file path based on run.py output structure.

    Args:
        input_dir (str): Base input directory
        model_name (str): Full model name
        layer (str): Layer number

    Returns:
        Path: Path to metadata CSV file
    """
    # Clean model name for file path
    clean_model_name = re.sub(r"[^a-zA-Z0-9]", "-", model_name)
    metadata_path = (
        Path(input_dir) / f"{clean_model_name}_{layer}_sae_activations_metadata.csv"
    )
    return metadata_path


def load_sae(layer, device, sae_release):
    """Load SAE model for specific layer."""
    logging.info(f"Loading SAE model from release '{sae_release}', layer '{layer}'")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=f"{sae_release}-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_post",
        device=device,
    )
    return sae, cfg_dict, sparsity


def process_data(metadata_df, sae, cfg_dict, last_token=False, top_n=5):
    """
    Process data samples and generate features.

    Args:
        metadata_df (pd.DataFrame): Metadata DataFrame
        sae (SAE): Loaded SAE model
        cfg_dict (dict): SAE configuration
        last_token (bool): Whether to use only last token
        top_n (int): Number of top values for feature extraction

    Returns:
        pd.DataFrame: Processed DataFrame with features
    """
    # Load NPZ files
    data_samples = []
    for idx in range(len(metadata_df)):
        try:
            npz_path = metadata_df.at[idx, "npz_file"]
            npz_data = np.load(npz_path)

            sample = {
                "sample_id": metadata_df.at[idx, "sample_id"],
                "label": metadata_df.at[idx, "label"],
                "hidden_state": npz_data["hidden_state"],
                "sae_acts": npz_data.get("sae_acts", None),
            }
            data_samples.append(sample)
        except Exception as e:
            logging.error(f"Error processing file {npz_path}: {e}")
            continue

    loaded_data_df = pd.DataFrame(data_samples)

    # Process features
    if (
        "sae_acts" in loaded_data_df.columns
        and loaded_data_df["sae_acts"].iloc[0] is not None
    ):
        logging.info("Using pre-computed SAE activations")
        if loaded_data_df["sae_acts"].iloc[0].shape[-1] != cfg_dict["d_sae"]:
            raise ValueError(f"SAE activations shape mismatch")
        if loaded_data_df["sae_acts"].iloc[0].shape[0] == 1:
            loaded_data_df["sae_acts"] = loaded_data_df["sae_acts"].apply(
                lambda x: x[0]
            )
    else:
        logging.info("Computing SAE activations from hidden states")
        if last_token:
            last_tokens = [state[-1] for state in loaded_data_df["hidden_state"]]
            hidden_tensor = (
                torch.tensor(np.stack(last_tokens)).to(torch.float32).to(device)
            )
            loaded_data_df["sae_acts"] = [sae.encode(hidden_tensor)]
        else:
            loaded_data_df["sae_acts"] = loaded_data_df["hidden_state"].apply(
                lambda x: sae.encode(torch.tensor(x).to(device).to(torch.float32))
            )

    # Generate features
    logging.info(f"Generating features with top_{top_n} approach")
    loaded_data_df["features"] = loaded_data_df["sae_acts"].apply(
        lambda x: optimized_top_n_to_one_hot(torch.tensor(x), top_n).cpu().numpy()
    )

    return loaded_data_df


def optimized_top_n_to_one_hot(tensor, top_n, binary=False):
    """Generate sparse one-hot representation."""
    token_length, dim_size = tensor.shape
    sparse_tensor = torch.zeros_like(tensor)
    top_n_indices = torch.topk(tensor, top_n, dim=1).indices
    sparse_tensor.scatter_(1, top_n_indices, 1)

    if binary:
        return (sparse_tensor.sum(dim=0).to(device) != 0).to(torch.int32)
    return sparse_tensor.sum(dim=0).to(device)


def save_features(df, output_dir, model_name, layer, model_type):
    """Save processed features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean model name
    clean_model_name = model_name.split("/")[-1]

    # Save features
    output_file = output_dir / f"{clean_model_name}_{layer}_{model_type}_features.npz"
    np.savez_compressed(
        output_file,
        features=np.stack(df["features"].values),
        sample_ids=df["sample_id"].values,
        labels=df["label"].values,
    )

    # Save metadata
    metadata_file = output_dir / f"{clean_model_name}_{layer}_{model_type}_metadata.csv"
    df[["sample_id", "label"]].to_csv(metadata_file, index=False)

    logging.info(f"Saved features to {output_file}")
    logging.info(f"Saved metadata to {metadata_file}")


def main():
    """Main execution function."""
    args = parse_arguments()

    try:
        # Get metadata path (using the corrected version from previous script)
        metadata_path = get_metadata_path(args.input_dir, args.model_name, args.layer)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Process data (using functions from previous script)
        metadata_df = pd.read_csv(metadata_path)
        sae, cfg_dict, sparsity = load_sae(args.layer, device, args.sae_release)
        processed_df = process_data(
            metadata_df=metadata_df,
            sae=sae,
            cfg_dict=cfg_dict,
            last_token=args.last_token,
            top_n=args.top_n,
        )

        # Save processed features
        save_features(
            df=processed_df,
            output_dir=args.output_dir,
            model_name=args.model_name,
            layer=args.layer,
            model_type=args.model_type,
        )

        # Train classifiers
        linear_results = train_linear_probe(
            processed_df, test_size=args.test_size, random_state=args.random_state
        )

        tree_results = train_decision_tree(
            processed_df,
            max_depth=args.tree_depth,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        # Save classification results
        save_classification_results(
            linear_results, args.output_dir, args.model_name, args.layer, "linear_probe"
        )

        save_classification_results(
            tree_results, args.output_dir, args.model_name, args.layer, "decision_tree"
        )

        # Plot decision tree if requested
        if args.save_plots:
            feature_names = [
                f"feature_{i}" for i in range(processed_df["features"].iloc[0].shape[0])
            ]
            class_names = processed_df["label"].unique().astype(str).tolist()
            plot_path = (
                Path(args.output_dir)
                / f"{args.model_name}_{args.layer}_decision_tree.png"
            )
            plot_decision_tree(tree_results, feature_names, class_names, plot_path)

        logging.info("Processing and classification completed successfully")

        # In your main function, modify the dashboard_data creation:
        dashboard_data = {
            "modelInfo": {
                "name": args.model_name,
                "layer": args.layer,
                "type": args.model_type,
            },
            "metrics": {
                "linearProbe": {
                    "accuracy": linear_results["accuracy"],
                    "per_class_metrics": {
                        str(class_name): {
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1_score": float(f1),
                        }
                        for class_name, precision, recall, f1 in zip(
                            linear_results["classes"],
                            linear_results["precision"],
                            linear_results["recall"],
                            linear_results["f1"],
                        )
                    },
                },
                "decisionTree": {
                    "accuracy": tree_results["accuracy"],
                    "per_class_metrics": {
                        str(class_name): {
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1_score": float(f1),
                        }
                        for class_name, precision, recall, f1 in zip(
                            tree_results["classes"],
                            tree_results["precision"],
                            tree_results["recall"],
                            tree_results["f1"],
                        )
                    },
                },
            },
            # Add the feature importances from the decision tree
            "feature_importances": tree_results["model"].feature_importances_.tolist(),
            "feature_names": [
                f"feature_{i}" for i in range(processed_df["features"].iloc[0].shape[0])
            ],
            "class_names": [str(name) for name in tree_results["classes"].tolist()],
        }

        tree_info = get_tree_info(tree_results["model"])
        dashboard_data["tree_info"] = {
            "children_left": tree_info["children_left"].tolist(),
            "children_right": tree_info["children_right"].tolist(),
            "feature": tree_info["feature"].tolist(),
            "threshold": tree_info["threshold"].tolist(),
            "n_node_samples": tree_info["n_node_samples"].tolist(),
            "impurity": tree_info["impurity"].tolist(),
            "value": tree_info["value"].tolist(),
        }

        # Save dashboard data
        dashboard_path = (
            Path(args.output_dir)
            / "google"
            / f"{args.model_name.split('/')[-1]}_{args.layer}_dashboard.json"
        )
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dashboard_path, "w") as f:
            json.dump(dashboard_data, f)

        logging.info(f"Dashboard data saved to {dashboard_path}")

        # Start the Dash server in a separate thread
        dash_thread = threading.Thread(
            target=run_dash_server, args=(str(dashboard_path),)
        )
        dash_thread.daemon = True
        dash_thread.start()

        # Keep the main thread running
        while True:
            time.sleep(1)

    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise e


if __name__ == "__main__":
    main()
