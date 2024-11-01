import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import re
import json
import os
import dash
from dash import dcc, html
import webbrowser
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from sklearn.preprocessing import LabelEncoder
import gc

from classifiers import ModelTrainer, TrainingConfig
from dash_utils import NumpyJSONEncoder, prepare_dashboard_data, get_tree_info
from models import get_sae_config

from utils import get_save_directory, setup_logging, get_dashboard_directory

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
        "--dashboard-dir",
        type=str,
        required=True,
        help="Dashboard results directory containing classifier outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (e.g., google/gemma-2b-it)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="SAE model name (e.g., google/gemma-2b-it)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vlm", "llm"],
        required=True,
        help="Model type (vlm or llm)",
    )
    parser.add_argument(
        "--sae_location", type=str, help="SAE location e.g. mlp or res", required=True
    )
    parser.add_argument(
        "--layer",
        type=str,
        help="Comma-separated layers to extract hidden states from (e.g., '7,8,9')",
        required=True,
    )
    parser.add_argument(
        "--width",
        type=str,
        default="16k",
        help="Width of the SAE encoder (e.g. 16k, 524k, 1m)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name (e.g., OncQA, MedQA, etc.)",
        required=True,
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        help="Dataset split (e.g., train, dev, test)",
        required=True,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top values for feature extraction",
    )
    parser.add_argument("--last-token", action="store_true", help="Use only last token")
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
    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--linear_c_values",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="C values for linear probe grid search",
    )
    parser.add_argument(
        "--tree_min_samples_split",
        type=int,
        default=2,
        help="Minimum samples required to split for decision tree",
    )
    parser.add_argument(
        "--tree_min_samples_leaf",
        type=int,
        default=1,
        help="Minimum samples required at leaf node for decision tree",
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
    logging.info(f"Metadata path: {metadata_path}")
    return metadata_path


def load_npz_file(npz_path, sample_id, label):
    try:
        npz_data = np.load(npz_path)
        return {
            "sample_id": sample_id,
            "label": label,
            "hidden_state": npz_data["hidden_state"],
            "sae_acts": npz_data.get("sae_acts", None),
        }
    except Exception as e:
        logging.error(f"Error processing file {npz_path}: {e}")
        return None


def process_batch(batch, sae, cfg_dict, last_token, top_n, device):
    batch_df = pd.DataFrame(batch)

    if "sae_acts" in batch_df.columns and batch_df["sae_acts"].iloc[0] is not None:
        if batch_df["sae_acts"].iloc[0].shape[-1] != cfg_dict["d_sae"]:
            raise ValueError("SAE activations shape mismatch")
        if batch_df["sae_acts"].iloc[0].shape[0] == 1:
            batch_df["sae_acts"] = batch_df["sae_acts"].apply(
                lambda x: x[0].astype("float16")
            )
    else:
        if last_token:
            last_tokens = [state[-1] for state in batch_df["hidden_state"]]
            hidden_tensor = (
                torch.tensor(np.stack(last_tokens)).to(torch.float32).to(device)
            )
            batch_df["sae_acts"] = sae.encode(hidden_tensor).cpu().numpy()
        else:
            batch_df["sae_acts"] = batch_df["hidden_state"].apply(
                lambda x: sae.encode(torch.tensor(x).to(device).to(torch.float32))
                .cpu()
                .numpy()
            )

    batch_df["features"] = batch_df["sae_acts"].apply(
        lambda x: optimized_top_n_to_one_hot(x, top_n)
    )

    return batch_df


def process_data(
    metadata_df,
    sae,
    cfg_dict,
    last_token=False,
    top_n=5,
    batch_size=1000,
    num_workers=4,
):
    """
    Process data samples and generate features efficiently.

    Args:
        metadata_df (pd.DataFrame): Metadata DataFrame
        sae (SAE): Loaded SAE model
        cfg_dict (dict): SAE configuration
        last_token (bool): Whether to use only last token
        top_n (int): Number of top values for feature extraction
        batch_size (int): Number of samples to process in each batch
        num_workers (int): Number of worker processes

    Returns:
        pd.DataFrame: Processed DataFrame with features
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Processing {len(metadata_df)} samples")

    # Load NPZ files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        load_func = partial(load_npz_file)
        futures = [
            executor.submit(load_func, row["npz_file"], row["sample_id"], row["label"])
            for _, row in metadata_df.iterrows()
        ]

        data_samples = []
        for future in as_completed(futures):
            result = future.result()
            if result:
                data_samples.append(result)

    # Process data in batches
    processed_dfs = []
    for i in range(0, len(data_samples), batch_size):
        batch = data_samples[i : i + batch_size]
        batch_df = process_batch(batch, sae, cfg_dict, last_token, top_n, device)
        processed_dfs.append(batch_df)

    # Combine all processed batches
    loaded_data_df = pd.concat(processed_dfs, ignore_index=True)

    # clear memory
    del data_samples
    del processed_dfs
    gc.collect()

    return loaded_data_df


def optimized_top_n_to_one_hot(array, top_n, binary=False):
    """Generate sparse one-hot representation."""
    token_length, dim_size = array.shape
    sparse_array = np.zeros_like(array)

    # Get indices of top n values for each row
    top_n_indices = np.argpartition(-array, top_n, axis=1)[:, :top_n]

    # Create row indices array
    row_indices = np.arange(token_length)[:, np.newaxis]
    row_indices = np.broadcast_to(row_indices, (token_length, top_n))

    # Set values to 1 at top n positions
    sparse_array[row_indices, top_n_indices] = 1

    # Sum along token dimension
    result = np.sum(sparse_array, axis=0)

    if binary:
        return result.astype(np.bool)

    return result.astype(np.uint16)


def save_features(df, layer_dir, model_type):
    """Save processed features."""
    output_dir = Path(layer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()
    layer = int(layer_dir.split("/")[-2].split("_")[1])
    print(f"Layer: {layer}")

    # # Convert hidden_states from pandas Series to numpy array
    hidden_states_array = np.array([i[layer] for i in df["hidden_state"]])

    # Save features
    output_file = output_dir / f"{model_type}_features.npz"
    logging.info(f"Saving features to {output_file}")
    np.savez_compressed(
        output_file,
        features=np.stack(df["features"].values),
        sample_ids=df["sample_id"].values,
        hidden_states=hidden_states_array,
        label=label_encoder.fit_transform(df["label"]),
    )

    # Save metadata
    logging.info(
        f"Inside Save features- Saving metadata to {output_dir} not using get metadata path"
    )
    metadata_file = output_dir / f"{model_type}_metadata.csv"
    df[["sample_id", "label"]].to_csv(metadata_file, index=False)

    logging.info(f"Saved features to {output_file}")
    logging.info(f"Saved metadata to {metadata_file}")

    return label_encoder


def load_features(layer_dir, model_type):
    """Load processed features."""
    input_dir = Path(layer_dir)

    file_path = input_dir / f"{model_type}_features.npz"

    return np.load(file_path, allow_pickle=True)


def main():
    """Main execution function."""
    args = parse_arguments()
    # Parse layers from comma-separated string
    layers = [int(layer.strip()) for layer in args.layer.split(",")]
    logging.info(f"Processing layers: {layers}")

    for layer in layers:
        # Create layer-specific save directory
        layer_save_dir = get_save_directory(
            args.input_dir,
            args.model_name,
            args.dataset_name,
            args.dataset_split,
            layer,
            args.width,
        )
        os.makedirs(layer_save_dir, exist_ok=True)

        # Setup logging for this layer
        setup_logging(layer_save_dir)
        logging.info(f"Processing layer {layer} with save directory: {layer_save_dir}")
        logging.info(f"Starting processing with arguments: {args}")

        try:
            # Get metadata path
            metadata_path = get_metadata_path(layer_save_dir, args.model_name, layer)
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            # Process data (using functions from previous script)
            metadata_df = pd.read_csv(metadata_path)

            if args.model_type == "llm" and "gemma-2" in args.checkpoint.lower():
                # Get SAE configuration based on model architecture
                sae_location, feature_id, explanation_path = get_sae_config(
                    args.checkpoint, layer, args.sae_location, args.width
                )

                logging.info(
                    f"Loading SAE model from release {sae_location}, feature {feature_id}"
                )
                sae, cfg_dict, _ = SAE.from_pretrained(
                    release=f"{sae_location}",
                    sae_id=feature_id,
                    device=device,
                )

            else:
                if "it" in args.checkpoint.lower():
                    sae_release = "gemma-2b-it"
                else:
                    sae_release = "gemma-2b"
                logging.info(
                    f"Loading SAE model from release {sae_release}, layer {layer}"
                )
                sae, cfg_dict, _ = SAE.from_pretrained(
                    release=f"{sae_release}-res-jb",
                    sae_id=f"blocks.{layer}.hook_resid_post",
                    device=device,
                )
                # TODO: Add explanation path for non-LLM models

            processed_df = process_data(
                metadata_df=metadata_df,
                sae=sae,
                cfg_dict=cfg_dict,
                last_token=args.last_token,
                top_n=args.top_n,
            )

            # Save processed features
            label_encoder = save_features(
                df=processed_df,
                layer_dir=layer_save_dir,
                model_type=args.model_type,
            )

            del processed_df

            # Initialize training configuration
            config = TrainingConfig(
                test_size=args.test_size,
                random_state=args.random_state,
                cv_folds=args.cv_folds if hasattr(args, "cv_folds") else 5,
            )

            # Initialize model trainer
            trainer = ModelTrainer(config, label_encoder)

            # Reload features
            features = load_features(layer_save_dir, args.model_type)

            # Train models with enhanced pipeline
            hidden_linear_results = trainer.train_linear_probe(features, hidden=True)
            hidden_tree_results = trainer.train_decision_tree(features, hidden=True)
            sae_linear_results = trainer.train_linear_probe(features, hidden=False)
            sae_tree_results = trainer.train_decision_tree(features, hidden=False)

            # Save results using new save method
            trainer.save_results(
                hidden_linear_results,
                layer_save_dir,
                args.model_name,
                args.layer,
                "linear_probe",
                hidden=True,
            )

            trainer.save_results(
                hidden_tree_results,
                layer_save_dir,
                args.model_name,
                args.layer,
                "decision_tree",
                hidden=True,
            )

            trainer.save_results(
                sae_linear_results,
                layer_save_dir,
                args.model_name,
                args.layer,
                "linear_probe",
                hidden=False,
            )

            trainer.save_results(
                sae_tree_results,
                layer_save_dir,
                args.model_name,
                args.layer,
                "decision_tree",
                hidden=False,
            )

            logging.info("Processing and classification completed successfully")

            # Use the explanation path in your dashboard preparation
            if explanation_path and explanation_path.exists():
                with open(explanation_path) as f:
                    feature_mapping = json.load(f)
            else:
                feature_mapping = {}  # Empty mapping if no explanations available

            # Prepare and save dashboard data
            hidden_dashboard_data = prepare_dashboard_data(
                linear_results=hidden_linear_results,
                tree_results=hidden_tree_results,
                args=args,
                layer=layer,
                tree_info=get_tree_info(hidden_tree_results["model"]),
                hidden=True,
                feature_mapping=feature_mapping,
                class_names=label_encoder.classes_,
            )
            sae_dashboard_data = prepare_dashboard_data(
                linear_results=sae_linear_results,
                tree_results=sae_tree_results,
                args=args,
                layer=layer,
                tree_info=get_tree_info(sae_tree_results["model"]),
                hidden=True,
                feature_mapping=feature_mapping,
                class_names=label_encoder.classes_,
            )

            dashboard_save_dir = get_dashboard_directory(
                args.dashboard_dir,
                args.model_name,
                args.dataset_name,
                layer,
                args.width,
            )
            os.makedirs(dashboard_save_dir, exist_ok=True)

            hidden_dashboard_path = (
                Path(dashboard_save_dir) / "hidden_classifier_results.json"
            )
            sae_dashboard_path = (
                Path(dashboard_save_dir) / "sae_classifier_results.json"
            )

            with open(hidden_dashboard_path, "w") as f:
                json.dump(hidden_dashboard_data, f, indent=2, cls=NumpyJSONEncoder)

            with open(sae_dashboard_path, "w") as f:
                json.dump(sae_dashboard_data, f, indent=2, cls=NumpyJSONEncoder)

            logging.info(f"Dashboard data saved to {hidden_dashboard_path}")

            # # Start the Dash server
            # dash_thread = threading.Thread(
            #     target=run_dash_server, args=(str(dashboard_path),)
            # )
            # dash_thread.daemon = True
            # dash_thread.start()

            # # Keep the main thread running
            # while True:
            #     time.sleep(1)

        except Exception as e:
            logging.error(f"Error processing data: {e}")
            raise e


if __name__ == "__main__":
    main()
