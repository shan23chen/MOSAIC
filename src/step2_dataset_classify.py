import pandas as pd
import torch
from sae_lens import SAE
from pathlib import Path

import logging
import argparse
import json
import os
from sort_load_datasets import (
    get_metadata_path,
    process_data,
    save_features,
    load_features,
)
from utils import convert_to_serializable
from classifiers import ModelTrainer, TrainingConfig
from utils_dash import NumpyJSONEncoder, prepare_dashboard_data, get_tree_info
from models import get_sae_config
from utils import (
    get_save_directory,
    setup_logging,
    get_dashboard_directory,
    parse_binarize_value,
)


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
        "--model-type",
        type=str,
        choices=["vlm", "llm"],
        required=True,
        help="Model type (vlm or llm)",
    )
    parser.add_argument(
        "--sae-location", type=str, help="SAE location e.g. mlp or res", required=True
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
        "--dataset-config-name",
        type=str,
        help="Dataset config name (e.g., subject)",
        default=None,
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
    parser.add_argument(
        "--binarize-value",
        default=None,
        type=str,
        help="Value to clip features values to (None for no clipping)",
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
        "--linear-c-values",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="C values for linear probe grid search",
    )
    parser.add_argument(
        "--tree-min-samples-split",
        type=int,
        default=2,
        help="Minimum samples required to split for decision tree",
    )
    parser.add_argument(
        "--tree-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required at leaf node for decision tree",
    )
    parser.add_argument(
        "--all-tokens",
        action="store_true",
        help="Use all tokens for feature extraction",
    )

    return parser.parse_args()


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
            args.dataset_config_name,
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

            if args.model_type == "llm" and "gemma-2" in args.model_name.lower():
                # Get SAE configuration based on model architecture
                sae_location, feature_id, explanation_path = get_sae_config(
                    args.model_name, layer, args.sae_location, args.width
                )

                logging.info(
                    f"Loading SAE model from release {sae_location}, feature {feature_id}"
                )
                sae, cfg_dict, _ = SAE.from_pretrained(
                    release=f"{sae_location}",
                    sae_id=feature_id,
                    device=device,
                )
            
            elif args.model_type == "vlm" and "paligemma2" in args.model_name.lower():
                sae_location, feature_id, explanation_path = get_sae_config(
                    args.model_name, layer, args.sae_location, args.width
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
                if "it" in args.model_name.lower():
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

            try:
                features, label_encoder = load_features(
                    layer_save_dir, args.model_type, args.top_n, layer
                )
                logging.info(f"Features loaded from {layer_save_dir}")
            except FileNotFoundError:
                logging.info(f"Features not found, processing data")

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
                    top_n=args.top_n,
                    layer=layer,
                    all_tokens=args.all_tokens,
                )

                del processed_df

                features, label_encoder = load_features(
                    layer_save_dir, args.model_type, args.top_n, layer
                )

            # Initialize training configuration
            config = TrainingConfig(
                test_size=args.test_size,
                random_state=args.random_state,
                cv_folds=args.cv_folds if hasattr(args, "cv_folds") else 5,
            )

            # Initialize model trainer
            trainer = ModelTrainer(config, label_encoder)

            binarize_value = parse_binarize_value(args.binarize_value)

            # Train models with enhanced pipeline
            print("===== Training models =====")
            print("--- Hidden states ---")
            print("Training linear probe on hidden states")
            hidden_linear_results = trainer.train_linear_probe(
                features, hidden=True, binarize_value=None
            )
            print("Training decision tree on hidden states")
            hidden_tree_results = trainer.train_decision_tree(
                features, hidden=True, binarize_value=None
            )
            print("--- SAE features ---")
            print("Training linear probe on SAE features")
            sae_linear_results = trainer.train_linear_probe(
                features, hidden=False, binarize_value=binarize_value
            )
            print("Training decision tree on SAE features")
            sae_tree_results = trainer.train_decision_tree(
                features, hidden=False, binarize_value=binarize_value
            )

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
                hidden=False,
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

            # convert to serializable format
            hidden_dashboard_data = convert_to_serializable(hidden_dashboard_data)
            sae_dashboard_data = convert_to_serializable(sae_dashboard_data)

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
