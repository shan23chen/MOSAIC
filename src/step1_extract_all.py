#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from get_hidden_states import get_hidden_states
from sae_lens import SAE
import gc

from utils import get_save_directory

# Debug configuration
# try:
#     import debugpy

#     debugpy.listen(("localhost", 9503))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process dataset using LLM or VLM and optionally apply SAE processing."
    )
    # Existing arguments
    parser.add_argument(
        "--model_name", type=str, help="Name or path of the model to use", required=True
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["vlm", "llm"],
        help="Type of the model (VLM or LLM)",
        required=True,
    )
    parser.add_argument(
        "--sae_location",
        type=str,
        default="res",
        choices=["res", "mlp", "att"],
        help="SAE location e.g. mlp, res or att",
        required=True,
    )
    parser.add_argument(
        "--layer",
        type=str,
        help="Comma-separated layers to extract hidden states from (e.g., '7,8,9')",
        required=True,
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for dataset processing"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./npz_files",
        help="Directory to save npz files",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name or path of the dataset to use",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Which split of the dataset to use (train/validation/test)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default=None,
        help="Field name for text data in the dataset",
    )
    parser.add_argument(
        "--image_field",
        type=str,
        default=None,
        help="Field name for image data in the dataset",
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default="label",
        help="Field name for label data in the dataset",
    )
    parser.add_argument(
        "--act_only",
        type=str,
        default="True",
        help="storing activation only or storing sae and activations",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (for debug/testing)",
    )
    parser.add_argument(
        "--width",
        type=str,
        default="16k",
        help="Width of the SAE encoder (e.g. 16k, 524k, 1m)",
    )
    parser.add_argument(
        "--all_tokens",
        type=str,
        default="False",
        help="Retrieve all tokens or only the last token, default getting all image tokens for VLM and last token for LLM",
    )
    return parser.parse_args()


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "processing.log")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def main():
    args = parse_args()

    # Parse layers from comma-separated string
    layers = [int(layer.strip()) for layer in args.layer.split(",")]
    logging.info(f"Processing layers: {layers}")

    for layer in layers:
        # Create layer-specific save directory
        layer_save_dir = get_save_directory(
            args.save_dir,
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
            # Get hidden states for current layer
            npz_files = get_hidden_states(
                model_name=args.model_name,
                model_type=args.model_type,
                layer=layer,  # Use current layer
                sae_location=args.sae_location,
                width=args.width,
                batch_size=args.batch_size,
                dataset_name=args.dataset_name,
                dataset_config_name=args.dataset_config_name,
                dataset_split=args.dataset_split,
                text_field=args.text_field,
                image_field=args.image_field,
                label_field=args.label_field,
                max_batches=args.max_batches,
                save_dir=layer_save_dir,  # Use layer-specific save directory
                activation_only=bool(args.act_only == "True"),
                all_tokens=bool(args.all_tokens == "True"),
            )

            logging.info(f"Processing completed successfully for layer {layer}")

            # clear the memory
            del npz_files
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logging.exception(f"An error occurred during processing of layer {layer}")
            continue  # Continue with next layer even if current one fails

    logging.info("All layer processing completed\n\n\n\n")


if __name__ == "__main__":
    main()
