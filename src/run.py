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

# Debug configuration
try:
    import debugpy

    debugpy.listen(("localhost", 9503))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

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
        "--sae_release", type=str, help="SAE release name", required=True
    )
    parser.add_argument(
        "--layer", type=int, help="Layer to extract hidden states from", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for dataset processing"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Model checkpoint path", required=True
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

    # New arguments for SAE processing
    parser.add_argument(
        "--process_sae",
        action="store_true",
        help="Whether to process hidden states with SAE",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top values to retain in SAE processing",
    )
    parser.add_argument(
        "--last_token",
        action="store_true",
        help="Process only the last token of sequences",
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
    setup_logging(args.save_dir)
    logging.info("Starting processing with arguments: %s", args)

    try:
        # Get hidden states
        npz_files = get_hidden_states(
            model_name=args.model_name,
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            layer=args.layer,
            sae_release=args.sae_release,
            batch_size=args.batch_size,
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            dataset_split=args.dataset_split,
            text_field=args.text_field,
            image_field=args.image_field,
            label_field=args.label_field,
            max_batches=args.max_batches,
            save_dir=args.save_dir,
            activation_only=bool(args.act_only == "True"),
        )

        # Process with SAE if requested
        if args.process_sae:
            logging.info("Processing hidden states with SAE")
            processed_data = process_sae_features(npz_files, args)
            logging.info("SAE processing completed successfully")

        logging.info("All processing completed successfully")

    except Exception as e:
        logging.exception("An error occurred during processing")
        sys.exit(1)


if __name__ == "__main__":
    main()
