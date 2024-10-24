# run.py

import argparse
import logging
import os
import sys
from get_hidden_states import get_hidden_states
import pandas as pd

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9503))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset using LLM or VLM and SAE.")
    parser.add_argument('--model_name', type=str, help="Name or path of the model to use", required=True)
    parser.add_argument('--model_type', type=str, choices=["vlm", "llm"], help="Type of the model (VLM or LLM)", required=True)
    parser.add_argument('--sae_release', type=str, help="SAE release name", required=True)
    parser.add_argument('--layer', type=int, help="Layer to extract hidden states from", required=True)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for dataset processing")
    parser.add_argument('--checkpoint', type=str, help="Model checkpoint path", required=True)
    parser.add_argument('--save_dir', type=str, default="./npz_files", help="Directory to save npz files")
    # New arguments for dataset
    parser.add_argument('--dataset_name', type=str, help="Name or path of the dataset to use", required=True)
    parser.add_argument('--dataset_config_name', type=str, default=None, help="Dataset configuration name")
    parser.add_argument('--dataset_split', type=str, default='train', help="Which split of the dataset to use (train/validation/test)")
    # New arguments for data fields
    parser.add_argument('--text_field', type=str, default=None, help="Field name for text data in the dataset")
    parser.add_argument('--image_field', type=str, default=None, help="Field name for image data in the dataset")
    parser.add_argument('--label_field', type=str, default='label', help="Field name for label data in the dataset")
    parser.add_argument('--act_only', type=str, default='True', help="storing activation only or storing sae and activations")
    # New argument for debug mode
    parser.add_argument('--max_batches', type=int, default=None, help="Maximum number of batches to process (for debug/testing)")
    return parser.parse_args()

def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'processing.log')
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def main():
    args = parse_args()
    setup_logging(args.save_dir)
    logging.info("Starting processing with arguments: %s", args)
    try:
        # Get hidden states and process them incrementally
        get_hidden_states(
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
            save_dir=args.save_dir,  # Pass the save directory
            activation_only = bool(args.act_only == 'True')
        )

        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.exception("An error occurred during processing")
        sys.exit(1)

if __name__ == "__main__":
    main()
