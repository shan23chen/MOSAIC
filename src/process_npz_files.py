import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
import logging
import argparse
from pathlib import Path
import re
import json
import os
from dash import dcc, html
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
from classifiers import ModelTrainer, TrainingConfig
from dash_utils import NumpyJSONEncoder, prepare_dashboard_data, get_tree_info
from models import get_sae_config

from utils import get_save_directory, setup_logging, get_dashboard_directory
from tqdm import tqdm

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


def load_npz_file(npz_path, sample_id, label, progress_dict=None):
    """Load NPZ file with progress tracking."""
    try:
        with np.load(npz_path) as npz_data:
            result = {
                "sample_id": sample_id,
                "label": label,
                "hidden_state": npz_data["hidden_state"].astype(np.float16),
                "sae_acts": npz_data.get("sae_acts", None)
            }
            if result["sae_acts"] is not None:
                result["sae_acts"] = result["sae_acts"].astype(np.float16)
            
            # Update progress if dictionary provided
            if progress_dict is not None:
                progress_dict['loaded_files'] += 1
                
            return result
    except Exception as e:
        logging.error(f"Error processing file {npz_path}: {e}")
        return None

def process_batch(batch, sae, cfg_dict, last_token, top_n, device, progress_dict=None):
    """Process batch with progress tracking."""
    try:
        if progress_dict is not None:
            progress_dict['processing_status'] = 'Converting to DataFrame'
            
        batch_df = pd.DataFrame(batch)
        
        if "sae_acts" in batch_df.columns and batch_df["sae_acts"].iloc[0] is not None:
            if batch_df["sae_acts"].iloc[0].shape[-1] != cfg_dict["d_sae"]:
                raise ValueError("SAE activations shape mismatch")
            
            if progress_dict is not None:
                progress_dict['processing_status'] = 'Processing SAE activations'
                
            def process_sae_acts(x, idx=None):
                if x is None:
                    return None
                if x.shape[0] == 1:
                    x = x[0]
                if last_token:
                    x = x[-1]
                if progress_dict is not None:
                    progress_dict['processed_items'] += 1
                return x.astype(np.float16)
            
            # Process with progress tracking
            total_items = len(batch_df)
            if progress_dict is not None:
                progress_dict['total_items'] = total_items
                progress_dict['processed_items'] = 0
            
            batch_df["sae_acts"] = batch_df["sae_acts"].apply(process_sae_acts)
            
        else:
            warnings.warn("SAE activations not found in batch, encoding hidden states")
            if progress_dict is not None:
                progress_dict['processing_status'] = 'Encoding hidden states'
            
            if last_token:
                last_tokens = np.stack([state[-1] for state in batch_df["hidden_state"]])
                with torch.no_grad():
                    hidden_tensor = torch.tensor(last_tokens, dtype=torch.float32, device=device)
                    batch_df["sae_acts"] = sae.encode(hidden_tensor).cpu().numpy().astype(np.float16)
                del hidden_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                def encode_sequence(x, idx=None):
                    with torch.no_grad():
                        tensor = torch.tensor(x, dtype=torch.float32, device=device)
                        result = sae.encode(tensor).cpu().numpy().astype(np.float16)
                        del tensor
                        if progress_dict is not None:
                            progress_dict['processed_items'] += 1
                        return result
                
                if progress_dict is not None:
                    progress_dict['total_items'] = len(batch_df)
                    progress_dict['processed_items'] = 0
                
                batch_df["sae_acts"] = batch_df["hidden_state"].apply(encode_sequence)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if progress_dict is not None:
            progress_dict['processing_status'] = 'Generating features'
            progress_dict['processed_items'] = 0
        
        batch_df["features"] = batch_df["sae_acts"].apply(
            lambda x: optimized_top_n_to_one_hot(x, top_n, progress_dict)
        )
        
        batch_df.drop("sae_acts", axis=1, inplace=True)
        
        return batch_df
    
    except Exception as e:
        logging.error(f"Error in process_batch: {e}")
        raise

def optimized_top_n_to_one_hot(array, top_n, progress_dict=None, binary=False):
    """Memory-efficient top-n to one-hot conversion with progress tracking."""
    if array is None:
        return None
    
    token_length, dim_size = array.shape
    
    top_n_indices = np.argpartition(-array, top_n, axis=1)[:, :top_n]
    sparse_array = np.zeros((token_length, dim_size), dtype=np.uint8)
    row_indices = np.arange(token_length)[:, np.newaxis]
    sparse_array[row_indices, top_n_indices] = 1
    
    result = np.sum(sparse_array, axis=0)
    
    del sparse_array, row_indices, top_n_indices
    
    if progress_dict is not None:
        progress_dict['processed_items'] += 1
    
    if binary:
        return result.astype(np.bool_)
    return result.astype(np.uint16)

def process_data(metadata_df, sae, cfg_dict, last_token=False, top_n=5, batch_size=1000, num_workers=10):
    """Main processing function with detailed progress tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Processing {len(metadata_df)} samples")
    
    # Progress tracking dictionary
    progress = {
        'total_files': len(metadata_df),
        'loaded_files': 0,
        'current_batch': 0,
        'total_batches': (len(metadata_df) + batch_size - 1) // batch_size,
        'processing_status': 'Starting',
        'processed_items': 0,
        'total_items': 0
    }
    
    num_batches = progress['total_batches']
    processed_dfs = [None] * num_batches
    batch_idx = 0
    
    # Main progress bar for overall process
    with tqdm(total=len(metadata_df), desc="Overall Progress") as pbar:
        try:
            for i in range(0, len(metadata_df), batch_size):
                progress['current_batch'] += 1
                progress['processing_status'] = f'Processing batch {progress["current_batch"]}/{num_batches}'
                
                batch_metadata = metadata_df.iloc[i:i + batch_size].copy()
                
                # Load NPZ files with progress tracking
                batch_data = []
                progress['loaded_files'] = 0
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(
                            load_npz_file, 
                            row["npz_file"], 
                            row["sample_id"], 
                            row["label"],
                            progress
                        ): idx 
                        for idx, (_, row) in enumerate(batch_metadata.iterrows())
                    }
                    
                    # Progress bar for file loading
                    with tqdm(total=len(futures), desc="Loading NPZ files") as load_pbar:
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                batch_data.append(result)
                            future.cancel()
                            load_pbar.update(1)
                
                if batch_data:
                    # Process batch with progress tracking
                    progress['processing_status'] = 'Processing batch'
                    progress['processed_items'] = 0
                    progress['total_items'] = len(batch_data)
                    
                    # Progress bar for batch processing
                    with tqdm(total=len(batch_data), desc="Processing items") as proc_pbar:
                        def update_proc_bar():
                            proc_pbar.n = progress['processed_items']
                            proc_pbar.refresh()
                        
                        batch_df = process_batch(
                            batch_data, 
                            sae, 
                            cfg_dict, 
                            last_token, 
                            top_n, 
                            device, 
                            progress
                        )
                        update_proc_bar()
                        
                        processed_dfs[batch_idx] = batch_df
                        batch_idx += 1
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update overall progress
                pbar.update(len(batch_metadata))
                pbar.set_description(
                    f"Overall Progress - Batch {progress['current_batch']}/{num_batches}"
                )
                del batch_metadata, batch_data
                gc.collect()

        
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            del processed_dfs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        try:
            # Final concatenation with progress tracking
            progress['processing_status'] = 'Concatenating results'
            processed_dfs = [df for df in processed_dfs if df is not None]
            final_df = pd.concat(processed_dfs, ignore_index=True)
            return final_df
        
        finally:
            del processed_dfs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def save_features(df, layer_dir, model_type):
    """Save processed features."""
    output_dir = Path(layer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()
    print(df.head())
    # # Convert hidden_states from pandas Series to numpy array
    hidden_states_array = np.array([i[0] for i in df["hidden_state"]])

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

            if args.model_type == "llm" and "gemma-2" in args.model_name.lower():
                # Get SAE configuration based on model architecture
<<<<<<< HEAD
                sae_location, feature_id, explanation_path = get_sae_config(
                    args.checkpoint, layer, args.sae_location, args.width
=======
                sae_location, feature_id = get_sae_config(
                    args.model_name, layer, args.sae_location, args.width
>>>>>>> origin/main
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
