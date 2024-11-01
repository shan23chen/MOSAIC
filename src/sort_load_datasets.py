import torch
import re
import gc
from pathlib import Path
import logging
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

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
                "sae_acts": npz_data.get("sae_acts", None),
            }
            if result["sae_acts"] is not None:
                result["sae_acts"] = result["sae_acts"].astype(np.float16)

            # Update progress if dictionary provided
            if progress_dict is not None:
                progress_dict["loaded_files"] += 1

            return result
    except Exception as e:
        logging.error(f"Error processing file {npz_path}: {e}")
        return None


def process_batch(batch, sae, cfg_dict, last_token, top_n, device, progress_dict=None):
    """Process batch with progress tracking."""
    try:
        if progress_dict is not None:
            progress_dict["processing_status"] = "Converting to DataFrame"

        batch_df = pd.DataFrame(batch)

        if "sae_acts" in batch_df.columns and batch_df["sae_acts"].iloc[0] is not None:
            if batch_df["sae_acts"].iloc[0].shape[-1] != cfg_dict["d_sae"]:
                raise ValueError("SAE activations shape mismatch")

            if progress_dict is not None:
                progress_dict["processing_status"] = "Processing SAE activations"

            def process_sae_acts(x, idx=None):
                if x is None:
                    return None
                if x.shape[0] == 1:
                    x = x[0]
                if last_token:
                    x = x[-1]
                if progress_dict is not None:
                    progress_dict["processed_items"] += 1
                return x.astype(np.float16)

            # Process with progress tracking
            total_items = len(batch_df)
            if progress_dict is not None:
                progress_dict["total_items"] = total_items
                progress_dict["processed_items"] = 0

            batch_df["sae_acts"] = batch_df["sae_acts"].apply(process_sae_acts)

        else:
            warnings.warn("SAE activations not found in batch, encoding hidden states")
            if progress_dict is not None:
                progress_dict["processing_status"] = "Encoding hidden states"

            if last_token:
                last_tokens = np.stack(
                    [state[-1] for state in batch_df["hidden_state"]]
                )
                with torch.no_grad():
                    hidden_tensor = torch.tensor(
                        last_tokens, dtype=torch.float32, device=device
                    )
                    batch_df["sae_acts"] = (
                        sae.encode(hidden_tensor).cpu().numpy().astype(np.float16)
                    )
                del hidden_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:

                def encode_sequence(x, idx=None):
                    with torch.no_grad():
                        tensor = torch.tensor(x, dtype=torch.float32, device=device)
                        result = sae.encode(tensor).cpu().numpy().astype(np.float16)
                        del tensor
                        if progress_dict is not None:
                            progress_dict["processed_items"] += 1
                        return result

                if progress_dict is not None:
                    progress_dict["total_items"] = len(batch_df)
                    progress_dict["processed_items"] = 0

                batch_df["sae_acts"] = batch_df["hidden_state"].apply(encode_sequence)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if progress_dict is not None:
            progress_dict["processing_status"] = "Generating features"
            progress_dict["processed_items"] = 0

        batch_df["features"] = batch_df["sae_acts"].apply(
            lambda x: optimized_top_n_to_one_hot(x, top_n, progress_dict)
        )

        batch_df.drop("sae_acts", axis=1, inplace=True)

        return batch_df

    except Exception as e:
        logging.error(f"Error in process_batch: {e}")
        raise


def optimized_top_n_to_one_hot(array, top_n, progress_dict=None, binary=False, int8=False):
    """Memory-efficient top-n to one-hot conversion with progress tracking."""
    if array is None:
        return None

    token_length, dim_size = array.shape

    # taking top_n activated functions 
    top_n_indices = np.argpartition(-array, top_n, axis=1)[:, :top_n]
    sparse_array = np.zeros((token_length, dim_size), dtype=np.uint8)

    # making [0.1, 0.5, 0.3] to [0, 1, 0] if top_n_indices = [1]
    row_indices = np.arange(token_length)[:, np.newaxis]
    sparse_array[row_indices, top_n_indices] = 1

    result = np.sum(sparse_array, axis=0)

    del sparse_array, row_indices, top_n_indices

    if progress_dict is not None:
        progress_dict["processed_items"] += 1

    if binary:
        # for multiple tokens:
        # [0, 5, 0, 3, 0] -> [0, 1, 0, 1, 0]
        return result.astype(np.bool_)
    if int8:
        # uint8 0~255 2^7
        # change all number over 255 to 255
        result[result > 255] = 255
        return result.astype(np.int8)
    else:
        # uint16 0~65535 2^15
        return result.astype(np.uint16)


def process_data(
    metadata_df,
    sae,
    cfg_dict,
    last_token=False,
    top_n=5,
    batch_size=1000,
    num_workers=10,
):
    """Main processing function with detailed progress tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Processing {len(metadata_df)} samples")

    # Progress tracking dictionary
    progress = {
        "total_files": len(metadata_df),
        "loaded_files": 0,
        "current_batch": 0,
        "total_batches": (len(metadata_df) + batch_size - 1) // batch_size,
        "processing_status": "Starting",
        "processed_items": 0,
        "total_items": 0,
    }

    num_batches = progress["total_batches"]
    processed_dfs = [None] * num_batches
    batch_idx = 0

    # Main progress bar for overall process
    with tqdm(total=len(metadata_df), desc="Overall Progress") as pbar:
        try:
            for i in range(0, len(metadata_df), batch_size):
                progress["current_batch"] += 1
                progress["processing_status"] = (
                    f'Processing batch {progress["current_batch"]}/{num_batches}'
                )

                batch_metadata = metadata_df.iloc[i : i + batch_size].copy()

                # Load NPZ files with progress tracking
                batch_data = []
                progress["loaded_files"] = 0
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(
                            load_npz_file,
                            row["npz_file"],
                            row["sample_id"],
                            row["label"],
                            progress,
                        ): idx
                        for idx, (_, row) in enumerate(batch_metadata.iterrows())
                    }

                    # Progress bar for file loading
                    with tqdm(
                        total=len(futures), desc="Loading NPZ files"
                    ) as load_pbar:
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                batch_data.append(result)
                            future.cancel()
                            load_pbar.update(1)

                if batch_data:
                    # Process batch with progress tracking
                    progress["processing_status"] = "Processing batch"
                    progress["processed_items"] = 0
                    progress["total_items"] = len(batch_data)

                    # Progress bar for batch processing
                    with tqdm(
                        total=len(batch_data), desc="Processing items"
                    ) as proc_pbar:

                        def update_proc_bar():
                            proc_pbar.n = progress["processed_items"]
                            proc_pbar.refresh()

                        batch_df = process_batch(
                            batch_data,
                            sae,
                            cfg_dict,
                            last_token,
                            top_n,
                            device,
                            progress,
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
            progress["processing_status"] = "Concatenating results"
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
    if df["hidden_state"].iloc[0].shape[0] == 1:
        print('《==》last token already, reducing dimension now')
        hidden_states_array = np.array([i[0] for i in df["hidden_state"]])
    else:
        print('《==》it is going in here - multiple tokens, taking last for residual_stream')
        hidden_states_array = np.array([i[-1] for i in df["hidden_state"]])

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