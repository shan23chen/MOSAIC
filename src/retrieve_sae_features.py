# retrieve_sae_features.py

import torch
import numpy as np
import os
import logging
import pandas as pd  # Import pandas to handle metadata
from models import analyze_with_sae
from tqdm import tqdm


def retrieve_sae_features(
    model_name,
    layer,
    hidden_states,
    labels,
    ids,
    sae,
    save_dir="./npz_files",
    activation_only=True,
):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    with torch.no_grad():
        for idx_in_batch, (hidden_state, label, sample_id) in enumerate(
            tqdm(
                zip(hidden_states, labels, ids),
                total=len(ids),
                desc="Processing Samples in Batch",
            )
        ):
            try:
                if activation_only:
                    # Store the activations as an npz file
                    file_id = f"sample_{sample_id}"
                    npz_file_path = os.path.join(save_dir, f"{file_id}.npz")
                    np.savez_compressed(
                        npz_file_path, hidden_state=hidden_state.cpu().numpy()
                    )

                    # Append metadata for tracking
                    result = {
                        "sample_id": sample_id,
                        "label": (
                            label.item() if isinstance(label, torch.Tensor) else label
                        ),
                        "npz_file": npz_file_path,
                    }
                    results.append(result)
                else:
                    # Pass the hidden states through the SAE model to get activations
                    sae_acts, reconstruction_error = analyze_with_sae(
                        sae, hidden_state.unsqueeze(0)
                    )

                    # Store the activations as an npz file
                    file_id = f"sample_{sample_id}"
                    npz_file_path = os.path.join(save_dir, f"{file_id}.npz")
                    np.savez_compressed(
                        npz_file_path,
                        sae_acts=sae_acts.cpu().numpy(),
                        hidden_state=hidden_state.cpu().numpy(),
                    )

                    # Append metadata for tracking
                    result = {
                        "sample_id": sample_id,
                        "label": (
                            label.item() if isinstance(label, torch.Tensor) else label
                        ),
                        "npz_file": npz_file_path,
                        "reconstruction_error": reconstruction_error.item(),
                    }
                    results.append(result)
            except Exception as e:
                logging.exception(f"Error processing sample ID {sample_id}")
                continue
    model_name = model_name.replace("/", "-")  # Replace '/' with '_' in model name
    # Append batch results to metadata CSV
    if activation_only:
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(
            save_dir, f"{model_name}_{layer}_activations_metadata.csv"
        )
    else:
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(
            save_dir, f"{model_name}_{layer}_sae_activations_metadata.csv"
        )
    if not os.path.isfile(results_csv_path):
        results_df.to_csv(results_csv_path, index=False)
    else:
        results_df.to_csv(results_csv_path, mode="a", header=False, index=False)

    return
