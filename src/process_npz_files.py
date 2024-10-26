import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
import logging

# Set device configuration for computation
device = "cuda" if torch.cuda.is_available() else "cpu"
last_token = False

def llm_type(model_name):
    """
    Determines the type of model (either "vlm" or "llm") based on the model name.
    
    Parameters:
    model_name (str): Name of the model.
    
    Returns:
    str: Model type ("vlm" for vision-language model or "llm" for language model).
    """
    if "llava" in model_name.lower() or "peli" in model_name.lower():
        return "vlm"
    elif "gemma" in model_name.lower():
        return "llm"
    else:
        raise ValueError(f"Model name '{model_name}' is not supported")

# Load the SAE model function
def load_sae(layer, device, sae_release):
    """
    Loads a pre-trained SAE model for a specific layer.
    
    Parameters:
    layer (str): Layer name or index.
    device (str): Device to load the model onto (CPU or CUDA).
    sae_release (str): Release version of the SAE model.
    
    Returns:
    Tuple: SAE model, configuration dictionary, and sparsity level.
    """
    logging.info(f"Loading SAE model from release '{sae_release}', layer '{layer}'")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=f"{sae_release}-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_post",
        device=device
    )
    return sae, cfg_dict, sparsity

# Tensor binarization function
def binarize_tensor(tensor):
    """
    Binarizes a tensor by setting all non-zero values to 1, keeping 0s unchanged.
    
    Parameters:
    tensor (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Binarized tensor.
    """
    return (tensor != 0).to(torch.int32)

# Sparse one-hot conversion function
def optimized_top_n_to_one_hot(tensor, top_n, binary=False):
    """
    Converts each row of the input tensor to a sparse one-hot representation.
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape (token_length, dimension_size).
    top_n (int): Number of top values to retain per row.
    binary (bool): If True, binarizes the output tensor.
    
    Returns:
    torch.Tensor: Sparsified tensor with top_n values retained.
    """
    token_length, dim_size = tensor.shape
    sparse_tensor = torch.zeros_like(tensor)
    top_n_indices = torch.topk(tensor, top_n, dim=1).indices
    sparse_tensor.scatter_(1, top_n_indices, 1)

    if binary:
        sparse_tensor = binarize_tensor(sparse_tensor.sum(dim=0).to(device))
    else:
        sparse_tensor = sparse_tensor.sum(dim=0).to(device)
    return sparse_tensor

# Hidden state to sparse vector transformation
def hidden2sparse(data, sae):
    """
    Encodes a given numpy array into a sparse vector using the SAE model.
    
    Parameters:
    data (np.ndarray): Input hidden state data.
    sae (SAE): Loaded SAE model.
    
    Returns:
    torch.Tensor or None: Encoded sparse tensor, or None if data is None.
    """
    if data is not None:
        return sae.encode(torch.tensor(data).to(device).to(torch.float32))
    return None

# Function to process each row in the metadata DataFrame
def load_npz_data(index, metadata_df):
    """
    Loads sample data from an NPZ file based on the metadata index.
    
    Parameters:
    index (int): Index of the row in the metadata DataFrame.
    metadata_df (pd.DataFrame): DataFrame containing metadata for each sample.
    
    Returns:
    dict: Dictionary with sample_id, label, sae_acts, and hidden_state.
    """
    sample_id = metadata_df.at[index, 'sample_id']
    label = metadata_df.at[index, 'label']
    npz_file_path = metadata_df.at[index, 'npz_file']
    npz_data = np.load(npz_file_path)
    
    return {
        'sample_id': sample_id,
        'label': label,
        'sae_acts': npz_data.get('sae_acts', None),
        'hidden_state': npz_data['hidden_state']
    }

# Main processing function
def main(model_name, metadata_csv_path, checkpoint, top_n=5):
    """
    Main function to load model, process data, and extract features.
    
    Parameters:
    model_name (str): Name of the model.
    metadata_csv_path (str): Path to the metadata CSV file.
    checkpoint (str): Checkpoint name for the model.
    top_n (int): Number of top values to retain in one-hot encoding.
    """
    # Extract layer info from metadata path and load DataFrame
    layer = metadata_csv_path.split('/')[-1].split('_')[1]
    metadata_df = pd.read_csv(metadata_csv_path)
    
    # Load SAE model and configuration
    sae, cfg_dict, sparsity = load_sae(layer, device, checkpoint)

    # Load data for all indices and convert to DataFrame
    data_samples = [load_npz_data(idx, metadata_df) for idx in range(len(metadata_df))]
    loaded_data_df = pd.DataFrame(data_samples)

    # Process 'sae_acts' or transform hidden states if absent
    if 'sae_acts' in loaded_data_df.columns:
        if loaded_data_df['sae_acts'][0].shape[-1] != cfg_dict['d_sae']:
            print(f"SAE activations shape mismatch: {loaded_data_df['sae_acts'][0].shape[-1]} != {cfg_dict['d_sae']}")
        else:
            if loaded_data_df['sae_acts'][0].shape[0] == 1:
                loaded_data_df['sae_acts'] = loaded_data_df['sae_acts'].apply(lambda x: x[0])
    else:
        if last_token:
            # Extract last token and stack into tensor
            last_tokens = [state[-1] for state in loaded_data_df['hidden_state']]
            hidden_tensor = torch.tensor(np.stack(last_tokens)).to(torch.float32).to(device)
            output = hidden2sparse(hidden_tensor, sae)
        else:
            loaded_data_df['sae_acts'] = loaded_data_df['hidden_state'].apply(lambda x: hidden2sparse(x, sae))

    # Apply sparse one-hot conversion for feature extraction
    loaded_data_df['features'] = loaded_data_df['sae_acts'].apply(lambda x: optimized_top_n_to_one_hot(torch.tensor(x), top_n))

    # Display final feature set
    print(loaded_data_df[['sample_id', 'label', 'features']].head())

# Run main if this file is executed as a script
if __name__ == "__main__":
    # Example usage parameters
    model_name = "google-gemma-2b-it"
    metadata_csv_path = "output_llm_both/google-gemma-2b-it_12_sae_activations_metadata.csv"
    checkpoint = "gemma-2b"
    
    main(model_name, metadata_csv_path, checkpoint)
g