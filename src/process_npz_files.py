# process_npz_files.py

import os
import numpy as np
import pandas as pd
from datasets import load_dataset

def process_npz_files(metadata_csv_path, dataset_name, dataset_config_name=None, dataset_split='train'):
    # Load the metadata CSV
    metadata_df = pd.read_csv(metadata_csv_path)

    # Load the original dataset
    dataset = load_dataset(dataset_name, name=dataset_config_name, split=dataset_split)

    # Ensure the dataset has the 'id' field
    if 'id' not in dataset.column_names:
        dataset = dataset.map(lambda example, idx: {'id': idx}, with_indices=True)

    # Convert the dataset to a Pandas DataFrame
    dataset_df = dataset.to_pandas()

    # Merge the metadata with the dataset on 'id' and 'sample_id'
    merged_df = pd.merge(dataset_df, metadata_df, left_on='id', right_on='sample_id')

    # Iterate over the merged DataFrame
    for idx, row in merged_df.iterrows():
        sample_id = row['sample_id']
        npz_file = row['npz_file']
        label = row['label']
        reconstruction_error = row['reconstruction_error']

        # Load the SAE activations
        data = np.load(npz_file)
        sae_acts = data['sae_acts']

        # Perform your analysis or processing
        # Example: Print the shape of the activations
        print(f"Sample ID: {sample_id}")
        print(f"Label: {label}")
        print(f"SAE Activations Shape: {sae_acts.shape}")
        print(f"Reconstruction Error: {reconstruction_error}")
        print("-" * 40)

        # Optionally, access original dataset fields
        # For example, if the dataset has a 'text' field
        # text = row.get('text')  # Replace 'text' with the actual field name
        # print(f"Original Text: {text}")

        # ... your processing code ...

if __name__ == "__main__":
    metadata_csv_path = 'npz_files/sae_features_metadata.csv'
    dataset_name = 'your_dataset_name'  # Replace with your dataset name
    dataset_config_name = None  # Replace with your dataset config name if applicable
    dataset_split = 'train'  # Or 'validation' / 'test'

    process_npz_files(metadata_csv_path, dataset_name, dataset_config_name, dataset_split)
