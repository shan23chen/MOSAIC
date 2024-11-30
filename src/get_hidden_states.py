# get_hidden_states.py -> step1_extract_all.py

import torch
from datasets import load_dataset
from models import load_models_and_tokenizer, prepare_inputs, extract_hidden_states
from retrieve_sae_features import retrieve_sae_features  # Import the function here
import logging
from tqdm import tqdm
import math
import gc
from datasets import concatenate_datasets
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
# retrieve_all_tokens = True

def get_valid_token(model_name, model_type, hidden_states, all_tokens):
    if model_type == "vlm":
        if model_name == "Intel/llava-gemma-2b":
            return hidden_states[:, 4:580, :]
        elif model_name == "google/paligemma-3b-mix-224":
            return hidden_states[:, 0:256, :]
    elif model_type == "llm" and all_tokens:
        # get all tokens' specific layer's hidden states
        return hidden_states
    else:
        # get only the last token's specific layer's hidden states
        # default setting for causal language models
        return hidden_states[:, -1:, :]



def get_hidden_states(
    model_name,
    model_type,
    layer,
    sae_location,
    width,
    batch_size=32,
    dataset_name=None,
    dataset_config_name=None,
    dataset_split="train",
    text_field=None,
    image_field=None,
    label_field="label",
    max_batches=None,
    save_dir="./npz_files",  # Add save_dir to pass to retrieve_sae_features
    activation_only=True,
    all_tokens=False,
):
    logging.info(f"Loading model and tokenizer/processor for {model_name}")
    # Load the model and tokenizer/processor
    model, tokenizer_or_processor, sae = load_models_and_tokenizer(
        model_name, layer, device, model_type, sae_location, width
    )
    model.eval()

    # Load the dataset
    logging.info(
        f"Loading dataset: {dataset_name}, config: {dataset_config_name}, split: {dataset_split}"
    )

    if dataset_name.split('/')[0] == 'local':
        # Extract the path after 'local/'
        path = dataset_name.split('local/')[1]+'.csv'
        
        # If path doesn't start with '/', make it absolute
        if not path.startswith('/'):
            path = '/' + path
            
        logging.info(f"Loading dataset from path: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found at: {path}")
        
        dataset = load_dataset("csv", data_files=path)["train"]
        print(dataset)
        

        
        # Initialize data files dictionary
        # data_files = {}
        # split_mapping = {
        #     'train': 'train.csv',
        #     'dev': 'dev.csv', 
        #     'test': 'test.csv'
        # }
        
        # # Check which files exist and load them
        # for split, filename in split_mapping.items():
        #     file_path = os.path.join(path, filename)
        #     if os.path.exists(file_path):
        #         data_files[split] = file_path
        
        # if not data_files:
        #     raise ValueError(f"No CSV files found in {path}")
            
        # # Load dataset from existing files
        # dataset_dict = load_dataset('csv', data_files=data_files)
        
        # # Add split column to each subset and concatenate
        # processed_datasets = []
        
        # for split_name, split_dataset in dataset_dict.items():
        #     # Add split column
        #     split_dataset = split_dataset.add_column(
        #         "split", 
        #         [split_name] * len(split_dataset)
        #     )
        #     processed_datasets.append(split_dataset)
        
        # # Concatenate all splits into one dataset
        # dataset = concatenate_datasets(processed_datasets)
    else:
        dataset = load_dataset(dataset_name, name=dataset_config_name, split=dataset_split)

    if "id" not in dataset.column_names:
        dataset = dataset.map(lambda example, idx: {"id": idx}, with_indices=True)

    def custom_collate_fn(batch):
        batch_out = {}
        keys = batch[0].keys()
        for key in keys:
            batch_out[key] = [sample[key] for sample in batch]
        return batch_out

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=custom_collate_fn
    )

    # Calculate total number of batches
    total_samples = len(dataset)
    total_batches = math.ceil(total_samples / batch_size)
    if max_batches is not None and max_batches < total_batches:
        total_batches = max_batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(data_loader, total=total_batches, desc="Processing Batches")
        ):
            if max_batches is not None and batch_idx >= max_batches:
                logging.info(
                    f"Reached maximum number of batches ({max_batches}). Stopping."
                )
                break
            try:
                # Retrieve texts and images based on specified field names
                if model_type == "vlm":
                    images = batch.get(image_field)
                    if images is None:
                        raise ValueError(
                            f"Image field '{image_field}' not found in the dataset."
                        )
                    texts = [""] * len(images)
                    inputs = prepare_inputs(
                        images, texts, tokenizer_or_processor, device, model_type
                    )
                else:
                    texts = batch.get(text_field)
                    if texts is None:
                        raise ValueError(
                            f"Text field '{text_field}' not found in the dataset."
                        )
                    inputs = prepare_inputs(
                        None, texts, tokenizer_or_processor, device, model_type
                    )

                # Extract hidden states for the current batch
                hidden_states = extract_hidden_states(model, inputs, layer, model_type)
                # print(type(hidden_states), hidden_states.size())
                hidden_states = get_valid_token(model_name, model_type, hidden_states, all_tokens)
                # print(type(hidden_states), hidden_states.size())
                labels = batch.get(label_field) # get label
                ids = batch.get("id")  # Get the unique ID/index for tracking 

                # Process and save the hidden states and labels immediately
                retrieve_sae_features(
                    model_name,
                    layer,
                    hidden_states,
                    labels,
                    ids,
                    sae,
                    save_dir=save_dir,
                    activation_only=activation_only,
                )
                # After processing the batch clear the memory
                del hidden_states, inputs
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logging.exception(f"Error processing batch {batch_idx + 1}")
                continue

    logging.info("Processing completed.")

    # Return path to the metadata file if needed
    return
