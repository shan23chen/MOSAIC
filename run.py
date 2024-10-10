import requests
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import pandas as pd
from sae_lens import SAE
from datasets import load_dataset
from tqdm import tqdm
import json
from torch.utils.data import DataLoader

# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

# Imports for displaying vis in Colab / notebook

torch.set_grad_enabled(False)

ds = "benjamin-paine/imagenet-1k-256x256"
# ds = "renumics/cifar100-enriched

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# Global configurations
checkpoint = "Intel/llava-gemma-2b"
# release = "gemma-2b-it"  # or "gemma-2b" depending on your use case
# layer = 12
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models_and_processor(checkpoint, layer, release, device):
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=f"{release}-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_post",
        device=device
    )
    return model, processor, sae

def prepare_image_inputs(images, texts, processor, device):
    if not isinstance(texts, list):
        texts = [texts] * len(images)
    prompts = []
    for text in texts:
        prompt = processor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': "<image>" + text}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    inputs = processor(text=prompts, images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs

def generate_text(model, inputs):
    generate_ids = model.generate(**inputs, max_length=1)
    outputs = model.processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return outputs

def extract_hidden_states(model, inputs, layer):
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True
    )

    # Accessing the encoder hidden states
    hidden_states = outputs.get('hidden_states', None)
    if hidden_states is None:
        raise ValueError("Hidden states not found in outputs")

    # hidden_states is a tuple of tensors, one per layer
    target_act = hidden_states[0][layer][:, 4:580, :].to(device)
    return target_act

def analyze_with_sae(sae, target_act):
    sae_acts = sae.encode(target_act.to(torch.float32))
    recon = sae.decode(sae_acts)
    reconstruction_error = 1 - torch.mean((recon[:, 1:] - target_act[:, 1:].to(torch.float32)) ** 2) / (target_act[:, 1:].to(torch.float32).var())
    return sae_acts, reconstruction_error

# Global variable to store the feature descriptions mapping
feature_descriptions_df = {}

def fetch_and_process_feature_descriptions(layer, sae):
    global feature_descriptions_df

    if layer not in feature_descriptions_df:
        url = "https://www.neuronpedia.org/api/explanation/export"
        payload = {
            "modelId": sae,  # "gemma-2b",
            "saeId": f"{layer}-res-jb"
        }
        headers = {"X-Api-Key": "YOUR_TOKEN"}
        response = requests.get(url, params=payload, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch feature descriptions: {response.text}")
        
        data = response.json()
        explanations_df = pd.DataFrame(data)
        explanations_df.rename(columns={"index": "feature"}, inplace=True)
        explanations_df["description"] = explanations_df["description"].apply(lambda x: x.lower())

        # Create a mapping from feature index to description
        feature_to_description = explanations_df.set_index('feature')['description'].to_dict()

        feature_descriptions_df[layer] = feature_to_description

def retrieve_feature_descriptions(inds, layer):
    global feature_descriptions_df

    if layer not in feature_descriptions_df:
        fetch_and_process_feature_descriptions(layer, release)

    feature_to_description = feature_descriptions_df[layer]

    batch_features = []
    for sample_inds in inds:
        sample_features = {}
        unique_inds = sample_inds.unique()
        for idx in unique_inds:
            idx_str = str(idx.item())
            description = feature_to_description.get(idx_str, None)
            if description:
                sample_features[idx_str] = description
        batch_features.append(sample_features)

    return batch_features

def process_dataset(dataset, batch_size=8, max_samples=None):

    model, processor, sae = load_models_and_processor(checkpoint, layer, release, device)
    model.eval()
    
    # If using mixed precision
    # model = model.half()

    # Fetch and process feature descriptions for the current layer
    fetch_and_process_feature_descriptions(layer, release)

    # Define a custom collate function
    def custom_collate_fn(batch):
        batch_out = {}
        keys = batch[0].keys()
        for key in keys:
            batch_out[key] = [sample[key] for sample in batch]
        return batch_out

    # Create DataLoader with the custom collate function
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    
    num_samples_processed = 0
    total_samples = max_samples if max_samples is not None else len(dataset)

    # Create a progress bar
    progress_bar = tqdm(total=total_samples, desc="Processing dataset")


    # Create a list to store the results
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                if ds == "benjamin-paine/imagenet-1k-256x256":
                    images = batch['image']
                    labels = batch.get('label', [''] * len(images))
                else:
                    images = batch['full_image']
                    fine_labels = batch.get('fine_label_str', [''] * len(images))
                    coarse_labels = batch.get('coarse_label_str', [''] * len(images))
                texts = [''] * len(images)  # Assuming no text input

                inputs = prepare_image_inputs(images, texts, processor, device)

                # If using mixed precision
                # inputs = inputs.half()

                # outputs = generate_text(model, inputs, processor)
                target_act = extract_hidden_states(model, inputs, layer)
                sae_acts, reconstruction_error = analyze_with_sae(sae, target_act)
                values, inds = sae_acts.max(-1)  # Shape: (batch_size, sequence_length)

                # Now retrieve features for each sample in the batch
                features = retrieve_feature_descriptions(inds, layer)  # List of features per sample

                if ds == "benjamin-paine/imagenet-1k-256x256":
                    for idx in range(len(images)):
                        result = {
                            'image_id': batch_idx * batch_size + idx,
                            'fine_label': labels[idx],
                            'retrieved_features': json.dumps(features[idx])  # Convert dict to JSON string
                        }
                        results.append(result)
                else:
                    # Store the results as a dictionary
                    for idx in range(len(images)):
                        result = {
                            'image_id': batch_idx * batch_size + idx,
                            'fine_label': fine_labels[idx],
                            'coarse_label': coarse_labels[idx],
                            'retrieved_features': json.dumps(features[idx])  # Convert dict to JSON string
                        }
                        results.append(result)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                if ds == "benjamin-paine/imagenet-1k-256x256":
                    for idx in range(len(images)):
                        results.append({
                            'image_id': batch_idx * batch_size + idx,
                            'fine_label': labels[idx],
                            'coarse_label': coarse_labels[idx],
                            'retrieved_features': None
                        })
                else:
                    for idx in range(len(images)):
                        results.append({
                            'image_id': batch_idx * batch_size + idx,
                            'fine_label': fine_labels[idx],
                            'coarse_label': coarse_labels[idx],
                            'retrieved_features': None
                        })
            finally:
                # Clear CUDA cache
                torch.cuda.empty_cache()

            # Update the progress bar
            progress_bar.update(len(images))

    # Close the progress bar
    progress_bar.close()

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    return df

# Load the dataset

if ds == "benjamin-paine/imagenet-1k-256x256":
    dataset = load_dataset(ds, split='validation')
else:
    dataset = load_dataset(ds, split='test')

# # Process the dataset
# processed_df = process_dataset(dataset, batch_size=32)

# # Save the DataFrame as CSV
# processed_df.to_csv(f"/home/shan/Desktop/multimodal/sae/cifar100-enriched_{release}_{layer}.csv", index=False)
# print("CSV file saved successfully!")

# Define the releases and layers to iterate through
releases = ["gemma-2b-it"]
layers = [12]#[0, 6, 10, 12]

# Iterate through each combination of release and layer
for release in releases:
    for layer in layers:
        print(f"Processing release: {release}, layer: {layer}")
        
        # Process the dataset
        processed_df = process_dataset(dataset, batch_size=32)
        
        # Save the DataFrame as CSV
        if ds == "benjamin-paine/imagenet-1k-256x256":
            output_file = f"/home/shan/Desktop/multimodal/sae/imagenet-1k-256x256_{release}_layer{layer}.csv"
        else:
            output_file = f"/home/shan/Desktop/multimodal/sae/cifar100-enriched_{release}_layer{layer}.csv"
        processed_df.to_csv(output_file, index=False)
        print(f"CSV file saved successfully: {output_file}")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()

print("All combinations processed successfully!")

