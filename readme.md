Explainable Classifier with SAE Features from VLLM/LLM

This project aims to create an explainable classifier by leveraging extracted hidden states from a Vision-Language Model (VLM) or Language Model (LLM), mapping these hidden states to features using a Sparse Autoencoder (SAE). The classifier built on these features will support model interpretability through feature-level insights.

## Project Overview

1. **VLLM/LLM Loading**: Load pre-trained Vision-Language or Language Models to obtain hidden states from specified layers.
2. **SAE Feature Extraction**: Pass the hidden states through a Sparse Autoencoder (SAE) to create sparse, interpretable feature representations. This approach allows the use of only specific extracted features for further classification.
3. **Classifier Integration**: The extracted SAE features will serve as inputs to explainable classifiers, with planned visualizations to interpret how each feature contributes to the final predictions.

## Repository Structure

### Files

- **`models.py` and get_hidden_states `.py`**: Contains functions for loading the model, extracting hidden states, processing these states via the SAE, and generating sparse, interpretable features.
- Sample run file at: *sae_llava/src/run_ds.sh*
- **`process_npz_files.py`**: Handles batch processing of `.npz` files, where each file contains hidden state information and SAE activations for a dataset. This script organizes the data, making it ready for feature extraction and classification tasks.
- **`playground.ipynb`**: A step-by-step Jupyter Notebook for exploring feature extraction, examining SAE activations, and running experiments. This notebook is ideal for interactive testing, visualization, and preliminary analyses.

### Important Functions

- **Model Loading**:

  - The `llm_type` function categorizes the model as either Vision-Language (`vlm`) or Language Model (`llm`).
  - The `load_sae` function loads the Sparse Autoencoder model for a specific layer, allowing targeted feature extraction based on the desired layer configuration.
- **Feature Extraction**:

  - **Hidden States Extraction**: The chosen layer's hidden states are loaded from the VLLM/LLM model, enabling a foundation for sparse feature extraction.
  - **SAE Mapping**: Hidden states are passed through the Sparse Autoencoder (`SAE`), producing a sparse vector representation. These sparse vectors serve as features for classifiers.
  - **Binarization**: `binarize_tensor` and `optimized_top_n_to_one_hot` functions generate sparse binary representations, retaining only the top features to reduce storage needs and improve interpretability.
- **Data Handling**:

  - `load_npz_data`: Loads `.npz` files containing hidden states and metadata, allowing batch processing of samples. `.npz` data structure makes it easy to retrieve sample-specific hidden states and SAE activations.

## Storage Management

The Sparse Autoencoder (SAE) activations are stored selectively to manage storage efficiently, given that storing all SAE activations is memory-intensive. Only the necessary activations and sparse representations required for classification are saved to optimize space.


# Running SAEfari Scripts

This repository contains scripts for analyzing models using Sparse Autoencoders (SAE). The process consists of two main steps: extracting hidden states and processing the results.

## Prerequisites

1. Create the environment from the `environment.yml` file:

   ```bash
   mamba env create -f environment.yml
  ```

2. Activate the environment:

   ```bash
   mamba activate saefari
   ```

## Step 1: Extract Hidden States (run.py)

This script extracts hidden states from either Language Models (LLM) or Vision Language Models (VLM).

### For Language Models

```bash
python run.py \
    --model_name google/gemma-2b-it \
    --model_type llm \
    --sae_release gemma-2b \
    --layer 12 \
    --checkpoint google/gemma-2b-it \
    --save_dir ./output_llm_both \
    --dataset_name shanchen/OncQA \
    --dataset_split train \
    --text_field question \
    --image_field NA \
    --label_field q1 \
    --act_only False \
    --max_batches 3
```

### For Vision Language Models

```bash
python run.py \
    --model_name Intel/llava-gemma-2b \
    --model_type vlm \
    --sae_release gemma-2b \
    --layer 12 \
    --checkpoint Intel/llava-gemma-2b \
    --save_dir ./output_vlm_both \
    --dataset_name renumics/cifar100-enriched \
    --dataset_split test \
    --text_field fine_label_str \
    --image_field full_image \
    --label_field fine_label_str \
    --act_only False \
    --max_batches 3
```

## Step 2: Process Results (process_npz_files.py)

After extracting hidden states, process the results and generate visualizations:

```bash
python process_npz_files.py \
    --input-dir ./output_llm_both \
    --model-name google/gemma-2b-it \
    --model-type llm \
    --layer 12 \
    --sae-release gemma-2b \
    --top-n 5 \
    --output-dir processed_features_llm \
    --test-size 0.2 \
    --tree-depth 5 \
    --save-plots
```

## Key Parameters

- `--model_name`: Name/path of the model (e.g., google/gemma-2b-it)
- `--model_type`: Type of model (llm or vlm)
- `--sae_release`: SAE release name
- `--layer`: Layer to extract hidden states from
- `--save_dir`/`--output-dir`: Directory for output files
- `--dataset_name`: HuggingFace dataset to use
- `--act_only`: Whether to store activations only (True) or both SAE and activations (False)
- `--max_batches`: Limit number of batches (useful for testing)

## Batch Processing

For processing multiple configurations, run a shell script:

   ```bash
   cd src
   chmod +x run_ds.sh
   bash run_ds.sh
   ```

## Output

The scripts will generate:
- Hidden state NPZ files
- Processed sae features
- Classification results for linear probe and decision tree
- Visualization plots (if --save-plots is enabled)
- Interactive dashboard (automatically launches in browser)


## To-Do and Next Steps

- [ ] update dashboard to show hidden vs sae feature results
- [ ] update training settings 
- [ ] add explanations for hover etc on sae
- [ ] all positions
- [ ] judger
  - [ ] get output
  - [ ] vllm
  - [ ] judge output 
  - [ ] store

## Citation
TBD
