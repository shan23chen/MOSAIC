# Sparse Autoencoder Features for Classifications and Transferability

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-orange)](https://arxiv.org/abs/XXXX.XXXXX)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/your-model)
[![LessWrong Post](https://img.shields.io/badge/LessWrong-Cross_Modal_SAE's-red)](https://www.lesswrong.com/posts/your-post)

**MOSAIC** (Multilingual and Multimodal Observations of Sparse Autoencoders for Interpretable Classification) is a simple and efficient pipeline for extracting model activations, enabling you to fit linear probes or use sparse autoencoders to fit explainable classifiers such as decision trees and linear probes.

We use a straightforward YAML file structure to allow extraction across layers, different pooling methods, and even across languages or modalities.

## Project Overview

1. **VLLM/LLM Loading**: Load pre-trained Vision-Language or Language Models to obtain hidden states from specified layers.
2. **SAE Feature Extraction**: Pass the hidden states through a Sparse Autoencoder (SAE) to create sparse, interpretable feature representations. This approach allows the use of only specific extracted features for further classification.
3. **Classifier Integration**: Use the extracted SAE features as inputs to explainable classifiers, with visualizations planned to interpret how each feature contributes to the final predictions.

### Key Findings

- Binarization improves performance.
- Features transfer across languages.
- Features transfer from text-only to image tasks, e.g., Gemma to LLAVA.

## Prerequisites

To set up the required environment, follow these steps:

1. Create the environment from the `environment.yml` file:

   ```bash
   mamba env create -f environment.yml
   ```
2. Activate the environment:

   ```bash
   mamba activate saefari
   ```

## Step 1: Update the Configuration File

Update the `conf.yaml` file with the desired settings to configure the model, datasets, and classifier parameters.

### Configuration File Structure (`conf.yaml`)

```yaml
settings:
  base_save_dir: "./output/activations"
  base_classify_dir: "./output/classifications"
  batch_size: 1
  model_type: "llm"
  sae_location: "res"
  test_size: 0.2
  tree_depth: 5
  act_only: True
  cuda_visible_devices: "0"

models:
  - name: "google/gemma-2b"
    layers: [6, 12, 17]
    widths: ["16k"]

  # Additional models can be added here


datasets:
  - name: "Anthropic/election_questions"
    config_name: ""
    split: "test"
    text_field: "question"
    label_field: "label"

  # Additional datasets can be added here

classification_params:
  top_n_values: [0, 20, 50]
  binarize_values: [null, 1.0]
  extra_top_n: -1
  extra_binarize_value: null
```

## Step 2: Run the Experiments

Use the `run_experiments.py` script to run feature extraction and classification experiments.

### Script: `run_experiments.py`

This script allows you to extract hidden states from models and use them to train explainable classifiers. The script reads the configuration from the `conf.yaml` file and automates both extraction and classification processes.

#### Usage:

```bash
python run_experiments.py [--extract-only] [--classify-only]
```

- `--extract-only`: Run only the feature extraction process.
- `--classify-only`: Run only the classification process.

The script performs the following tasks:

1. **Feature Extraction**: Extracts hidden states from the specified layers of each model, processes them through the SAE, and saves the resulting activations.
2. **Classification**: Uses the extracted SAE features to train explainable classifiers. Various configurations of top-N values and binarization settings are explored to identify the optimal feature representations.

All data is saved into the `activations` or `classifications` directories.

## Step 3: Visualise Performance

You can visualize these differences using a simple Dash app configured to search according to your YAML directories.

You can visualize either specific configurations and top feature activations or compare performance across models and hyperparameters.

To run the visualization app, simply execute:

```bash
python app/main.py
```

This will open the app in a new tab.

### Example Workflow

1. Define model, dataset, and classification settings in `conf.yaml`.
2. Run `run_experiments.py` to extract features and classify them:

   ```bash
   python run_experiments.py
   ```
3. Run `app/main.py` to visualize the results.

## Directory Structure

- **src/**: Contains scripts for feature extraction (`step1_extract_all.py`) and classification (`step2_dataset_classify.py`).
- **output/**: Stores extracted activations (`activations`) and classification results (`classifications`).
- **app/**: Contains the Dash app (\`app.py\`)for visualization.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

BibTeX citation coming soon.
