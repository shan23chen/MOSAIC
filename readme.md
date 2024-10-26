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

## To-Do and Next Steps

1. **Classifier Connection**: Connect extracted SAE features to an explainable classifier.
2. **Classifier Visualizations**: Develop visualizations to interpret classifier outputs, enabling insight into feature importance and the classifier’s decision-making process.

This work focuses on transforming hidden layers from VLMs/LLMs into sparse, interpretable features, allowing for a more explainable approach to model-based classification tasks. The step-by-step structure in `playground.ipynb` and batch functionality in `process_npz_files.py` ensure that this pipeline is both flexible and scalable.

## Citation
TBD
