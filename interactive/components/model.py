import base64
import io

import json

import joblib

from PIL import Image

import torch

import numpy as np

from transformers import LlavaForConditionalGeneration, AutoProcessor
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(args):
    if "it" in args.vlm_model:
        sae_release = "gemma-2b-it"
    else:
        sae_release = "gemma-2b"

    model = LlavaForConditionalGeneration.from_pretrained(args.vlm_model, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    sae, _, _ = SAE.from_pretrained(
            release=f"{sae_release}-res-jb",
            sae_id=f"blocks.{args.sae_layer}.hook_resid_post",
            device=device,
    )

    with open(args.neuronpedia_cache, "r") as f:
        neuron_cache = json.load(f)

    model.eval()
    sae.eval()

    linear_predictor_file  = joblib.load(args.classifier)

    classifier = linear_predictor_file["model"]
    label_encoder = linear_predictor_file["label_encoder"]

    return model, processor, sae, neuron_cache, classifier, label_encoder

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


def analyze_with_sae(sae, target_act):
    sae_acts = sae.encode(target_act.to(torch.float32))
    recon = sae.decode(sae_acts)
    mse = torch.mean((recon - target_act.to(torch.float32)) ** 2)
    var = torch.var(target_act.to(torch.float32))
    reconstruction_error = 1 - mse / var if var > 0 else torch.tensor(0.0)
    return sae_acts, reconstruction_error

def get_valid_token(model_name, model_type, hidden_states, all_tokens):
    if model_type == "vlm":
        if model_name == "Intel/llava-gemma-2b":
            return hidden_states[:, 4:580, :]
        elif model_name == "google/paligemma-3b-mix-224":
            return hidden_states[:, 0:256, :]
    elif model_type == "llm" and all_tokens:
        return hidden_states
    else:
        # get only the last token's specific layer's hidden states
        # default setting for causal language models
        return hidden_states[:, -1:, :]

def extract_hidden_states(model, inputs, layer, model_type):
    # TODO @ shan: make sure load and process function hook to different position nex week
    if model_type == "llm":
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        print(f"Hidden states shape: {len(hidden_states)}")
        target_act = hidden_states[layer].to(device)
    else:  # For VLM models
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.get("hidden_states", None)
        if hidden_states is None:
            raise ValueError("Hidden states not found in outputs")
        # Get the hidden states at the specified layer
        target_act = hidden_states[0][layer].to(device)
    
    return target_act

def process_image(contents, model_name, model, processor, sae, neuron_cache, sae_layer, classifier, label_encoder, top_n):
    # Decode the image
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    # Preprocess image and prepare it for the VLM model
    text = ""
    sanitized_text = text.strip().replace("\n", " ")
    prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<image>" + sanitized_text}],
            tokenize=False,
            add_generation_prompt=True,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    hidden_state = extract_hidden_states(model, inputs, sae_layer, "vlm")

    hidden_state = get_valid_token(model_name, "vlm", hidden_state, False)

    # Pass VLM features through the SAE
    sae_acts, reconstruction_error = analyze_with_sae(sae, hidden_state.unsqueeze(0))
    
    sae_acts = sae_acts[0][0].detach().to("cpu")

    top_n_sae = optimized_top_n_to_one_hot(sae_acts, top_n)

    prediction = label_encoder.inverse_transform(classifier.predict(top_n_sae.reshape(1, -1)))

    feature_importances = classifier.coef_ * top_n_sae

    feature_importances = feature_importances[0]

    top_10_features = np.argsort(feature_importances)[:10]

    top_10_names = [neuron_cache[str(x)] for x in top_10_features]

    top_10_scores = [feature_importances[x] for x in top_10_features]

    return top_10_names, top_10_scores