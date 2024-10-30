# models.py

import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
from sae_lens import SAE
import logging
from typing import Tuple, Literal, Union

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models_and_tokenizer(
    checkpoint: str,
    layer: int,
    device: str,
    model_type: Literal["llm", "vlm"],
    sae_location: str,
    width: str,
) -> Tuple[
    Union[AutoModelForCausalLM, LlavaForConditionalGeneration],
    Union[AutoTokenizer, AutoProcessor],
    "SAE",
]:
    """
    Load model, tokenizer/processor, and SAE for a given checkpoint.

    Args:
        checkpoint: Model checkpoint name/path
        layer: Layer number to extract features from
        device: Device to load models on
        model_type: Type of model ('llm' or 'vlm')
        sae_location: SAE release type ('res' or 'mlp')

    Returns:
        Tuple of (model, tokenizer/processor, SAE)
    """
    if model_type == "llm":
        logging.info(f"Loading LLM model ++ {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    elif model_type == "vlm":
        logging.info(f"Loading VLM model ++ {checkpoint}")
        processor = AutoProcessor.from_pretrained(checkpoint)
        if "paligemma" in checkpoint.lower():
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                checkpoint, device_map="auto"
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                checkpoint, device_map="auto"
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.to(device)

    # Get SAE configuration based on model architecture
    sae_location, feature_id = get_sae_config(checkpoint, layer, sae_location, width)

    logging.info(f"Loading SAE model from release {sae_location}, feature {feature_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=f"{sae_location}",
        sae_id=feature_id,
        device=device,
    )

    if model_type == "llm":
        return model, tokenizer, sae
    else:
        return model, processor, sae


def get_sae_config(
    model_name: str, layer: int, sae_location: str, width: str
) -> Tuple[str, str]:
    """
    Generate SAE release name and feature ID based on model architecture.

    Args:
        model_name: Name/path of the model checkpoint
        layer: Layer number to extract features from
        sae_location: Type of SAE release ('res' or 'mlp')
        width: Width parameter for the SAE configuration (e.g. 16k)

    Returns:
        Tuple of (sae_location, feature_id)
    """
    model_name = model_name.lower()

    # Configuration dictionary for different model architectures
    model_configs = {
        "google/gemma-2-2b": {
            "release_template": "gemma-scope-2b-pt-{}-canonical",
            "id_template": "layer_{}/{}/canonical",
            "supported_widths": ["width_16k", "width_524k", "width_1m"],
        },
        "google/gemma-2-9b": {
            "release_template": "gemma-scope-9b-pt-{}-canonical",
            "id_template": "layer_{}/{}/canonical",
            "supported_widths": ["width_16k", "width_524k", "width_1m"],
        },
        "google/gemma-2-9b-it": {
            "release_template": "gemma-scope-9b-it-{}-canonical",
            "id_template": "layer_{}/{}/canonical",
            "supported_widths": ["width_16k", "width_524k", "width_1m"],
        },
    }

    # Determine which model architecture we're working with
    model_arch = next((arch for arch in model_configs if arch in model_name), None)
    if not model_arch:
        raise ValueError(f"Unsupported model architecture in checkpoint: {model_name}")

    config = model_configs[model_arch]

    # Validate width format
    if not width.startswith("width_"):
        width = f"width_{width}"

    if width not in config["supported_widths"]:
        raise ValueError(
            f"Unsupported width '{width}' for model {model_name}. "
            f"Supported widths are: {', '.join(config['supported_widths'])}"
        )

    # Generate release name and feature ID
    feature_type = "res" if "res" in sae_location else "mlp"
    sae_location = config["release_template"].format(feature_type)
    feature_id = config["id_template"].format(layer, width)

    return sae_location, feature_id


def prepare_inputs(images, texts, tokenizer_or_processor, device, model_type):
    if model_type == "vlm":
        if images is None:
            raise ValueError("Images are required for VLM models.")
        if texts is None:
            texts = [""] * len(images)
        return prepare_image_inputs(images, texts, tokenizer_or_processor, device)
    elif model_type == "llm":
        if texts is None:
            raise ValueError("Texts are required for LLM models.")
        # Directly tokenize the input texts
        inputs = tokenizer_or_processor(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        return inputs
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def prepare_image_inputs(images, texts, processor, device):
    if not isinstance(texts, list):
        texts = [texts] * len(images)
    prompts = []
    for text in texts:
        sanitized_text = text.strip().replace("\n", " ")
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<image>" + sanitized_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    inputs = processor(
        text=prompts, images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs


def extract_hidden_states(model, inputs, layer, model_type):
    if model_type == "llm":
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
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


def analyze_with_sae(sae, target_act):
    sae_acts = sae.encode(target_act.to(torch.float32))
    recon = sae.decode(sae_acts)
    mse = torch.mean((recon - target_act.to(torch.float32)) ** 2)
    var = torch.var(target_act.to(torch.float32))
    reconstruction_error = 1 - mse / var if var > 0 else torch.tensor(0.0)
    return sae_acts, reconstruction_error
