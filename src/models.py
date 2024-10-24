# models.py

import torch
from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, PaliGemmaForConditionalGeneration
from sae_lens import SAE
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models_and_tokenizer(checkpoint, layer, device, model_type, sae_release):
    if model_type == "llm":
        logging.info(f"Loading LLM model ++ {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    elif model_type == "vlm":
        logging.info("Loading VLM model ++ {checkpoint}")
        processor = AutoProcessor.from_pretrained(checkpoint)
        if 'paligemma' in checkpoint:
            model = PaliGemmaForConditionalGeneration.from_pretrained(checkpoint, device_map="auto") 
        else:
            model = LlavaForConditionalGeneration.from_pretrained(checkpoint, device_map="auto")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.to(device)

    logging.info(f"Loading SAE model from release {sae_release}, layer {layer}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=f"{sae_release}-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_post",
        device=device
    )
    if model_type == "llm":
        return model, tokenizer, sae
    else:
        return model, processor, sae

def prepare_inputs(images, texts, tokenizer_or_processor, device, model_type):
    if model_type == "vlm":
        if images is None:
            raise ValueError("Images are required for VLM models.")
        if texts is None:
            texts = [''] * len(images)
        return prepare_image_inputs(images, texts, tokenizer_or_processor, device)
    elif model_type == "llm":
        if texts is None:
            raise ValueError("Texts are required for LLM models.")
        # Directly tokenize the input texts
        inputs = tokenizer_or_processor(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        return inputs
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def prepare_image_inputs(images, texts, processor, device):
    if not isinstance(texts, list):
        texts = [texts] * len(images)
    prompts = []
    for text in texts:
        sanitized_text = text.strip().replace('\n', ' ')
        prompt = processor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': "<image>" + sanitized_text}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    inputs = processor(
        text=prompts,
        images=list(images),
        return_tensors="pt",
        padding=True
    ).to(device)
    return inputs

def extract_hidden_states(model, inputs, layer, model_type):
    if model_type == "llm":
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states
        target_act = hidden_states[layer].to(device)
    else:  # For VLM models
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True
        )
        hidden_states = outputs.get('hidden_states', None)
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
