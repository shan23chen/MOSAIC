#!/usr/bin/env python3

import os
import re
import torch
import math
from typing import Dict, Any, List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################################################
# 1. DATASET CONFIGS
##############################################################################


def preprocess_pubmed_qa(example: Dict[str, Any]) -> Dict[str, Any]:
    possible_answers = ["yes", "no", "maybe"]
    raw_answer = example.get("answer", "")
    # if list, take first element
    if isinstance(raw_answer, list):
        raw_answer = raw_answer[0].lower()
    try:
        correct_idx = possible_answers.index(raw_answer)
    except ValueError:
        correct_idx = -1
    return {
        "id": str(example["id"]),
        "question": example["question"],
        "context": example.get("context", ""),
        "choices": possible_answers,
        "correct_idx": correct_idx,
    }


def preprocess_boolq(example: Dict[str, Any]) -> Dict[str, Any]:
    # For BoolQ: answer is boolean => (False => "no", True => "yes")
    choices = ["no", "yes"]
    bool_answer = example.get("answer", False)
    correct_idx = 1 if bool_answer else 0
    ex_id = str(example["id"]) if "id" in example else "N/A"
    return {
        "id": ex_id,
        "question": example["question"],
        "context": example["passage"],
        "choices": choices,
        "correct_idx": correct_idx,
    }


def preprocess_piqa(example: Dict[str, Any]) -> Dict[str, Any]:
    goal = example.get("goal", "")
    sol1 = example.get("sol1", "")
    sol2 = example.get("sol2", "")
    label = example.get("label", 0)  # 0 => sol1 correct, 1 => sol2 correct
    ex_id = str(example["id"]) if "id" in example else "N/A"
    return {
        "id": ex_id,
        "question": goal,
        "context": "",  # No separate context in PIQA
        "choices": [sol1, sol2],
        "correct_idx": label,
    }


DATASET_CONFIGS = {
    "pubmed_qa": {
        "huggingface_id": "bigbio/pubmed_qa",
        "subset": "pubmed_qa_labeled_fold0_bigbio_qa",
        "split": "test",
        "preprocess_fn": preprocess_pubmed_qa,
    },
    "boolq": {
        "huggingface_id": "boolq",
        "subset": None,
        "split": "validation",
        "preprocess_fn": preprocess_boolq,
    },
    "piqa": {
        "huggingface_id": "piqa",
        "subset": None,
        "split": "validation",
        "preprocess_fn": preprocess_piqa,
    },
}

##############################################################################
# 2. UTILITY: NAME CLEANING
##############################################################################


def clean_name(name: str) -> str:
    """
    Replace all non-alphanumeric or underscore characters with underscores
    so we can form a valid repository name on Hugging Face.
    """
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)


##############################################################################
# 3. LOAD + PREPROCESS
##############################################################################


def load_and_preprocess_dataset(task_name: str) -> List[Dict[str, Any]]:
    if task_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown task_name: {task_name}")

    config = DATASET_CONFIGS[task_name]
    ds_name = config["huggingface_id"]
    subset = config["subset"]
    split = config["split"]
    preprocess_fn = config["preprocess_fn"]

    if subset:
        raw_ds = load_dataset(ds_name, subset, split=split)
    else:
        raw_ds = load_dataset(ds_name, split=split)

    preprocessed_data = []
    for ex in raw_ds:
        item = preprocess_fn(ex)
        preprocessed_data.append(item)
    return preprocessed_data


##############################################################################
# 4. CREATE PROMPT
##############################################################################


def create_mcq_prompt(question: str, context: str, choices: List[str]) -> str:
    prompt = ""
    if context.strip():
        prompt += f"Context: {context}\n\n"
    prompt += f"Question: {question}\n\nOptions:\n"
    for i, ch in enumerate(choices):
        prompt += f"{i+1}. {ch}\n"
    prompt += "\nAnswer:"
    return prompt


##############################################################################
# 5. LOAD MODEL
##############################################################################


def get_model_device_config():
    """
    Return a config for single-GPU usage on GPU 0.
    - device_map={"": "cuda:0"} means the entire model is on GPU 0.
    - If no GPU is available, fall back to CPU.
    """
    if not torch.cuda.is_available():
        return {"device_map": "cpu", "torch_dtype": torch.float16}, "cpu"

    # Single GPU config
    config = {
        "device_map": {"": "cuda:0"},  # could also do device_map="cuda:0"
        "torch_dtype": torch.float16,
    }
    return config, "cuda:0"


def clear_gpu_memory():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f"cuda:{i}"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i} after clearing:")
            print(f"  Allocated: {memory_allocated:.2f}GB")
            print(f"  Reserved:  {memory_reserved:.2f}GB")


##############################################################################
# 6. LOGPROB CALCULATION
##############################################################################


def compute_choice_logprob(prefix: str, choice: str) -> float:
    prefix_tokens = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        choice_tokens = tokenizer(choice, add_special_tokens=False, return_tensors="pt")

    # Move to GPU 0 explicitly
    prefix_tokens = {k: v.to("cuda:0") for k, v in prefix_tokens.items()}
    choice_tokens = {k: v.to("cuda:0") for k, v in choice_tokens.items()}

    # Concatenate
    input_ids = torch.cat(
        [prefix_tokens["input_ids"], choice_tokens["input_ids"]], dim=1
    )
    attention_mask = torch.cat(
        [prefix_tokens["attention_mask"], choice_tokens["attention_mask"]], dim=1
    )

    # labels = input_ids as well, so that also ends up on GPU
    labels = input_ids

    # Forward pass on GPU
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # same as input_ids
        )

    logits = outputs.logits[0, :-1, :]
    target_ids = input_ids[0, 1:]

    prefix_len = prefix_tokens["input_ids"].shape[1]
    choice_len = choice_tokens["input_ids"].shape[1]
    choice_start = prefix_len
    choice_end = prefix_len + choice_len

    logprob_sum = 0.0
    for pos in range(choice_start, choice_end):
        token_logits = logits[pos - 1]
        token_id = target_ids[pos - 1]
        logprob_sum += torch.log_softmax(token_logits, dim=-1)[token_id].item()

    return logprob_sum


##############################################################################
# 7. EVALUATION + PUSH
##############################################################################


def evaluate_and_push(
    data: List[Dict[str, Any]], org_name: str, model_name: str, task_name: str
):
    clean_model = clean_name(model_name)
    clean_task = clean_name(task_name)
    repo_id = f"{org_name}/{clean_model}_{clean_task}"

    results_list = []
    for ex in data:
        ex_id = ex["id"]
        question = ex["question"]
        context = ex["context"]
        choices = ex["choices"]
        correct_idx = ex["correct_idx"]

        # Build prompt
        prompt = create_mcq_prompt(question, context, choices)

        # Compute logprobs for each choice
        log_probs = []
        for ch in choices:
            lp = compute_choice_logprob(prefix=prompt, choice=ch)
            log_probs.append(lp)

        # Best choice is the one with the highest logprob
        best_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
        best_choice = choices[best_idx]

        # If correct_idx is valid, retrieve that choice; else "N/A"
        if 0 <= correct_idx < len(choices):
            correct_answer = choices[correct_idx]
        else:
            correct_answer = "N/A"

        # Determine correctness
        if 0 <= correct_idx < len(choices):
            model_score = "Correct" if best_idx == correct_idx else "Incorrect"
        else:
            model_score = "N/A"

        # Convert list of log probs to a string: "choice0:-5.1; choice1:-3.2"
        choice_logits_str = "; ".join(
            f"{choices[i]}:{log_probs[i]:.4f}" for i in range(len(choices))
        )

        row = {
            "id": ex_id,
            "input_prompt": prompt,
            "correct_answer": correct_answer,
            "choice_logits": choice_logits_str,
            "best_choice": best_choice,
            "model_score": model_score,  # <-- Added field
        }
        results_list.append(row)

    results_dataset = Dataset.from_list(results_list)
    print(f"\nPushing dataset to HF Hub => {repo_id}")
    results_dataset.push_to_hub(repo_id, private=False)
    print("Push complete!\n")


##############################################################################
# 8. MAIN
##############################################################################


def evaluate_model(model_name: str, tokenizer, model, org: str):
    """Evaluate a single model on all datasets"""
    print(f"\n=== Evaluating {model_name} ===")

    # 1) PubMedQA
    print("\n== Evaluating PubMedQA ==")
    pubmed_data = load_and_preprocess_dataset("pubmed_qa")
    evaluate_and_push(pubmed_data, org, model_name, "pubmed_qa")

    # 2) BoolQ
    print("\n== Evaluating BoolQ ==")
    boolq_data = load_and_preprocess_dataset("boolq")
    evaluate_and_push(boolq_data, org, model_name, "boolq")

    # 3) PIQA
    print("\n== Evaluating PIQA ==")
    piqa_data = load_and_preprocess_dataset("piqa")
    evaluate_and_push(piqa_data, org, model_name, "piqa")


if __name__ == "__main__":
    org = "AIM-Harvard"

    gemma_models = [
        # "google/gemma-2-2b",
        # "google/gemma-2-9b",
        "google/gemma-2-9b-it",
    ]

    for model_name in gemma_models:
        try:
            print(f"\nLoading model: {model_name}")
            torch.cuda.empty_cache()
            device_config, primary_device = get_model_device_config()

            # Load model + tokenizer
            global tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            global model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **device_config,  # device_map={"": "cuda:0"} + torch_dtype=float16
            )
            model.eval()
            model.to(primary_device)

            # Print memory usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                    print(f"GPU {i} memory allocated: {memory_allocated:.2f}GB")

            # Evaluate
            evaluate_model(model_name, tokenizer, model, org)

            # Cleanup
            del model
            del tokenizer
            clear_gpu_memory()

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

    print("\nAll models evaluated!")
