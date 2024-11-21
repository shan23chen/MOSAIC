import logging
import subprocess
import time
import os
import psutil
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from vllm import LLM, SamplingParams
import requests
import time
import signal
from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, multiple_choice
from inspect_ai.scorer import (
    includes,
    match,
    pattern,
    answer,
    exact,
    f1,
    model_graded_qa,
    model_graded_fact,
    choice,
)
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    model: str
    dataset: str
    split: str
    config: Optional[str]
    input_column: str
    label_column: str
    id_column: str
    num_samples: int
    output_dir: Path
    system_prompt: str
    grader_model: str
    debug: bool
    eval_type: str
    scorer_name: str
    scorer_args: Dict[str, Any]
    choice_columns: Optional[List[str]] = None
    target_mapping: Optional[Dict[int, str]] = None


# Multiple choice template
MC_TEMPLATE = """
Answer the following multiple choice question. Your final answer must be in the format 'ANSWER: X' where X is one of the choices provided (A, B, C, or D).
Think through the question step by step before providing your answer.

Question: {question}

Choices:
{choices}
""".strip()


def get_unique_label_choices(dataset, config: EvalConfig) -> List[str]:
    """Get unique choices from the label column."""
    unique_labels = sorted(set(str(item[config.label_column]) for item in dataset))
    logger.info(f"Extracted choices from label column: {unique_labels}")
    return unique_labels


def format_choices(choices: List[str]) -> str:
    """Format multiple choice options."""
    return "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", choices))


def get_choice_letter(value: str, choices: List[str]) -> str:
    """Get the letter (A, B, C, D) corresponding to the choice."""
    try:
        idx = choices.index(value)
        return "ABCD"[idx]
    except (ValueError, IndexError):
        raise ValueError(f"Value '{value}' not found in choices: {choices}")


def process_dataset_item(
    item: Dict[str, Any], config: EvalConfig, global_choices: Optional[List[str]] = None
) -> Optional[Sample]:
    """Process a single dataset item into a Sample."""
    try:
        # Validate required columns
        for col in [config.input_column, config.label_column]:
            if col not in item:
                logger.error(f"Required column '{col}' not found in dataset item")
                return None

        item_id = str(item.get(config.id_column, ""))
        input_text = str(item[config.input_column])

        if config.eval_type == "classification":
            # Get choices either from choice_columns or use global choices
            choices = (
                [str(item[col]) for col in config.choice_columns]
                if config.choice_columns
                else global_choices
            )

            if not choices:
                raise ValueError("No choices available for classification task")

            # Format question with choices
            formatted_question = MC_TEMPLATE.format(
                question=input_text, choices=format_choices(choices)
            )

            # Get the actual label value as string
            label_value = str(item[config.label_column])

            # Find which letter (A, B, C, D) corresponds to this label
            target = get_choice_letter(label_value, choices)

            return Sample(
                input=[
                    ChatMessageSystem(content=config.system_prompt),
                    ChatMessageUser(content=formatted_question),
                ],
                target=label_value,  # Keep original label for scoring
                choices=choices,
                id=item_id,
            )
        else:  # open_ended
            return Sample(
                input=[
                    ChatMessageSystem(content=config.system_prompt),
                    ChatMessageUser(content=formatted_question),
                ],
                target=str(item[config.label_column]),
                id=item_id,
            )

    except Exception as e:
        logger.error(f"Error processing dataset item: {e}")
        if config.debug:
            raise
        return None


def load_dataset_samples(config: EvalConfig) -> List[Sample]:
    """Load and process dataset samples."""
    try:
        dataset = load_dataset(config.dataset, config.config, split=config.split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    if config.debug:
        dataset = dataset.select(range(min(10, len(dataset))))

    # Get global choices only if it's a classification task without choice columns
    global_choices = None
    if config.eval_type == "classification" and not config.choice_columns:
        global_choices = get_unique_label_choices(dataset, config)
        logger.info(f"Using global choices for classification: {global_choices}")

    samples = []
    for i, item in enumerate(dataset):
        if i >= config.num_samples:
            break

        sample = process_dataset_item(item, config, global_choices)
        if sample:
            samples.append(sample)

    if not samples:
        raise ValueError("No valid samples could be loaded from the dataset")

    logger.info(f"Successfully loaded {len(samples)} samples")
    return samples


@task
def evaluate_model(config: EvalConfig):
    """Main evaluation task."""
    samples = load_dataset_samples(config)
    scorer = get_scorer(config.scorer_name, config.scorer_args, config.eval_type)

    solver = multiple_choice() if config.eval_type == "classification" else generate()

    return Task(
        dataset=samples,
        solver=[solver],
        scorer=scorer,
    )


def get_scorer(scorer_name: str, scorer_args: Dict[str, Any], eval_type: str):
    """Get appropriate scorer based on name and arguments."""
    scorers = {
        "includes": includes,
        "match": match,
        "pattern": pattern,
        "answer": answer,
        "exact": exact,
        "f1": f1,
        "model_graded_qa": model_graded_qa,
        "model_graded_fact": model_graded_fact,
    }

    if scorer_name not in scorers:
        raise ValueError(
            f"Unknown scorer: {scorer_name}. Available scorers: {list(scorers.keys())}"
        )

    # Special handling for multiple choice with model_graded_qa
    if eval_type == "classification" and scorer_name == "model_graded_qa":
        scorer_args.update(
            {
                "instructions": """
            Please evaluate if the model's answer matches the correct answer for this multiple choice question.
            Grade the response as correct (C) only if the model clearly indicates the same answer choice.
            If the model provides a different answer or is ambiguous, grade it as incorrect (I).
            
            Provide your grade in the format:
            GRADE: C
            or
            GRADE: I
            """,
            }
        )

    return scorers[scorer_name](**scorer_args)


def clean_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ["-", "_", "."] else "_" for c in name)


def get_model_config(config: EvalConfig) -> Dict[str, Any]:
    """Get the appropriate model configuration."""
    if config.vllm_config.use_vllm:
        # join openai to config model
        vllm_model_name = "openai/" + config.model

        return {
            "model": vllm_model_name,
            "model_base_url": f"http://{config.vllm_config.host}:{config.vllm_config.port}/v1",
            "max_connections": config.vllm_config.max_connections,
        }
    return {"model": config.model, "model_base_url": None, "max_connections": None}


def parse_scorer_args(args_str: str) -> Dict[str, Any]:
    """Parse scorer arguments from string format (e.g., 'case_sensitive=true,ignore_whitespace=false')."""
    if not args_str:
        return {}

    args_dict = {}
    pairs = args_str.split(",")

    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Convert string values to appropriate types
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "none":
            value = None
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "").isdigit():
            value = float(value)

        args_dict[key] = value

    return args_dict


def unique_choices(label_column: List[str]) -> List[str]:
    """Get unique choices from a list of labels."""
    return sorted(set(label_column))


def make_log_dir(config: EvalConfig) -> Path:
    """Create a log directory for the evaluation."""
    model_name = clean_name(config.model)
    dataset_name = clean_name(config.dataset)
    split_name = clean_name(config.split)
    config_name = clean_name(config.config) if config.config else "default"
    scorer_name = clean_name(config.scorer_name)

    # Construct log directory path
    log_dir = (
        Path(config.output_dir)
        / f"{model_name}_{dataset_name}_{split_name}_{config_name}_{scorer_name}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir
