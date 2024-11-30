import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from vllm import LLM, SamplingParams
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

    # Handle 'model' argument for model_graded_qa
    if (
        scorer_name == "model_graded_qa"
        or "model_graded_fact"
        and "model" in scorer_args
    ):
        scorer_args["model"] = [scorer_args["model"]]

    return scorers[scorer_name](**scorer_args)


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
