import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datasets import load_dataset
from inspect_ai import Task, task, eval
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
)
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem

from utils_output import parse_scorer_args, get_scorer


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    model: str
    model_base_url: Optional[str]
    max_connections: Optional[int]
    dataset: str
    split: str
    config: Optional[str]
    input_column: str
    context_column: Optional[str]
    label_column: str
    id_column: Optional[str]
    limit: Optional[int]
    system_prompt: str
    eval_type: str
    scorer_name: str
    scorer_args: Dict[str, Any]
    choice_columns: Optional[List[str]] = None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AI model")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., openai/gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--model-base-url",
        type=str,
        help="Base URL for model API (e.g., http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--max-connections", type=int, help="Maximum number of concurrent connections"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset identifier (e.g., hellaswag)",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate on"
    )
    parser.add_argument("--config", type=str, help="Dataset configuration name")
    parser.add_argument(
        "--input-column", type=str, required=True, help="Column name for input text"
    )
    parser.add_argument(
        "--context-column", type=str, help="Column name for context text"
    )
    parser.add_argument(
        "--label-column", type=str, required=True, help="Column name for labels"
    )
    parser.add_argument(
        "--id-column", type=str, default="id", help="Column name for sample IDs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Number of samples to evaluate",
    )
    # Evaluation configuration
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["classification", "open_ended"],
        required=True,
        help="Type of evaluation task",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        choices=[
            "includes",
            "match",
            "pattern",
            "answer",
            "exact",
            "f1",
            "model_graded_qa",
            "model_graded_fact",
        ],
        default="exact",
        help="Scoring method to use",
    )
    parser.add_argument(
        "--scorer-args",
        type=str,
        default="",
        help="Additional scorer arguments (key=value,key2=value2)",
    )
    parser.add_argument(
        "--choice-columns",
        type=str,
        help="Comma-separated column names for multiple choice options",
    )

    args = parser.parse_args()

    # Process arguments
    choice_columns = args.choice_columns.split(",") if args.choice_columns else None
    scorer_args = parse_scorer_args(args.scorer_args)

    return EvalConfig(
        model=args.model,
        model_base_url=args.model_base_url,
        max_connections=args.max_connections,
        dataset=args.dataset,
        split=args.split,
        config=args.config,
        input_column=args.input_column,
        context_column=args.context_column,
        label_column=args.label_column,
        id_column=args.id_column,
        limit=args.limit,
        system_prompt=args.system_prompt,
        eval_type=args.eval_type,
        scorer_name=args.scorer,
        scorer_args=scorer_args,
        choice_columns=choice_columns,
    )


# Template for multiple choice questions
MC_TEMPLATE = """
Answer the following multiple choice question. Your answer should be just the letter of the correct choice (A, B, C, or D).

Question: {question}

Choices:
{choices}
""".strip()


def process_dataset_samples(config: EvalConfig) -> List[Sample]:
    """Load and process dataset samples."""
    dataset = load_dataset(config.dataset, config.config, split=config.split)

    # Check if "id" column is present
    if config.id_column not in dataset.column_names:
        # Add "id" column with values as row number
        dataset = dataset.map(lambda example, idx: {"id": idx}, with_indices=True)

    samples = []
    for item in dataset:
        if config.eval_type == "classification":
            choices = (
                [str(item[col]) for col in config.choice_columns]
                if config.choice_columns
                else None
            )
            if not choices:
                raise ValueError(
                    "Choice columns must be specified for classification tasks"
                )

            # add context to question if context column is provided
            if config.context_column:
                question = MC_TEMPLATE.format(
                    question=str(item[config.context_column])
                    + "\n"
                    + str(item[config.input_column]),
                    choices="\n".join(
                        f"{letter}. {text}" for letter, text in zip("ABCD", choices)
                    ),
                )
            else:
                question = MC_TEMPLATE.format(
                    question=str(item[config.input_column]),
                    choices="\n".join(
                        f"{letter}. {text}" for letter, text in zip("ABCD", choices)
                    ),
                )

            # Check if the model is Gemma
            if "gemma" in config.model.lower():
                # For Gemma, include system prompt in the user message
                # if config.context_column and not none
                if config.context_column and config.context_column != "none":
                    user_content = f"{config.system_prompt}\n\n{str(item[config.context_column])}\n\n{question}"
                else:
                    user_content = f"{config.system_prompt}\n\n{question}"
                sample = Sample(
                    input=[ChatMessageUser(content=user_content)],
                    target=str(item[config.label_column]),
                    id=str(item.get(config.id_column, "")),
                    choices=choices,
                )
            else:
                # For other models, use separate system and user messages
                if config.context_column:
                    sample = Sample(
                        input=[
                            ChatMessageSystem(content=config.system_prompt),
                            ChatMessageUser(content=str(item[config.context_column])),
                            ChatMessageUser(content=question),
                        ],
                        target=str(item[config.label_column]),
                        id=str(item.get(config.id_column, "")),
                        choices=choices,
                    )
                else:
                    sample = Sample(
                        input=[
                            ChatMessageSystem(content=config.system_prompt),
                            ChatMessageUser(content=question),
                        ],
                        target=str(item[config.label_column]),
                        id=str(item.get(config.id_column, "")),
                        choices=choices,
                    )
        else:
            # Similar logic for open-ended questions
            if "gemma" in config.model.lower():
                if config.context_column and config.context_column != "none":
                    user_content = f"{config.system_prompt}\n\n{str(item[config.context_column])}\n\n{str(item[config.input_column])}"
                else:
                    user_content = (
                        f"{config.system_prompt}\n\n{str(item[config.input_column])}"
                    )
                # Check if the label_column is "sorry"
                if config.label_column.lower() == "sorry":
                    target_label = "Model refuses request"
                else:
                    target_label = str(item[config.label_column])

                sample = Sample(
                    input=[ChatMessageUser(content=user_content)],
                    target=target_label,
                    id=str(item.get(config.id_column, "")),
                )
            else:
                if config.context_column:
                    sample = Sample(
                        input=[
                            ChatMessageSystem(content=config.system_prompt),
                            ChatMessageUser(content=str(item[config.context_column])),
                            ChatMessageUser(content=str(item[config.input_column])),
                        ],
                        target=str(item[config.label_column]),
                        id=str(item.get(config.id_column, "")),
                    )
                else:
                    sample = Sample(
                        input=[
                            ChatMessageSystem(content=config.system_prompt),
                            ChatMessageUser(content=str(item[config.input_column])),
                        ],
                        target=str(item[config.label_column]),
                        id=str(item.get(config.id_column, "")),
                    )

        samples.append(sample)

    return samples


@task
def evaluate_model(
    config: EvalConfig,
    input_column: str,
    label_column: str,
    dataset_name: str,
    split: str,
    limit: Optional[int] = None,
):
    """Create evaluation task."""
    samples = process_dataset_samples(config)
    scorer = get_scorer(config.scorer_name, config.scorer_args, config.eval_type)
    solver = multiple_choice() if config.eval_type == "classification" else generate()

    return Task(
        dataset=samples,
        solver=[solver],
        scorer=scorer,
    )


def main():
    """Main execution function."""
    config = parse_arguments()

    # Get model configuration
    model_config = {
        "model": config.model,
        "model_base_url": config.model_base_url,
        "max_connections": config.max_connections,
        "device": "0,1",
        "gpu_memory_utilization": 0.95,
    }

    # Run evaluation
    eval(
        evaluate_model(
            config,
            config.input_column,
            config.label_column,
            config.dataset,
            config.split,
        ),
        **{k: v for k, v in model_config.items() if v is not None},
        limit=config.limit,
    )


if __name__ == "__main__":
    main()
