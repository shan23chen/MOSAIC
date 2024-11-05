import os
import argparse
import subprocess
import time
from pathlib import Path

from utils_output import (
    make_log_dir,
    parse_scorer_args,
    start_vllm_service,
    get_model_config,
    VLLMConfig,
    EvalConfig,
    evaluate_model,
    cleanup_vllm,
    force_kill_process,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AI model")

    # Basic arguments
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-column", type=str, required=True)
    parser.add_argument("--label-column", type=str, required=True)
    parser.add_argument("--id-column", type=str)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument(
        "--system-prompt", type=str, default="You are a helpful AI assistant."
    )
    parser.add_argument("--grader-model", type=str, default="openai/gpt-3.5-turbo")
    parser.add_argument("--debug", action="store_true")

    # Evaluation type and scoring
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["classification", "open_ended"],
        required=True,
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
        help="Scorer to use for evaluation",
    )
    parser.add_argument(
        "--scorer-args",
        type=str,
        default="",
        help="Comma-separated key=value pairs for scorer configuration",
    )

    # Multiple choice specific arguments
    parser.add_argument(
        "--choice-columns",
        type=str,
        help="Comma-separated list of choice columns for multiple choice questions",
    )

    # VLLM-specific arguments
    vllm_group = parser.add_argument_group("VLLM Configuration")
    vllm_group.add_argument(
        "--use-vllm", action="store_true", help="Use VLLM for local model serving"
    )
    vllm_group.add_argument(
        "--vllm-model-path", type=str, help="Path to the model for VLLM"
    )
    vllm_group.add_argument(
        "--vllm-host", type=str, default="0.0.0.0", help="Host for VLLM service"
    )
    vllm_group.add_argument(
        "--vllm-port", type=int, default=8081, help="Port for VLLM service"
    )
    vllm_group.add_argument(
        "--vllm-dtype", type=str, default="auto", help="Data type for VLLM"
    )
    vllm_group.add_argument(
        "--vllm-quantization", type=str, help="Quantization method for VLLM"
    )
    vllm_group.add_argument("--vllm-load-format", type=str, help="Load format for VLLM")
    vllm_group.add_argument(
        "--max-connections", type=int, default=32, help="Maximum number of connections"
    )

    args = parser.parse_args()

    args = parser.parse_args()

    # Process existing arguments
    choice_columns = args.choice_columns.split(",") if args.choice_columns else None
    target_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    scorer_args = parse_scorer_args(args.scorer_args)

    # Create VLLM config
    vllm_config = VLLMConfig(
        use_vllm=args.use_vllm,
        model_path=args.vllm_model_path if args.use_vllm else None,
        host=args.vllm_host,
        port=args.vllm_port,
        dtype=args.vllm_dtype,
        quantization=args.vllm_quantization,
        load_format=args.vllm_load_format,
        max_connections=args.max_connections,
    )

    return EvalConfig(
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        config=args.config,
        input_column=args.input_column,
        label_column=args.label_column,
        id_column=args.id_column,
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        system_prompt=args.system_prompt,
        grader_model=args.grader_model,
        debug=args.debug,
        eval_type=args.eval_type,
        scorer_name=args.scorer,
        scorer_args=scorer_args,
        vllm_config=vllm_config,
        choice_columns=choice_columns,
        target_mapping=target_mapping,
    )


def main():
    """Main execution function with improved cleanup."""
    config = parse_arguments()
    vllm_process = None
    actual_port = config.vllm_config.port

    try:
        if config.vllm_config.use_vllm:
            # Start VLLM with potential new port
            vllm_process, actual_port = start_vllm_service(config.vllm_config)
            # Update config with actual port being used
            config.vllm_config.port = actual_port
            print("VLLM service started successfully")

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Get model configuration
        model_config = get_model_config(config)

        # Run evaluation
        from inspect_ai import eval

        # Clean names for log directory
        log_dir = make_log_dir(config)

        # Run evaluation with appropriate configuration
        eval(
            evaluate_model(config),
            model=model_config["model"],
            model_base_url=model_config["model_base_url"],
            max_connections=model_config["max_connections"],
            log_dir=str(log_dir),
        )

        print(f"Results saved to: {log_dir}")

    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Cleaning up...")
        if vllm_process:
            force_kill_process(vllm_process)
            cleanup_vllm(actual_port, config.vllm_config.host)
        raise

    except Exception as e:
        print(f"Error in main execution: {e}")
        if vllm_process:
            force_kill_process(vllm_process)
            cleanup_vllm(actual_port, config.vllm_config.host)
        raise

    finally:
        # Clean up when done
        if vllm_process:
            try:
                # Try graceful shutdown first
                vllm_process.terminate()
                try:
                    vllm_process.wait(timeout=10)  # Wait up to 10 seconds
                except subprocess.TimeoutExpired:
                    # If graceful shutdown fails, force kill
                    force_kill_process(vllm_process)

                # Verify and cleanup port
                cleanup_vllm(actual_port, config.vllm_config.host)

            except Exception as e:
                print(f"Error during final cleanup: {e}")
                # Last resort: force kill
                force_kill_process(vllm_process)


if __name__ == "__main__":
    main()
