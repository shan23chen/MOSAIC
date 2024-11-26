import os
import logging
import sys
import numpy as np
from datetime import datetime


def sanitize_path(path_str):
    """Convert path string to a safe directory name by replacing / with _"""
    return path_str.replace("/", "_").replace("\\", "_")


def get_save_directory(
    base_dir, model_name, dataset_name, split, layer, width, config_name=None
):
    """
    Create and return a structured save directory path

    Args:
        base_dir (str): Base directory for saving
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        split (str): Dataset split
        layer (int): Layer number
        width (str): Width parameter
        config_name (str, optional): Dataset configuration name
    """
    # Sanitize model and dataset names
    safe_model_name = sanitize_path(model_name)
    safe_dataset_name = sanitize_path(dataset_name)
    safe_split = sanitize_path(split)
    safe_width = sanitize_path(width)

    # Create the directory components
    components = [base_dir, safe_model_name, safe_dataset_name]

    # Add config_name to path if provided
    if config_name:
        safe_config_name = sanitize_path(config_name)
        components.append(safe_config_name)

    # Add remaining components
    components.extend([safe_split, f"layer_{layer}", safe_width])

    # Join all components to create the final path
    save_dir = os.path.join(*components)

    return save_dir


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "processing.log")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def get_dashboard_directory(base_dir, model_name, dataset_name, layer, width):
    """Create and return a structured save directory path"""
    # Sanitize model and dataset names
    safe_model_name = sanitize_path(model_name)
    safe_dataset_name = sanitize_path(dataset_name)
    safe_width = sanitize_path(width)

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create path: base_dir/model_name/dataset_name/layer_{layer}/timestamp
    save_dir = os.path.join(
        base_dir,
        safe_model_name,
        safe_dataset_name,
        f"layer_{layer}",
        safe_width,
        timestamp,
    )

    return save_dir


def parse_binarize_value(binarize_value_input):
    if binarize_value_input is None:
        return None
    if isinstance(binarize_value_input, str):
        if binarize_value_input.lower() == "none":
            return None
        else:
            try:
                return float(binarize_value_input)
            except ValueError:
                raise ValueError(
                    f"Invalid binarize_value: {binarize_value_input}. Must be a float or 'None'."
                )
    elif isinstance(binarize_value_input, (int, float)):
        return float(binarize_value_input)
    else:
        raise ValueError(
            f"Invalid binarize_value type: {type(binarize_value_input)}. Must be a float or 'None'."
        )
