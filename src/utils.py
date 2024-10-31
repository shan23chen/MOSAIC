import os
import logging
import sys


def sanitize_path(path_str):
    """Convert path string to a safe directory name by replacing / with _"""
    return path_str.replace("/", "_").replace("\\", "_")


def get_save_directory(base_dir, model_name, dataset_name, split, layer, width):
    """Create and return a structured save directory path"""
    # Sanitize model and dataset names
    safe_model_name = sanitize_path(model_name)
    safe_dataset_name = sanitize_path(dataset_name)
    safe_split = sanitize_path(split)
    safe_width = sanitize_path(width)

    # Create path: base_dir/model_name/dataset_name/layer_{layer}
    save_dir = os.path.join(
        base_dir,
        safe_model_name,
        safe_dataset_name,
        safe_split,
        f"layer_{layer}",
        safe_width,
    )

    return save_dir


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
