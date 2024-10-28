from pathlib import Path
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    # This will point to the directory containing 'src' and 'app'
    return Path(__file__).parent.parent.parent


def get_dashboard_dir(dashboard_dir: str = None) -> Path:
    """Get the dashboard directory path."""
    if dashboard_dir:
        return Path(dashboard_dir)

    # Default path relative to project root
    return get_project_root() / "src" / "processed_features_llm" / "dashboards"
