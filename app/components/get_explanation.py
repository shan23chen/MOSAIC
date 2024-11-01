import json
import logging
from pathlib import Path
from typing import Optional


def _get_cache_path(
    self, model: str, layer: str, width: str, sae_location: str
) -> Path:
    """Get path for cached explanations file."""
    return self.cache_dir / f"{model}_layer{layer}_{width}_{sae_location}.json"


def get_feature_explanation(
    self, model: str, layer: str, width: str, sae_location: str, feature_id: int
) -> Optional[str]:
    """Get explanation for a specific feature ID from cache."""
    cache_path = self._get_cache_path(model, layer, width, sae_location)

    try:
        if cache_path.exists():
            with open(cache_path, "r") as f:
                explanations = json.load(f)
                return explanations.get(str(feature_id))
        return None
    except Exception as e:
        logging.error(f"Error retrieving explanation: {e}")
        return None
