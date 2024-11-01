import yaml
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple


class FeatureLookup:
    def __init__(self, config_path: Path, cache_dir: Path):
        """Initialize with paths to YAML config and cache directory."""
        self.config_path = Path(config_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load and parse YAML configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _get_cache_path(
        self, model: str, layer: str, width: str, sae_location: str
    ) -> Path:
        """Get path for cached explanations file."""
        return self.cache_dir / f"{model}_layer{layer}_{width}_{sae_location}.json"

    def get_neuronpedia_id(self, sae_name: str, feature_id: str) -> Optional[str]:
        """Get Neuronpedia ID from SAE name and feature ID."""
        if sae_name not in self.config:
            logging.error(f"SAE {sae_name} not found in config")
            return None

        sae_config = self.config[sae_name]
        for sae in sae_config.get("saes", []):
            if sae["id"] == feature_id:
                return sae["neuronpedia"]

        logging.error(f"Feature ID {feature_id} not found in {sae_name}")
        return None

    def fetch_and_save_explanations(
        self,
        model: str,
        layer: str,
        width: str,
        sae_location: str,
        neuronpedia_id: str,
        api_key: str,
    ) -> bool:
        """Fetch explanations from Neuronpedia and save to cache."""
        url = "https://www.neuronpedia.org/api/explanation/export"
        sae_id = neuronpedia_id.split("/")[-1]

        try:
            response = requests.get(
                url,
                params={"modelId": model, "saeId": sae_id},
                headers={"X-Api-Key": api_key},
            )
            response.raise_for_status()

            explanations = {
                str(item["index"]): item["description"]
                for item in response.json()
                if "index" in item
            }

            cache_path = self._get_cache_path(model, layer, width, sae_location)
            with open(cache_path, "w") as f:
                json.dump(explanations, f)

            logging.info(
                f"Saved {len(explanations)} explanations for {model} layer {layer}"
            )
            return True
        except Exception as e:
            logging.error(f"Error retrieving explanation: {e}")
            return None
