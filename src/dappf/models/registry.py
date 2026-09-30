"""Model Registry: Serialization, Deserialization, and Artifact Tracking."""

from pathlib import Path
from typing import Any, Dict, Optional
import pickle
import json
from loguru import logger


class ModelRegistry:
    """Manages saving and loading of trained model artifacts."""

    def __init__(self, artifacts_dir: str | Path = "artifacts/models"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        zone: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model and its associated metadata."""
        save_path = self.artifacts_dir / f"{zone}_{model_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        if metadata:
            meta_path = self.artifacts_dir / f"{zone}_{model_name}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved model artifact: {save_path}")
        return save_path

    def load_model(self, zone: str, model_name: str) -> Any:
        """Load a saved model artifact."""
        load_path = self.artifacts_dir / f"{zone}_{model_name}.pkl"
        if not load_path.exists():
            raise FileNotFoundError(f"No artifact found at {load_path}")
        with open(load_path, "rb") as f:
            return pickle.load(f)
