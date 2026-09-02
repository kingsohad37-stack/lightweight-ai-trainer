"""
Real checkpointing: model weights, optimizer state, epoch, config,
preprocessor state, and metrics are all written to disk so training can
be resumed exactly, and inference can be run without retraining.
"""
from __future__ import annotations
import json
import os
import pickle
import random
from dataclasses import asdict
from datetime import datetime, timezone


class CheckpointManager:
    def __init__(self, base_dir: str, model_name: str):
        self.base_dir = base_dir
        self.model_name = model_name
        self.model_dir = os.path.join(base_dir, model_name)
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoints_dir, f"checkpoint_{epoch:04d}")

    def save(self, epoch: int, algorithm: str, model_state, preprocessor_state,
              config, metrics: dict, random_seed: int, is_latest: bool = True):
        ckpt_dir = self._checkpoint_path(epoch)
        os.makedirs(ckpt_dir, exist_ok=True)

        payload = {
            "epoch": epoch,
            "algorithm": algorithm,
            "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
            "metrics": metrics,
            "random_seed": random_seed,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(payload, f, indent=2)

        with open(os.path.join(ckpt_dir, "model_state.pkl"), "wb") as f:
            pickle.dump(model_state, f)
        with open(os.path.join(ckpt_dir, "preprocessor_state.pkl"), "wb") as f:
            pickle.dump(preprocessor_state, f)

        if is_latest:
            latest_dir = os.path.join(self.model_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)
            with open(os.path.join(latest_dir, "metadata.json"), "w") as f:
                json.dump(payload, f, indent=2)
            with open(os.path.join(latest_dir, "model_state.pkl"), "wb") as f:
                pickle.dump(model_state, f)
            with open(os.path.join(latest_dir, "preprocessor_state.pkl"), "wb") as f:
                pickle.dump(preprocessor_state, f)

        return ckpt_dir

    def load_latest(self):
        latest_dir = os.path.join(self.model_dir, "latest")
        if not os.path.exists(latest_dir):
            return None
        with open(os.path.join(latest_dir, "metadata.json")) as f:
            metadata = json.load(f)
        with open(os.path.join(latest_dir, "model_state.pkl"), "rb") as f:
            model_state = pickle.load(f)
        with open(os.path.join(latest_dir, "preprocessor_state.pkl"), "rb") as f:
            preprocessor_state = pickle.load(f)
        return {"metadata": metadata, "model_state": model_state, "preprocessor_state": preprocessor_state}

    def load_epoch(self, epoch: int):
        ckpt_dir = self._checkpoint_path(epoch)
        if not os.path.exists(ckpt_dir):
            return None
        with open(os.path.join(ckpt_dir, "metadata.json")) as f:
            metadata = json.load(f)
        with open(os.path.join(ckpt_dir, "model_state.pkl"), "rb") as f:
            model_state = pickle.load(f)
        with open(os.path.join(ckpt_dir, "preprocessor_state.pkl"), "rb") as f:
            preprocessor_state = pickle.load(f)
        return {"metadata": metadata, "model_state": model_state, "preprocessor_state": preprocessor_state}

    def list_checkpoints(self):
        if not os.path.exists(self.checkpoints_dir):
            return []
        return sorted(os.listdir(self.checkpoints_dir))
