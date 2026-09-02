"""
Real inference: loads an actual saved checkpoint (weights/model) and runs
a genuine forward pass / sklearn predict(). No hardcoded outputs.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from trainer.training.checkpoint import CheckpointManager
from trainer.preprocessing.tabular import TabularPreprocessor
from trainer.preprocessing.text import TextPreprocessor
from trainer.algorithms.neural import DenseNeuralNetwork


class Predictor:
    def __init__(self, models_root: str, experiment_name: str, epoch: int | None = None):
        self.mgr = CheckpointManager(models_root, experiment_name)
        ckpt = self.mgr.load_epoch(epoch) if epoch else self.mgr.load_latest()
        if ckpt is None:
            raise FileNotFoundError(
                f"No trained checkpoint found for '{experiment_name}'. Train a model first."
            )
        self.metadata = ckpt["metadata"]
        self.model_state = ckpt["model_state"]
        pp_state = ckpt["preprocessor_state"]
        if pp_state.get("kind") == "text":
            self.preprocessor = TextPreprocessor.from_state(pp_state)
        else:
            self.preprocessor = TabularPreprocessor.from_state(pp_state)
        self.algorithm = self.metadata["algorithm"]
        self.config = self.metadata["config"]

        if "sklearn_model" in self.model_state:
            self.model = self.model_state["sklearn_model"]
            self.kind = "classical"
        elif "net_state" in self.model_state:
            self.model = DenseNeuralNetwork.from_state(self.model_state["net_state"])
            self.kind = "neural"
        else:
            raise ValueError("Unrecognized model_state format in checkpoint.")

    def predict(self, records: list[dict]) -> list[dict]:
        """records: list of {column_name: value} dicts matching input_columns."""
        df = pd.DataFrame(records)
        X = self.preprocessor.transform(df)

        outputs = []
        if self.kind == "classical":
            preds = self.model.predict(X)
            probs = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(X)
                except Exception:
                    probs = None
            for i, p in enumerate(preds):
                label = self.preprocessor.decode_target([p])[0] if self.preprocessor.is_classification_target_ else float(p)
                entry = {"prediction": label}
                if probs is not None:
                    class_labels = (
                        self.preprocessor.decode_target(self.model.classes_)
                        if self.preprocessor.is_classification_target_ else self.model.classes_
                    )
                    entry["probabilities"] = {
                        str(c): float(pr) for c, pr in zip(class_labels, probs[i])
                    }
                outputs.append(entry)
        else:
            raw = self.model.predict(X)
            if self.preprocessor.is_classification_target_:
                if raw.shape[1] == 1:
                    prob_pos = raw.flatten()
                    labels = (prob_pos > 0.5).astype(int)
                    decoded = self.preprocessor.decode_target(labels)
                    for i in range(len(decoded)):
                        outputs.append({
                            "prediction": decoded[i],
                            "probabilities": {"positive_class_prob": float(prob_pos[i])},
                        })
                else:
                    labels = np.argmax(raw, axis=1)
                    decoded = self.preprocessor.decode_target(labels)
                    for i in range(len(decoded)):
                        outputs.append({
                            "prediction": decoded[i],
                            "probabilities": {
                                str(c): float(p) for c, p in
                                zip(self.preprocessor.target_encoder_.classes_, raw[i])
                            },
                        })
            else:
                y_mean = self.model_state.get("y_mean")
                y_std = self.model_state.get("y_std")
                for v in raw.flatten():
                    val = float(v) * y_std + y_mean if y_mean is not None else float(v)
                    outputs.append({"prediction": val})
        return outputs
