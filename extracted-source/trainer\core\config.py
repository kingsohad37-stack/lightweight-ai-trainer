"""
Structured training configuration. This is the contract between the
natural-language planner and the real training engine — the planner may
only ever produce one of these, never a "training complete" claim itself.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List


class ConfigValidationError(Exception):
    pass


VALID_TASKS = {"classification", "regression", "text_classification", "clustering", "language_modeling"}
VALID_ALGORITHMS = {
    "linear_regression", "logistic_regression", "naive_bayes", "knn",
    "decision_tree", "random_forest", "svm", "kmeans", "neural_network",
    "tiny_transformer",
}


@dataclass
class TrainingConfig:
    task: str
    algorithm: str
    dataset_path: str
    input_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    test_size: float = 0.2
    random_seed: int = 42
    # neural-network specific (ignored by classical algorithms)
    hidden_layers: List[int] = field(default_factory=lambda: [32])
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "adam"          # sgd | momentum | adam
    loss: Optional[str] = None        # inferred if not given
    activation: str = "relu"          # relu | sigmoid | tanh | gelu
    early_stopping_patience: Optional[int] = None
    checkpoint_every: int = 1
    experiment_name: str = "experiment"
    n_clusters: Optional[int] = None  # for kmeans

    def validate(self):
        errors = []
        if self.task not in VALID_TASKS:
            errors.append(f"Unknown task '{self.task}'. Valid: {sorted(VALID_TASKS)}")
        if self.algorithm not in VALID_ALGORITHMS:
            errors.append(f"Unknown algorithm '{self.algorithm}'. Valid: {sorted(VALID_ALGORITHMS)}")
        if not (0.0 < self.test_size < 1.0):
            errors.append(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.task != "clustering" and self.task != "language_modeling" and not self.target_column:
            errors.append(f"task '{self.task}' requires a target_column")
        if self.algorithm == "kmeans" and not self.n_clusters:
            errors.append("kmeans requires n_clusters")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if self.optimizer not in {"sgd", "momentum", "adam"}:
            errors.append(f"Unknown optimizer '{self.optimizer}'")
        if self.activation not in {"relu", "sigmoid", "tanh", "gelu"}:
            errors.append(f"Unknown activation '{self.activation}'")

        # incompatibility checks
        if self.algorithm in {"linear_regression"} and self.task != "regression":
            errors.append("linear_regression is only valid for task=regression")
        if self.algorithm in {"logistic_regression", "naive_bayes", "knn", "svm"} and self.task not in {
            "classification", "text_classification"
        }:
            errors.append(f"{self.algorithm} is only valid for classification tasks")
        if self.algorithm == "kmeans" and self.task != "clustering":
            errors.append("kmeans requires task=clustering")

        if errors:
            raise ConfigValidationError("; ".join(errors))
        return True

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(asdict(self), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @staticmethod
    def from_dict(d: dict) -> "TrainingConfig":
        known = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return TrainingConfig(**filtered)

    @staticmethod
    def from_json(path: str) -> "TrainingConfig":
        with open(path) as f:
            return TrainingConfig.from_dict(json.load(f))
