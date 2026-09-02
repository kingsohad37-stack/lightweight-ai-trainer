"""
Natural-language -> structured TrainingConfig planner.

CRITICAL BOUNDARY: this module only ever produces a TrainingConfig. It never
trains anything and never claims training happened. That is the exclusive
job of trainer.training.trainer. This keeps the "AI planner" and the "real
training engine" strictly separated, as required.

The heuristics below are rule-based (keyword/statistics-driven), not an LLM
call — this keeps the planner fully local and inspectable. A real LLM-backed
planner could be swapped in later behind the same `plan()` interface.
"""
from __future__ import annotations
import re
from dataclasses import asdict

from trainer.core.config import TrainingConfig
from trainer.datasets import loader, analyzer

REGRESSION_HINTS = [
    "price", "predict the value", "how much", "regression", "estimate", "forecast",
    "score", "amount", "revenue", "salary", "temperature",
]
CLASSIFICATION_HINTS = [
    "classify", "classifier", "category", "spam", "sentiment", "positive or negative",
    "pass or fail", "predict whether", "detect whether", "yes or no", "will",
]
CLUSTERING_HINTS = ["cluster", "group similar", "segment", "unsupervised"]
TEXT_HINTS = ["chatbot", "answer questions", "language model", "text generation", "notes"]


def _detect_task_from_text(prompt: str) -> str:
    p = prompt.lower()
    if any(h in p for h in TEXT_HINTS):
        return "language_modeling"
    if any(h in p for h in CLUSTERING_HINTS):
        return "clustering"
    if any(h in p for h in CLASSIFICATION_HINTS):
        return "classification"
    if any(h in p for h in REGRESSION_HINTS):
        return "regression"
    return "unknown"


def _guess_target_column(prompt: str, columns: list[str]) -> str | None:
    p = prompt.lower()

    # Direct mentions of a column name (exact, underscore-insensitive, or as
    # a whole word) — pick whichever mentioned column appears EARLIEST in the
    # prompt, since phrasing like "predict the price of a house from its
    # size, bedrooms and age" names the target before the inputs.
    best_col, best_pos = None, None
    for col in columns:
        col_l = col.lower()
        candidates = {col_l, col_l.replace("_", " ")}
        for cand in candidates:
            pos = p.find(cand)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_col, best_pos = col, pos
    if best_col:
        return best_col

    # common target-ish names
    for candidate in ["label", "target", "class", "outcome", "result", "y"]:
        for col in columns:
            if col.lower() == candidate:
                return col
    return None


def _choose_algorithm(task: str, ds_stats: dict) -> str:
    n_samples = ds_stats.get("n_samples", 0)
    if task == "regression":
        return "linear_regression" if n_samples < 5000 else "random_forest"
    if task in {"classification", "text_classification"}:
        # small/simple → logistic regression; larger/nonlinear-looking → random forest
        return "logistic_regression" if n_samples < 2000 else "random_forest"
    if task == "clustering":
        return "kmeans"
    if task == "language_modeling":
        return "tiny_transformer"
    raise ValueError(f"Cannot choose algorithm for unknown task: {task}")


def plan(prompt: str, dataset_path: str, target_column: str | None = None,
         algorithm: str | None = None, **overrides) -> dict:
    """
    Returns a dict with:
      - config: TrainingConfig (validated)
      - reasoning: human-readable explanation of what was inferred and why
      - dataset_analysis: real stats used to make the decision
    Raises on ambiguous/invalid requests rather than guessing silently.
    """
    reasoning = []

    fmt = loader.detect_format(dataset_path)
    is_text_dataset = fmt in {"txt", "text_folder"}

    if is_text_dataset:
        docs = loader.load(dataset_path)
        ds_stats = analyzer.analyze_text(docs, path=dataset_path if fmt == "txt" else None)
        task = "language_modeling"
        reasoning.append(f"Dataset is plain text ({fmt}) -> treating as language modeling data.")
        input_columns, resolved_target = [], None
    else:
        df = loader.load(dataset_path)
        ds_stats = analyzer.analyze_tabular(df, path=dataset_path)
        columns = list(df.columns)

        task = _detect_task_from_text(prompt)
        if task == "unknown":
            resolved_target_probe = target_column or _guess_target_column(prompt, columns)
            if resolved_target_probe and resolved_target_probe in ds_stats["columns"]:
                col_kind = ds_stats["columns"][resolved_target_probe]["kind"]
                task = "classification" if col_kind == "categorical" else "regression"
                reasoning.append(
                    f"No task keyword found in prompt; inferred task='{task}' from the "
                    f"target column '{resolved_target_probe}' being {col_kind}."
                )
            else:
                raise ValueError(
                    "Could not determine the task from the prompt or dataset. "
                    "Please mention e.g. 'classify', 'predict the price of', or specify target_column explicitly. "
                    f"Available columns: {columns}"
                )
        else:
            reasoning.append(f"Detected task='{task}' from prompt keywords.")

        resolved_target = target_column or _guess_target_column(prompt, columns)
        if task != "clustering" and not resolved_target:
            raise ValueError(
                f"Could not identify the target column for task='{task}'. "
                f"Available columns: {columns}. Please specify target_column explicitly."
            )
        input_columns = [c for c in columns if c != resolved_target]
        if resolved_target:
            reasoning.append(f"Target column: '{resolved_target}'. Input columns: {input_columns}.")

        # Free-text input column -> this is really a text-classification
        # task (needs TF-IDF features, not label-encoded row IDs).
        if task == "classification" and len(input_columns) == 1:
            only_col_kind = ds_stats["columns"].get(input_columns[0], {}).get("kind")
            if only_col_kind == "text":
                task = "text_classification"
                reasoning.append(
                    f"Input column '{input_columns[0]}' is free text -> "
                    f"switched task to 'text_classification' (TF-IDF features)."
                )

    chosen_algorithm = algorithm or _choose_algorithm(task, ds_stats)
    reasoning.append(f"Selected algorithm='{chosen_algorithm}' based on task and dataset size "
                      f"({ds_stats.get('n_samples', ds_stats.get('n_documents'))} samples).")

    cfg_kwargs = dict(
        task=task,
        algorithm=chosen_algorithm,
        dataset_path=dataset_path,
        input_columns=input_columns,
        target_column=resolved_target,
    )
    cfg_kwargs.update(overrides)
    if task == "clustering" and "n_clusters" not in overrides:
        cfg_kwargs["n_clusters"] = overrides.get("n_clusters", 3)
        reasoning.append("No cluster count specified; defaulting n_clusters=3 (override if needed).")

    config = TrainingConfig.from_dict(cfg_kwargs)
    config.validate()

    return {
        "config": config,
        "reasoning": reasoning,
        "dataset_analysis": ds_stats,
    }
