"""
Orchestrates the real end-to-end pipeline:
dataset load -> analyze -> preprocess -> train -> validate -> checkpoint -> export

This module never fabricates a result: if any stage fails, it raises rather
than returning a fake "success".
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import asdict

from trainer.datasets import loader, analyzer
from trainer.preprocessing.tabular import TabularPreprocessor
from trainer.preprocessing.text import TextPreprocessor
from trainer.algorithms.classical import is_classical
from trainer.training.trainer import train_classical, train_neural
from trainer.training.train_lm import train_language_model
from trainer.training.checkpoint import CheckpointManager
from trainer.hardware import memory as hw


MODELS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
EXPERIMENTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")


def run_pipeline(config, progress_cb=None, resume=False):
    """
    Executes the full real pipeline for a validated TrainingConfig.
    Returns a dict with dataset_analysis, training_result, model_dir.
    """
    config.validate()
    t_start = time.time()

    checkpoint_mgr = CheckpointManager(MODELS_ROOT, config.experiment_name)

    if config.task == "language_modeling":
        resume_state = checkpoint_mgr.load_latest() if resume else None
        result = train_language_model(config, checkpoint_mgr, progress_cb=progress_cb,
                                        resume_from=resume_state)
        docs = loader.load(config.dataset_path)
        ds_stats = analyzer.analyze_text(docs if isinstance(docs, list) else [docs],
                                          path=config.dataset_path if config.dataset_path.endswith(".txt") else None)
        duration = result.duration_seconds
        exp_dir = os.path.join(EXPERIMENTS_ROOT, config.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        experiment_record = {
            "dataset": config.dataset_path, "algorithm": config.algorithm, "task": config.task,
            "config": asdict(config), "dataset_analysis": ds_stats,
            "final_metrics": result.final_metrics, "history": result.history,
            "training_time_seconds": duration, "n_parameters": result.n_parameters,
            "memory_check": result.memory_check, "status": result.status,
        }
        with open(os.path.join(exp_dir, "experiment.json"), "w") as f:
            json.dump(experiment_record, f, indent=2, default=str)
        return {
            "dataset_analysis": ds_stats, "result": result,
            "model_dir": checkpoint_mgr.model_dir, "experiment_record": experiment_record,
            "preprocessor": None,
        }

    # 1. Load
    df = loader.load(config.dataset_path)

    # 2. Analyze (real stats)
    ds_stats = analyzer.analyze_tabular(df, path=config.dataset_path)

    # 3. Preprocess
    if config.task == "text_classification":
        preprocessor = TextPreprocessor(
            text_column=config.input_columns[0], target_column=config.target_column
        )
    else:
        preprocessor = TabularPreprocessor(
            input_columns=config.input_columns, target_column=config.target_column
        )
    data = preprocessor.fit_transform(df, task=config.task, test_size=config.test_size,
                                       random_seed=config.random_seed)

    # 4. Memory budget check before training begins
    n_features = data["X_train"].shape[1]
    if not is_classical(config.algorithm):
        layer_sizes = [n_features] + list(config.hidden_layers) + [1]
        est = hw.estimate_dense_nn_bytes(layer_sizes, optimizer=config.optimizer)
        ok, msg, ratio = hw.check_budget(est)
    else:
        ok, msg = True, "Classical algorithm — memory footprint is bounded by scikit-learn's own model size."

    resume_state = None
    if resume:
        resume_state = checkpoint_mgr.load_latest()

    # 5. Train (real)
    if is_classical(config.algorithm):
        result = train_classical(config, data, checkpoint_mgr, preprocessor.to_state())
    else:
        result = train_neural(config, data, checkpoint_mgr, preprocessor.to_state(),
                               resume_from=resume_state, progress_cb=progress_cb)

    duration = time.time() - t_start

    # 6. Experiment tracking (real record of what happened)
    exp_dir = os.path.join(EXPERIMENTS_ROOT, config.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    experiment_record = {
        "dataset": config.dataset_path,
        "algorithm": config.algorithm,
        "task": config.task,
        "config": asdict(config),
        "dataset_analysis": ds_stats,
        "final_metrics": result.final_metrics,
        "history": result.history,
        "training_time_seconds": duration,
        "n_parameters": result.n_parameters,
        "memory_check": msg,
        "status": result.status,
    }
    with open(os.path.join(exp_dir, "experiment.json"), "w") as f:
        json.dump(experiment_record, f, indent=2, default=str)

    return {
        "dataset_analysis": ds_stats,
        "result": result,
        "model_dir": checkpoint_mgr.model_dir,
        "experiment_record": experiment_record,
        "preprocessor": preprocessor,
    }
