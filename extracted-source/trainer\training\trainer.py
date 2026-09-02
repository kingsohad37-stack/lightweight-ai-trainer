"""
The real training engine. Executes an already-validated TrainingConfig.
Never reports success unless training actually ran and produced a model
that can predict on held-out data.
"""
from __future__ import annotations
import time
import numpy as np

from trainer.hardware import memory as hw
from trainer.algorithms.classical import build_model, is_classical
from trainer.algorithms.neural import DenseNeuralNetwork
from trainer.evaluation.metrics import classification_metrics, regression_metrics
from trainer.training.checkpoint import CheckpointManager


class TrainingResult:
    def __init__(self):
        self.history = []          # per-epoch metrics (neural) or single-shot (classical)
        self.final_metrics = {}
        self.model = None
        self.checkpoint_dir = None
        self.status = "not_started"
        self.error = None
        self.n_parameters = None
        self.duration_seconds = None


def _to_onehot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y.astype(int)] = 1
    return out


def train_classical(cfg, data, checkpoint_mgr: CheckpointManager, preprocessor_state) -> TrainingResult:
    result = TrainingResult()
    t0 = time.time()
    try:
        model = build_model(cfg.algorithm, cfg)

        if cfg.task == "clustering":
            model.fit(data["X_train"])
            labels = model.predict(data["X_test"]) if hasattr(model, "predict") else model.labels_
            metrics = {"inertia": float(getattr(model, "inertia_", float("nan")))}
        else:
            model.fit(data["X_train"], data["y_train"])
            y_pred = model.predict(data["X_test"])
            if cfg.task == "regression":
                metrics = regression_metrics(data["y_test"], y_pred)
            else:
                metrics = classification_metrics(data["y_test"], y_pred)

        result.history = [{"epoch": 1, **metrics}]
        result.final_metrics = metrics
        result.model = model
        result.status = "completed"

        ckpt_dir = checkpoint_mgr.save(
            epoch=1, algorithm=cfg.algorithm, model_state={"sklearn_model": model},
            preprocessor_state=preprocessor_state, config=cfg, metrics=metrics,
            random_seed=cfg.random_seed,
        )
        result.checkpoint_dir = ckpt_dir
        result.n_parameters = None
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        raise
    finally:
        result.duration_seconds = time.time() - t0
    return result


def train_neural(cfg, data, checkpoint_mgr: CheckpointManager, preprocessor_state,
                  resume_from: dict | None = None, progress_cb=None) -> TrainingResult:
    result = TrainingResult()
    t0 = time.time()

    X_train, y_train = data["X_train"], data.get("y_train")
    X_test, y_test = data["X_test"], data.get("y_test")
    n_features = X_train.shape[1]

    is_classification = data.get("n_classes") is not None
    n_classes = data.get("n_classes")
    result_y_mean, result_y_std = None, None

    if is_classification and n_classes == 2:
        output_size, output_activation, loss = 1, "sigmoid", "binary_cross_entropy"
        y_train_t = y_train.reshape(-1, 1).astype(float)
        y_test_t = y_test.reshape(-1, 1).astype(float)
    elif is_classification and n_classes and n_classes > 2:
        output_size, output_activation, loss = n_classes, "softmax", "cross_entropy"
        y_train_t = _to_onehot(y_train, n_classes)
        y_test_t = _to_onehot(y_test, n_classes)
    else:
        # Regression: standardize the target too. Unscaled targets (e.g.
        # house prices in the hundreds of thousands) blow up MSE loss and
        # make gradient-based training unstable/useless at normal learning
        # rates. We train in scaled space and invert for reporting/inference.
        output_size, output_activation, loss = 1, "linear", "mse"
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train)) or 1.0
        y_train_t = ((y_train - y_mean) / y_std).reshape(-1, 1).astype(float)
        y_test_t = ((y_test - y_mean) / y_std).reshape(-1, 1).astype(float)
        result_y_mean, result_y_std = y_mean, y_std

    layer_sizes = [n_features] + list(cfg.hidden_layers) + [output_size]

    est_bytes = hw.estimate_dense_nn_bytes(layer_sizes, optimizer=cfg.optimizer)
    ok, msg, ratio = hw.check_budget(est_bytes)
    result.memory_check = {"ok": ok, "message": msg, "estimated_bytes": est_bytes}
    if not ok:
        # auto-shrink hidden layers rather than crash
        shrunk = [max(4, int(h * ratio)) for h in cfg.hidden_layers]
        layer_sizes = [n_features] + shrunk + [output_size]
        result.memory_check["adjusted_hidden_layers"] = shrunk

    if resume_from is not None:
        net = DenseNeuralNetwork.from_state(resume_from["model_state"]["net_state"])
        start_epoch = resume_from["metadata"]["epoch"] + 1
        result.history = resume_from["metadata"].get("history", [])
        if resume_from["model_state"].get("y_mean") is not None:
            result_y_mean = resume_from["model_state"]["y_mean"]
            result_y_std = resume_from["model_state"]["y_std"]
            y_train_t = ((y_train - result_y_mean) / result_y_std).reshape(-1, 1).astype(float)
            y_test_t = ((y_test - result_y_mean) / result_y_std).reshape(-1, 1).astype(float)
    else:
        net = DenseNeuralNetwork(
            layer_sizes=layer_sizes, activation=cfg.activation, output_activation=output_activation,
            loss=loss, optimizer=cfg.optimizer, learning_rate=cfg.learning_rate, random_seed=cfg.random_seed,
        )
        start_epoch = 1
        result.history = []

    n_samples = X_train.shape[0]
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        rng = np.random.default_rng(cfg.random_seed + epoch)
        perm = rng.permutation(n_samples)
        X_shuf, y_shuf = X_train[perm], y_train_t[perm]

        batch_losses = []
        for start in range(0, n_samples, cfg.batch_size):
            end = start + cfg.batch_size
            xb, yb = X_shuf[start:end], y_shuf[start:end]
            loss_val = net.train_step(xb, yb)
            batch_losses.append(loss_val)

        train_loss = float(np.mean(batch_losses))
        val_pred = net.predict(X_test)
        val_loss = {"mse": lambda p, t: float(np.mean((p - t) ** 2)),
                    "binary_cross_entropy": lambda p, t: float(-np.mean(
                        t * np.log(np.clip(p, 1e-12, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-12, 1)))),
                    "cross_entropy": lambda p, t: float(-np.mean(np.sum(t * np.log(np.clip(p, 1e-12, 1)), axis=1))),
                    }[loss](val_pred, y_test_t)

        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                          "learning_rate": cfg.learning_rate, "samples_processed": n_samples * epoch}

        if is_classification:
            if output_activation == "sigmoid":
                y_pred_labels = (val_pred.flatten() > 0.5).astype(int)
            else:
                y_pred_labels = np.argmax(val_pred, axis=1)
            cm = classification_metrics(y_test, y_pred_labels)
            epoch_metrics.update(cm)
        else:
            # invert scaling so reported MSE/MAE/R2 are in the original units
            unscaled_pred = val_pred.flatten() * result_y_std + result_y_mean
            rm = regression_metrics(y_test, unscaled_pred)
            epoch_metrics.update(rm)

        result.history.append(epoch_metrics)
        if progress_cb:
            progress_cb(epoch_metrics)

        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            checkpoint_mgr.save(
                epoch=epoch, algorithm=cfg.algorithm,
                model_state={"net_state": net.get_state(),
                             "y_mean": result_y_mean, "y_std": result_y_std},
                preprocessor_state=preprocessor_state, config=cfg,
                metrics=epoch_metrics, random_seed=cfg.random_seed,
            )

        if cfg.early_stopping_patience:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    break

    result.final_metrics = result.history[-1] if result.history else {}
    result.model = net
    result.status = "completed"
    result.n_parameters = net.n_parameters()
    result.duration_seconds = time.time() - t0
    return result
