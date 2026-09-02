"""
Real training loop for the tiny transformer on text data. Loads actual
text, builds a real char-level vocabulary, trains with genuine batches,
and checkpoints — following the same "no fake progress" contract as the
rest of the training engine.
"""
from __future__ import annotations
import time
import numpy as np

from trainer.datasets import loader, analyzer
from trainer.tokenizers.char_tokenizer import CharTokenizer
from trainer.algorithms.transformer import TinyTransformer, TinyTransformerConfig
from trainer.training.checkpoint import CheckpointManager
from trainer.hardware import memory as hw


def _select_conservative_dims(vocab_size: int):
    """Pick model dims that fit comfortably in a 6GB budget by default."""
    return dict(context_length=32, d_model=64, n_layers=2, n_heads=4, d_ff=128)


def build_training_examples(token_ids, context_length):
    """Slides a window of context_length+1 over the token stream."""
    examples = []
    for start in range(0, len(token_ids) - context_length - 1, context_length):
        chunk = token_ids[start:start + context_length + 1]
        if len(chunk) == context_length + 1:
            examples.append(chunk)
    return np.array(examples)


class LMTrainingResult:
    def __init__(self):
        self.history = []
        self.final_metrics = {}
        self.model = None
        self.tokenizer = None
        self.status = "not_started"
        self.duration_seconds = None
        self.n_parameters = None
        self.memory_check = None


def train_language_model(cfg, checkpoint_mgr: CheckpointManager, progress_cb=None,
                          resume_from: dict | None = None) -> LMTrainingResult:
    result = LMTrainingResult()
    t0 = time.time()

    fmt = loader.detect_format(cfg.dataset_path)
    docs = loader.load(cfg.dataset_path)
    text = "\n".join(docs) if isinstance(docs, list) else str(docs)
    if len(text) < 200:
        raise ValueError(
            f"Text corpus is very small ({len(text)} characters) — the tiny transformer "
            f"needs more data to learn anything meaningful. Provide a larger .txt file."
        )

    tokenizer = CharTokenizer().fit(text)
    token_ids = np.array(tokenizer.encode(text))

    dims = _select_conservative_dims(tokenizer.vocab_size)
    tt_cfg = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=dims["context_length"], d_model=dims["d_model"],
        n_layers=dims["n_layers"], n_heads=dims["n_heads"], d_ff=dims["d_ff"],
        learning_rate=cfg.learning_rate, random_seed=cfg.random_seed,
    )

    # rough parameter memory estimate (embeddings + per-layer weights), reusing the
    # dense-layer estimator as an approximation of total learnable weight volume
    approx_layer_sizes = [tt_cfg.d_model, tt_cfg.d_ff, tt_cfg.d_model] * tt_cfg.n_layers
    est_bytes = hw.estimate_dense_nn_bytes(approx_layer_sizes, optimizer="adam") + \
        tokenizer.vocab_size * tt_cfg.d_model * 4 * 3
    ok, msg, ratio = hw.check_budget(est_bytes)
    result.memory_check = {"ok": ok, "message": msg}
    if not ok:
        tt_cfg.d_model = max(16, int(tt_cfg.d_model * ratio))
        tt_cfg.d_ff = max(32, int(tt_cfg.d_ff * ratio))
        result.memory_check["adjusted"] = {"d_model": tt_cfg.d_model, "d_ff": tt_cfg.d_ff}

    examples = build_training_examples(token_ids, tt_cfg.context_length)
    if len(examples) < 2:
        raise ValueError("Not enough text to build even one training batch at this context length.")

    n_val = max(1, int(len(examples) * 0.1))
    rng = np.random.default_rng(cfg.random_seed)
    perm = rng.permutation(len(examples))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_examples, val_examples = examples[train_idx], examples[val_idx]

    if resume_from is not None:
        model = TinyTransformer.from_state(resume_from["model_state"]["transformer_state"])
        tokenizer = CharTokenizer.from_state(resume_from["preprocessor_state"])
        start_epoch = resume_from["metadata"]["epoch"] + 1
        result.history = resume_from["metadata"].get("history", [])
    else:
        model = TinyTransformer(tt_cfg)
        start_epoch = 1

    batch_size = min(cfg.batch_size, len(train_examples))

    for epoch in range(start_epoch, cfg.epochs + 1):
        perm_e = rng.permutation(len(train_examples))
        batch_losses = []
        for start in range(0, len(train_examples), batch_size):
            idx = perm_e[start:start + batch_size]
            batch = train_examples[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            loss_val = model.train_step(x, y)
            batch_losses.append(loss_val)
        train_loss = float(np.mean(batch_losses))

        val_x, val_y = val_examples[:, :-1], val_examples[:, 1:]
        val_loss, _ = model.loss_and_backward(val_x, val_y)  # note: this also computes grads we discard
        # recompute val loss without mutating params (loss_and_backward doesn't apply grads by itself, safe)

        epoch_metrics = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": float(val_loss),
            "perplexity": float(np.exp(min(train_loss, 20))),
            "samples_processed": len(train_examples) * epoch,
        }
        result.history.append(epoch_metrics)
        if progress_cb:
            progress_cb(epoch_metrics)

        checkpoint_mgr.save(
            epoch=epoch, algorithm="tiny_transformer",
            model_state={"transformer_state": model.get_state()},
            preprocessor_state=tokenizer.to_state(), config=cfg,
            metrics=epoch_metrics, random_seed=cfg.random_seed,
        )

    result.final_metrics = result.history[-1] if result.history else {}
    result.model = model
    result.tokenizer = tokenizer
    result.status = "completed"
    result.n_parameters = model.n_parameters()
    result.duration_seconds = time.time() - t0
    return result
