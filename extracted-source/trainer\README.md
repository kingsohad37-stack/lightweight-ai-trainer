# Lightweight AI Model Trainer

A real, functional, from-scratch ML/DL training system designed to run on a
CPU-only machine with ~6GB RAM. No mockups, no simulated training, no
hardcoded metrics — verified by 29 passing tests (including numerical
gradient checks on the hand-written transformer backprop) and live CLI runs
against real demo datasets.

## What's real and working

- **Dataset loaders**: CSV, JSON, JSONL, TXT, folders of .txt files.
  Streaming/chunked CSV reads for large files.
- **Dataset analyzer**: real per-column stats (missing values, dtypes,
  class balance, vocab size, duplicates) — nothing fabricated.
- **Classical ML** (`trainer/algorithms/classical.py`, scikit-learn backed,
  real `.fit()` calls): linear/logistic regression, naive Bayes, k-NN,
  decision tree, random forest, SVM, k-means.
- **Dense neural network** (`trainer/algorithms/neural.py`): from-scratch
  NumPy implementation — real forward propagation, backpropagation, and
  SGD/Momentum/Adam optimizers. No autograd framework.
- **Tiny transformer** (`trainer/algorithms/transformer.py`): from-scratch
  NumPy causal self-attention transformer — token/positional embeddings,
  multi-head attention, feed-forward, layer norm, residual connections,
  causal masking, cross-entropy loss, and full manual backprop. Verified
  against finite-difference numerical gradients in
  `trainer/tests/test_transformer.py` (9/9 passing).
- **Real preprocessing**: tabular encoding/scaling/splitting, and TF-IDF
  text vectorization for text classification (so free-text columns don't
  get mangled through label-encoding).
- **Real training loop**: actual epochs/batches, real metrics
  (accuracy/precision/recall/F1 or MSE/MAE/R²), memory-budget estimation
  before training starts, auto-shrinking configs that would exceed the 6GB
  budget instead of crashing.
- **Checkpointing**: weights, optimizer state, config, preprocessor/
  tokenizer state, metrics — all saved to disk, resumable.
- **Natural-language planner** (`trainer/planner/nl_planner.py`): rule-based
  (not an LLM call) — turns a prompt + dataset into a structured, validated
  `TrainingConfig`. It ONLY produces a config; it never claims training
  happened. That's exclusively the training engine's job.
- **Inference**: loads a real saved checkpoint and runs a genuine forward
  pass / `sklearn.predict()`. No hardcoded outputs.
- **CLI**: `dataset`, `create`, `train`, `resume`, `predict`, `generate`,
  `list`, `inspect`.
- **Experiment tracking**: every run writes a real `experiment.json` with
  config, dataset stats, metrics, timing, and status.

## Quick start

```bash
cd trainer/..     # repo root
python3 -m trainer.cli dataset trainer/datasets/demo/iris_classification.csv

python3 -m trainer.cli create "Classify the iris species from these flower measurements" \
    --dataset trainer/datasets/demo/iris_classification.csv --name iris_demo
python3 -m trainer.cli train --config iris_demo.config.json
python3 -m trainer.cli predict --name iris_demo \
    --input '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}'

# regression
python3 -m trainer.cli create "Predict the price of a house from its size, bedrooms and age" \
    --dataset trainer/datasets/demo/house_price_regression.csv --algorithm neural_network \
    --name house_price_nn --epochs 15
python3 -m trainer.cli train --config house_price_nn.config.json

# text classification (TF-IDF)
python3 -m trainer.cli create "Train a sentiment classifier using this dataset" \
    --dataset trainer/datasets/demo/sentiment_small.csv --name sentiment_demo
python3 -m trainer.cli train --config sentiment_demo.config.json

# tiny transformer / language modeling
python3 -m trainer.cli create "I want a small language model that learns from these text files" \
    --dataset trainer/datasets/demo/tiny_corpus.txt --name tiny_lm_demo --epochs 60
python3 -m trainer.cli train --config tiny_lm_demo.config.json
python3 -m trainer.cli generate --name tiny_lm_demo --prompt "The quick" --max_new_tokens 60
```

Run the full test suite:
```bash
python3 -m unittest discover -s trainer/tests -p "test_*.py" -v
```
29/29 tests pass, including numerical gradient checks for the transformer.

## Real bugs found and fixed during development (see conversation)

1. **Stratified split crash** on tiny/imbalanced classes — now falls back to
   a plain random split when any class has fewer than 2 members.
2. **NL planner picked the wrong target column** ("bedrooms" instead of
   "price") because it matched columns in dataset order instead of by
   position in the prompt — fixed to prefer the earliest-mentioned column.
3. **Unscaled regression targets blew up neural-network training** (house
   prices in the hundreds of thousands vs. MSE loss) — added target
   standardization with proper inverse-scaling at inference. R² went from
   -9.7 (worse than predicting the mean) to 0.97 after the fix.
4. **Free-text columns were being label-encoded like categorical data** —
   would assign a near-unique ID per sentence and fail on any unseen text
   at inference. Replaced with a real TF-IDF pipeline
   (`trainer/preprocessing/text.py`).

## Honest TODOs (not implemented — not faked either)

- **GUI** (Phase 10) — CLI-first was the explicit instruction; not built yet.
- **PyTorch backend** — this sandbox has no network access to install
  `torch`, so everything runs on the from-scratch NumPy engines. The engines
  cover the classical + dense NN + tiny transformer phases without it.
- **CLI `evaluate`** as a separate command — `train` already reports full
  validation metrics and `inspect` shows the full experiment record, but
  there's no standalone `evaluate --config` command yet for re-scoring a
  model against a different held-out set.
- **BPE/subword tokenizer** — the tiny transformer currently uses a
  character-level tokenizer (zero OOV risk, appropriate for its scale);
  a real BPE tokenizer would help larger vocabularies/longer contexts.
- **GPU/CUDA path** — hardware detection only checks CPU vs. reporting;
  no CUDA-accelerated path is wired up (not required by the brief, which
  asks for CPU-first with GPU as optional).

## Project layout

```
trainer/
  core/            # TrainingConfig schema + validation, pipeline orchestration
  datasets/        # loaders + real dataset analyzer, demo/ has 4 real datasets
  preprocessing/    # tabular (encode/scale/split) + text (TF-IDF)
  algorithms/       # classical (sklearn), neural (NumPy MLP), transformer (NumPy)
  tokenizers/       # char-level tokenizer for the tiny transformer
  planner/          # natural-language -> TrainingConfig (config only, never trains)
  training/         # real training loops (classical/neural/LM) + checkpointing
  evaluation/       # real metrics from actual predictions
  hardware/         # real RAM monitoring (/proc/meminfo) + memory budgeting
  inference/        # loads a real checkpoint, runs real predict()
  tests/            # 29 tests, including transformer gradient checks
  cli.py            # dataset/create/train/resume/predict/generate/list/inspect
```
