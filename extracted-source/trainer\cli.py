#!/usr/bin/env python3
"""
Real command-line interface. Every command here executes actual code paths
(dataset analysis, real training, real inference) — nothing is a stub.

Usage:
  python -m trainer.cli dataset <path>
  python -m trainer.cli create "<prompt>" --dataset <path> [--target COL] [--algorithm ALG] [--name NAME]
  python -m trainer.cli train --config <config.json>
  python -m trainer.cli resume --config <config.json>
  python -m trainer.cli predict --name <experiment_name> --input '{"col": val, ...}'
  python -m trainer.cli list
  python -m trainer.cli inspect --name <experiment_name>
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.datasets import loader, analyzer
from trainer.planner.nl_planner import plan
from trainer.core.pipeline import run_pipeline, MODELS_ROOT, EXPERIMENTS_ROOT
from trainer.core.config import TrainingConfig, ConfigValidationError
from trainer.inference.predictor import Predictor
from trainer.training.checkpoint import CheckpointManager
from trainer.algorithms.transformer import TinyTransformer
from trainer.tokenizers.char_tokenizer import CharTokenizer


def cmd_dataset(args):
    fmt = loader.detect_format(args.path)
    data = loader.load(args.path)
    if fmt in {"txt", "text_folder"}:
        stats = analyzer.analyze_text(data, path=args.path if fmt == "txt" else None)
    else:
        stats = analyzer.analyze_tabular(data, path=args.path)
    print(json.dumps(stats, indent=2, default=str))


def cmd_create(args):
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.name:
        overrides["experiment_name"] = args.name
    try:
        planned = plan(args.prompt, args.dataset, target_column=args.target,
                        algorithm=args.algorithm, **overrides)
    except (ValueError, ConfigValidationError) as e:
        print(f"PLANNING FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== Planner reasoning ===")
    for line in planned["reasoning"]:
        print(f" - {line}")
    print("\n=== Dataset analysis (real) ===")
    print(json.dumps(planned["dataset_analysis"], indent=2, default=str))
    print("\n=== Generated training configuration ===")
    print(planned["config"].to_json())

    cfg_path = f"{planned['config'].experiment_name}.config.json"
    planned["config"].to_json(cfg_path)
    print(f"\nSaved config to {cfg_path}. Run: python -m trainer.cli train --config {cfg_path}")


def cmd_train(args, resume=False):
    config = TrainingConfig.from_json(args.config)
    try:
        config.validate()
    except ConfigValidationError as e:
        print(f"CONFIG INVALID: {e}", file=sys.stderr)
        sys.exit(1)

    def progress(m):
        print(f"epoch {m['epoch']}: train_loss={m['train_loss']:.5f} val_loss={m['val_loss']:.5f}")

    out = run_pipeline(config, progress_cb=progress, resume=resume)
    result = out["result"]
    print("\n=== Training complete ===")
    print(f"status: {result.status}")
    print(f"duration: {result.duration_seconds:.2f}s")
    print(f"final metrics: {json.dumps(result.final_metrics, indent=2, default=str)}")
    print(f"model saved at: {out['model_dir']}")


def cmd_predict(args):
    predictor = Predictor(MODELS_ROOT, args.name, epoch=args.epoch)
    record = json.loads(args.input)
    records = record if isinstance(record, list) else [record]
    preds = predictor.predict(records)
    print(json.dumps(preds, indent=2, default=str))


def cmd_generate(args):
    mgr = CheckpointManager(MODELS_ROOT, args.name)
    ckpt = mgr.load_epoch(args.epoch) if args.epoch else mgr.load_latest()
    if ckpt is None:
        print(f"No trained checkpoint found for '{args.name}'.", file=sys.stderr)
        sys.exit(1)
    model = TinyTransformer.from_state(ckpt["model_state"]["transformer_state"])
    tokenizer = CharTokenizer.from_state(ckpt["preprocessor_state"])
    prompt_ids = tokenizer.encode(args.prompt)
    out_ids = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(tokenizer.decode(out_ids))


def cmd_list(args):
    if not os.path.exists(MODELS_ROOT):
        print("No trained models yet.")
        return
    for name in sorted(os.listdir(MODELS_ROOT)):
        print(name)


def cmd_inspect(args):
    exp_path = os.path.join(EXPERIMENTS_ROOT, args.name, "experiment.json")
    if not os.path.exists(exp_path):
        print(f"No experiment record found for '{args.name}'.", file=sys.stderr)
        sys.exit(1)
    with open(exp_path) as f:
        print(json.dumps(json.load(f), indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(prog="trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dataset = sub.add_parser("dataset", help="Analyze a dataset")
    p_dataset.add_argument("path")
    p_dataset.set_defaults(func=cmd_dataset)

    p_create = sub.add_parser("create", help="Plan a training config from a natural-language prompt")
    p_create.add_argument("prompt")
    p_create.add_argument("--dataset", required=True)
    p_create.add_argument("--target", default=None)
    p_create.add_argument("--algorithm", default=None)
    p_create.add_argument("--name", default=None)
    p_create.add_argument("--epochs", type=int, default=None)
    p_create.set_defaults(func=cmd_create)

    p_train = sub.add_parser("train", help="Run real training from a config file")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=lambda a: cmd_train(a, resume=False))

    p_resume = sub.add_parser("resume", help="Resume training from the latest checkpoint")
    p_resume.add_argument("--config", required=True)
    p_resume.set_defaults(func=lambda a: cmd_train(a, resume=True))

    p_predict = sub.add_parser("predict", help="Run inference with a trained model")
    p_predict.add_argument("--name", required=True)
    p_predict.add_argument("--input", required=True, help="JSON object or list of objects")
    p_predict.add_argument("--epoch", type=int, default=None)
    p_predict.set_defaults(func=cmd_predict)

    p_generate = sub.add_parser("generate", help="Generate text from a trained tiny transformer")
    p_generate.add_argument("--name", required=True)
    p_generate.add_argument("--prompt", required=True)
    p_generate.add_argument("--max_new_tokens", type=int, default=100)
    p_generate.add_argument("--temperature", type=float, default=0.8)
    p_generate.add_argument("--epoch", type=int, default=None)
    p_generate.set_defaults(func=cmd_generate)

    p_list = sub.add_parser("list", help="List trained models")
    p_list.set_defaults(func=cmd_list)

    p_inspect = sub.add_parser("inspect", help="Show full experiment record")
    p_inspect.add_argument("--name", required=True)
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
