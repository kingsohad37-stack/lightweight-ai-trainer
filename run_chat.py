#!/usr/bin/env python3
"""Run an exported lightweight-ai-trainer tiny-transformer model locally."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))

from trainer.algorithms.transformer import TinyTransformer
from trainer.tokenizers.char_tokenizer import CharTokenizer
from trainer.training.checkpoint import CheckpointManager


def chat_prompt(messages):
    parts = []
    for message in messages:
        role = {"system": "System", "user": "User", "assistant": "Assistant"}.get(message["role"].lower(), "User")
        parts.append(f"{role}: {message['content'].strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def load_model(model_dir: Path):
    manager = CheckpointManager(str(model_dir.parent), model_dir.name)
    checkpoint = manager.load_latest()
    if not checkpoint:
        raise RuntimeError(f"No checkpoint found in {model_dir}")
    metadata = checkpoint.get("metadata", {})
    if metadata.get("algorithm") != "tiny_transformer":
        raise RuntimeError("This export is not a tiny-transformer language model.")
    state = checkpoint.get("model_state", {})
    if not state.get("transformer_state") or not checkpoint.get("preprocessor_state"):
        raise RuntimeError("The exported checkpoint is missing transformer/tokenizer state.")
    return (TinyTransformer.from_state(state["transformer_state"]),
            CharTokenizer.from_state(checkpoint["preprocessor_state"]), checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Chat with an exported trained model")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("--max-new-tokens must be between 1 and 1024")
    if not 0 < args.temperature <= 2:
        parser.error("--temperature must be > 0 and <= 2")

    model, tokenizer, checkpoint = load_model(Path(args.model_dir).resolve())
    messages = []

    def answer(user_text):
        messages.append({"role": "user", "content": user_text})
        prompt_ids = tokenizer.encode(chat_prompt(messages))
        generated = model.generate(prompt_ids, args.max_new_tokens, args.temperature)
        text = tokenizer.decode(generated[len(prompt_ids):]).strip()
        for marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
            text = text.split(marker, 1)[0].strip()
        messages.append({"role": "assistant", "content": text})
        return text

    if args.prompt:
        print(answer(args.prompt))
        return

    print(f"Loaded trained model: {Path(args.model_dir).name} (epoch {checkpoint.get('metadata', {}).get('epoch', 'latest')})")
    print("Type /exit to quit, /clear to reset conversation.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/clear":
            messages.clear()
            print("Conversation cleared.\n")
            continue
        try:
            print(f"Assistant: {answer(user_text)}\n")
        except Exception as exc:
            messages.pop()
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
