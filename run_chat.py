#!/usr/bin/env python3
"""Run an exported lightweight-ai-trainer tiny-transformer model locally.

Usage:
  python run_chat.py
  python run_chat.py --prompt "Hello" --max-new-tokens 160

The exported model directory is expected at ./model (or pass --model-dir).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))

from trainer.checkpoint import CheckpointManager
from trainer.models.tiny_transformer import TinyTransformer
from trainer.preprocessing.tokenizer import CharTokenizer


def chat_prompt(messages):
    parts = []
    for message in messages:
        role = message["role"].strip().capitalize()
        parts.append(f"{role}: {message['content'].strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def load_model(model_dir: Path):
    manager = CheckpointManager(str(model_dir.parent), model_dir.name)
    checkpoint = manager.load_latest()
    if not checkpoint:
        raise RuntimeError(f"No checkpoint found in {model_dir}")
    state = checkpoint.get("model_state", {})
    if not state.get("transformer_state") or not state.get("preprocessor_state"):
        raise RuntimeError("This export is not a tiny-transformer language model.")
    model = TinyTransformer.from_state(state["transformer_state"])
    tokenizer = CharTokenizer.from_state(state["preprocessor_state"])
    return model, tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Chat with an exported trained model")
    parser.add_argument("--model-dir", default="model", help="Exported model directory")
    parser.add_argument("--prompt", help="Send one prompt and exit")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_new_tokens > 1024:
        parser.error("--max-new-tokens must be between 1 and 1024")
    if args.temperature <= 0 or args.temperature > 2:
        parser.error("--temperature must be > 0 and <= 2")

    model_dir = Path(args.model_dir).resolve()
    model, tokenizer, checkpoint = load_model(model_dir)
    messages = []

    def answer(user_text):
        messages.append({"role": "user", "content": user_text})
        prompt = chat_prompt(messages)
        prompt_ids = tokenizer.encode(prompt)
        generated = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        continuation = generated[len(prompt_ids):]
        text = tokenizer.decode(continuation).strip()
        for marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
            text = text.split(marker, 1)[0].strip()
        messages.append({"role": "assistant", "content": text})
        return text

    if args.prompt:
        print(answer(args.prompt))
        return

    epoch = checkpoint.get("epoch", "latest")
    print(f"Loaded trained model: {model_dir.name} (epoch {epoch})")
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
            print("Conversation cleared.")
            continue
        try:
            print(f"Assistant: {answer(user_text)}\n")
        except Exception as exc:
            messages.pop()
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
