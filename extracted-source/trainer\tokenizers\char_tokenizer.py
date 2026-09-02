"""
Real character-level tokenizer. Simplest possible tokenization that has
zero out-of-vocabulary risk — appropriate for a tiny educational transformer
running on a 6GB machine, where a full BPE tokenizer would be overkill.
"""
from __future__ import annotations
from typing import List, Dict, Any


class CharTokenizer:
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    def fit(self, text: str):
        chars = sorted(set(text))
        # reserve 0 for a padding/unknown token
        self.char_to_id = {"<unk>": 0}
        self.id_to_char = {0: "<unk>"}
        for i, ch in enumerate(chars, start=1):
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(int(i), "") for i in ids)

    def to_state(self) -> Dict[str, Any]:
        return {"kind": "char_tokenizer", "char_to_id": self.char_to_id}

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "CharTokenizer":
        t = CharTokenizer()
        t.char_to_id = {k: int(v) for k, v in state["char_to_id"].items()}
        t.id_to_char = {v: k for k, v in t.char_to_id.items()}
        return t
