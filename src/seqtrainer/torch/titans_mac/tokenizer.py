"""Fixed, dependency-free tokenizer for DNA bases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import Tensor


class DNABaseTokenizer:
    """Map DNA bases to ``PAD=0, N/UNK=1, A=2, C=3, G=4, T=5``."""

    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 1
    N_TOKEN_ID = 1
    A_TOKEN_ID = 2
    C_TOKEN_ID = 3
    G_TOKEN_ID = 4
    T_TOKEN_ID = 5
    TOKEN_TO_ID = {"PAD": 0, "N": 1, "A": 2, "C": 3, "G": 4, "T": 5}
    ID_TO_TOKEN = {0: "", 1: "N", 2: "A", 3: "C", 4: "G", 5: "T"}

    def __init__(self, max_length: Optional[int] = None) -> None:
        self.max_length = max_length

    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> list[int]:
        limit = max_length if max_length is not None else self.max_length
        sequence = (sequence or "").upper()
        if limit is not None and len(sequence) > limit:
            if not truncation:
                raise ValueError(f"sequence length {len(sequence)} exceeds max_length={limit}")
            sequence = sequence[:limit]
        ids = [self.TOKEN_TO_ID.get(base, self.UNK_TOKEN_ID) for base in sequence]
        if padding and limit is not None:
            ids.extend([self.PAD_TOKEN_ID] * (limit - len(ids)))
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        bases: list[str] = []
        for token_id in token_ids:
            value = int(token_id)
            if value == self.PAD_TOKEN_ID and skip_special_tokens:
                continue
            bases.append(self.ID_TO_TOKEN.get(value, "N"))
        return "".join(bases)

    def batch_encode(
        self,
        sequences: list[str],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> tuple[Tensor, Tensor]:
        limit = max_length if max_length is not None else self.max_length
        if padding and limit is None:
            limit = max((len(sequence) for sequence in sequences), default=0)
        encoded = [self.encode(seq, max_length=limit, padding=padding) for seq in sequences]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = input_ids.ne(self.PAD_TOKEN_ID).long()
        return input_ids, attention_mask

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "DNABaseTokenizer",
            "version": 1,
            "token_to_id": self.TOKEN_TO_ID,
            "max_length": self.max_length,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_file(cls, path: str | Path) -> "DNABaseTokenizer":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if values.get("token_to_id") != cls.TOKEN_TO_ID:
            raise ValueError("tokenizer file does not use the SeqTrainer DNA token contract")
        return cls(max_length=values.get("max_length"))
