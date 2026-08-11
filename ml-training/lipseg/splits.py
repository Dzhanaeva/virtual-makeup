from __future__ import annotations

import hashlib


def subject_split(subject_id: str, split_seed: str, splits: dict[str, float]) -> str:
    digest = hashlib.sha256(f"{split_seed}:{subject_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    train_limit = splits["train"]
    validation_limit = train_limit + splits["validation"]
    if value < train_limit:
        return "train"
    if value < validation_limit:
        return "validation"
    return "test"
