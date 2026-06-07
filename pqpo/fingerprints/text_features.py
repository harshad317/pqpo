"""Lightweight deterministic text features for non-behavioural controls.

We avoid heavy embedding dependencies: a hashed character-n-gram vector is enough
to give lexical/embedding controls a fair, reproducible representation. For real
runs, swap in a sentence-transformer; the interface (dict prompt_id -> np.ndarray)
is unchanged.
"""
from __future__ import annotations

import hashlib

import numpy as np


def hashed_embedding(text: str, dim: int = 128, ngram: int = 3) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    s = text.lower()
    for i in range(max(1, len(s) - ngram + 1)):
        g = s[i:i + ngram]
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_embeddings(prompts, dim: int = 128) -> dict[str, np.ndarray]:
    return {p.prompt_id: hashed_embedding(p.prompt_text, dim) for p in prompts}


def lexical_token_set(text: str) -> set[str]:
    return set(text.lower().split())
