"""Deterministic request hashing + local on-disk response cache.

Every target-model request is hashed over the fields that affect the output. A
cache hit replays the stored output at zero marginal dollar cost but is still
*counted* as a logical call where the protocol requires it (the CostLedger
distinguishes cache hits from misses). This makes whole experiments replayable
without re-billing the provider.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Optional


def request_hash(
    model_key: str,
    model_version: Optional[str],
    full_input_text: str,
    temperature: float,
    max_output_tokens: int,
    seed: Optional[int],
) -> str:
    payload = {
        "model_key": model_key,
        "model_version": model_version or "",
        "input": full_input_text,
        "temperature": round(float(temperature), 6),
        "max_output_tokens": int(max_output_tokens),
        "seed": seed if seed is not None else "none",
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class APICache:
    """A simple JSONL-backed key/value cache keyed by request_hash."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()       # safe under parallel workers
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._mem[rec["request_hash"]] = rec

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._mem.get(key)

    def put(self, key: str, record: dict) -> None:
        with self._lock:
            self._mem[key] = record
            if self.path:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"request_hash": key, **record}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._mem)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
