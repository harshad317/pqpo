"""Real benchmark loaders for the MVP triple (Banking77 / GSM8K / HoVer).

Each loader returns a ``LoadedTask`` with a TaskSpec, the matching output
normalizer, and deterministic splits: an unlabelled *sentinel* panel, a labelled
selector-dev set, and a held-out test set (Sec 5.1). HuggingFace ``datasets`` is
an optional dependency — if it is absent (or offline), pass ``source="synthetic"``
to fall back to the simulated generator so the pipeline still runs.

Splits are carved deterministically from a fixed RNG seed so runs are reproducible.

Fingerprint note: the sentinel panel is drawn from the *task inputs* with labels
withheld, so behaviour fingerprints never see gold labels (Sec 2.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .datasets import TaskSpec, make_synthetic_examples
from .datastructures import TaskExample
from ..fingerprints.normalizers import (Banking77Normalizer, GSM8KNormalizer,
                                        HoVerNormalizer, OutputNormalizer)


@dataclass
class LoadedTask:
    spec: TaskSpec
    normalizer: OutputNormalizer
    sentinels: list[TaskExample]
    dev: list[TaskExample]
    test: list[TaskExample]
    label_set: list[str]


def _have_datasets() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except Exception:
        return False


def _split(examples: list[TaskExample], n_sentinel: int, n_dev: int, n_test: int,
           seed: int) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))
    take = lambda a, b: [examples[i] for i in idx[a:b]]
    s = take(0, n_sentinel)
    d = take(n_sentinel, n_sentinel + n_dev)
    t = take(n_sentinel + n_dev, n_sentinel + n_dev + n_test)
    for u in s:                       # sentinels are unlabelled
        u.target = None
    return s, d, t


# --------------------------------------------------------------------------- #
# Individual loaders
# --------------------------------------------------------------------------- #
def load_banking77(n_sentinel=12, n_dev=96, n_test=300, seed=0,
                   source="hf") -> LoadedTask:
    label_names = _BANKING77_INTENTS
    spec = TaskSpec("banking77", "classification", label_names,
                    "Classify the customer banking query into one of 77 intents.")
    if source == "hf" and _have_datasets():
        from datasets import load_dataset
        ds = load_dataset("banking77", split="train")
        names = ds.features["label"].names
        spec.label_set = names
        ex = [TaskExample(f"banking77_{i}", "banking77", r["text"], names[r["label"]])
              for i, r in enumerate(ds)]
    else:
        ex = _synthetic_for(spec, n_sentinel + n_dev + n_test + 50, seed)
    s, d, t = _split(ex, n_sentinel, n_dev, n_test, seed)
    return LoadedTask(spec, Banking77Normalizer(spec.label_set), s, d, t, spec.label_set)


def load_gsm8k(n_sentinel=12, n_dev=96, n_test=300, seed=0, source="hf") -> LoadedTask:
    spec = TaskSpec("gsm8k", "reasoning", [], "Solve the grade-school math word "
                    "problem. End with '#### <integer answer>'.")
    if source == "hf" and _have_datasets():
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        ex = []
        for i, r in enumerate(ds):
            ans = r["answer"].split("####")[-1].strip().replace(",", "")
            ex.append(TaskExample(f"gsm8k_{i}", "gsm8k", r["question"], ans))
    else:
        ex = _synthetic_for(TaskSpec("gsm8k", "reasoning", ["A", "B", "C", "D"], ""),
                            n_sentinel + n_dev + n_test + 50, seed)
    s, d, t = _split(ex, n_sentinel, n_dev, n_test, seed)
    return LoadedTask(spec, GSM8KNormalizer(), s, d, t, spec.label_set)


def load_hover(n_sentinel=12, n_dev=96, n_test=300, seed=0, source="hf") -> LoadedTask:
    labels = ["SUPPORTED", "NOT_SUPPORTED"]
    spec = TaskSpec("hover", "verification", labels,
                    "Verify whether the claim is SUPPORTED or NOT_SUPPORTED by "
                    "world knowledge. Answer with the label.")
    if source == "hf" and _have_datasets():
        from datasets import load_dataset
        ds = load_dataset("hover", split="train")
        # HoVer label: 0 = SUPPORTED, 1 = NOT_SUPPORTED
        ex = [TaskExample(f"hover_{i}", "hover", r["claim"], labels[r["label"]])
              for i, r in enumerate(ds)]
    else:
        ex = _synthetic_for(TaskSpec("hover", "classification", labels, ""),
                            n_sentinel + n_dev + n_test + 50, seed)
    s, d, t = _split(ex, n_sentinel, n_dev, n_test, seed)
    return LoadedTask(spec, HoVerNormalizer(), s, d, t, labels)


LOADERS = {"banking77": load_banking77, "gsm8k": load_gsm8k, "hover": load_hover}


def load_task(name: str, **kw) -> LoadedTask:
    if name not in LOADERS:
        raise ValueError(f"unknown task '{name}'; choose from {list(LOADERS)}")
    return LOADERS[name](**kw)


def _synthetic_for(spec: TaskSpec, n: int, seed: int) -> list[TaskExample]:
    if not spec.label_set:
        spec.label_set = ["A", "B", "C", "D"]
    return make_synthetic_examples(spec, n, np.random.default_rng(seed), "all")


# A compact static fallback list of Banking77 intent names (used when offline so
# the synthetic path still has the right 77-way label space). The canonical names
# come from the HF dataset features when available.
_BANKING77_INTENTS = [f"intent_{i:02d}" for i in range(77)]
