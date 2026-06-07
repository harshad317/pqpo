"""Shared selector scaffolding.

All fixed-pool selectors consume a *labelled budget* B = number of labelled
(prompt, example) evaluations on the selector-dev split. The BudgetMeter enforces
the cap; the CostLedger (via the executor) independently records dollar cost.
This separation lets us report both "labelled evals" and "dollars" honestly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..data.datastructures import BehaviorFingerprint, CandidatePrompt, TaskExample
from ..evaluation.scorers import Scorer


class BudgetMeter:
    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def spend(self, n: int = 1) -> bool:
        if self.remaining <= 0:
            return False
        self.used += n
        return True


@dataclass
class SelectorContext:
    prompts: list[CandidatePrompt]
    dev_examples: list[TaskExample]
    scorer: Scorer
    rng: np.random.Generator
    budget: int
    fingerprints: dict[str, BehaviorFingerprint] = field(default_factory=dict)
    D: Optional[np.ndarray] = None
    ids: list[str] = field(default_factory=list)
    prompt_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    source_methods: dict[str, str] = field(default_factory=dict)
    weights: dict = None

    def __post_init__(self):
        self.prompt_by_id = {p.prompt_id: p for p in self.prompts}
        self.index_of = {pid: i for i, pid in enumerate(self.ids)}
        self._dev_cursor = 0

    def sample_dev(self, n: int) -> list[TaskExample]:
        """Draw n dev examples without replacement (cycling if exhausted)."""
        out = []
        for _ in range(n):
            out.append(self.dev_examples[self._dev_cursor % len(self.dev_examples)])
            self._dev_cursor += 1
        return out

    def eval_prompt(self, prompt_id: str, examples: list[TaskExample],
                    meter: BudgetMeter) -> list[float]:
        scores = []
        for ex in examples:
            if not meter.spend(1):
                break
            scores.append(self.scorer.score_one(self.prompt_by_id[prompt_id], ex))
        return scores


@dataclass
class SelectorResult:
    selected_prompt_id: str
    selected_cell_id: Optional[str] = None
    labeled_evals_used: int = 0
    metadata: dict = field(default_factory=dict)
