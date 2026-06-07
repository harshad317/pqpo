"""Scoring of prompts on labelled examples (Sec 3.x / 5).

A scorer runs a prompt on labelled examples through the APIExecutor and compares
the normalized answer to the gold target. Every call is logged to the cost
ledger with the supplied call_type, so labelled-eval budget is accounted exactly.
"""
from __future__ import annotations

from typing import Optional

from ..api.executor import APIExecutor
from ..data.datastructures import CandidatePrompt, TaskExample
from ..fingerprints.normalizers import OutputNormalizer


class Scorer:
    def __init__(self, executor: APIExecutor, normalizer: OutputNormalizer):
        self.executor = executor
        self.normalizer = normalizer

    def score_one(self, prompt: CandidatePrompt, example: TaskExample,
                  call_type: str = "selector_dev") -> float:
        trace = self.executor.run_prompt_on_example(prompt, example, call_type=call_type)
        parsed = self.normalizer.parse(trace.output_text)
        return self._match(parsed.normalized_answer, example.target)

    def score_many(self, prompt: CandidatePrompt, examples: list[TaskExample],
                   call_type: str = "selector_dev") -> list[float]:
        return [self.score_one(prompt, ex, call_type) for ex in examples]

    @staticmethod
    def _match(pred: Optional[str], gold: Optional[str]) -> float:
        if pred is None or gold is None:
            return 0.0
        return 1.0 if str(pred).strip().lower() == str(gold).strip().lower() else 0.0


def precompute_score_matrix(prompts, examples, scorer: Scorer,
                            call_type="selector_dev") -> dict[str, dict[str, float]]:
    """Optional full prompt x example matrix (Sec 5.5). Reproducibility convenience.
    Reported budget curves must still use *algorithm-visible* cost, not this."""
    matrix = {}
    for p in prompts:
        matrix[p.prompt_id] = {ex.example_id: scorer.score_one(p, ex, call_type)
                               for ex in examples}
    return matrix
