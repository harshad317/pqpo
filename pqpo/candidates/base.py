"""Shared scaffolding for candidate generators (Sec 3.1 candidates/, Sec 4.4–4.6).

A CandidateGenerator turns a task + seed prompts into a pool of CandidatePrompt
objects, logging every proposal/reflection call to the cost ledger and recording
full provenance (source_method, parent, generation_seed, proposal/reflection
call ids). The generation algorithms (genetic loop, bootstrap, racing) live in the
per-method adapters; this module holds the common pieces.

Dual-mode
  * provider='sim'  -> a SimProposalLLM supplies prompt text AND a hidden behaviour
    profile so the simulated target differentiates candidates; meta-call cost is
    logged via executor.record_meta_call.
  * provider real   -> an LLMProposalLLM calls the billed provider via
    executor.complete; no profile is attached (the real target defines behaviour).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..api.executor import APIExecutor
from ..data.datastructures import CandidatePrompt, TaskExample
from ..evaluation.scorers import Scorer


@dataclass
class GenerationConfig:
    task_id: str
    task_description: str
    label_set: list[str]
    pool_size: int = 60               # target number of candidates from this method
    minibatch: int = 8                # labelled examples used per feedback step
    generations: int = 6
    population: int = 8
    seed: int = 0
    max_prompt_tokens: int = 220      # CAPO length cap / general guard


@dataclass
class GenerationResult:
    method: str
    task_id: str
    candidates: list[CandidatePrompt]
    proposal_calls: int = 0
    reflection_calls: int = 0
    target_calls: int = 0
    metadata: dict = field(default_factory=dict)


class CandidateGenerator:
    """Base class. Subclasses implement ``generate``."""

    name = "base"

    def __init__(self, proposal_llm, scorer: Scorer, executor: APIExecutor):
        self.proposal_llm = proposal_llm
        self.scorer = scorer
        self.executor = executor

    def generate(self, cfg: GenerationConfig, seed_prompts: list[CandidatePrompt],
                 train: list[TaskExample], rng: np.random.Generator) -> GenerationResult:
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------- #
    def minibatch_feedback(self, prompt: CandidatePrompt, train: list[TaskExample],
                           cfg: GenerationConfig, rng) -> tuple[float, str, int]:
        """Run the prompt on a labelled minibatch; return (accuracy, feedback, n_calls).

        The feedback string summarises errors for the reflective proposer. These
        target-model calls are counted (call_type=selector_dev) because generators
        legitimately consume labelled budget (Sec 4.1)."""
        idx = rng.choice(len(train), size=min(cfg.minibatch, len(train)), replace=False)
        batch = [train[i] for i in idx]
        correct, wrong_examples = 0, []
        for ex in batch:
            s = self.scorer.score_one(prompt, ex, call_type="selector_dev")
            if s >= 0.5:
                correct += 1
            elif len(wrong_examples) < 3:
                wrong_examples.append(ex.input_text[:80])
        acc = correct / len(batch)
        fb = (f"accuracy={acc:.2f} on {len(batch)} examples; "
              f"sample failures: {wrong_examples}")
        return acc, fb, len(batch)


def subsample_spread(candidates: list, k: int) -> list:
    """Keep k candidates evenly spread across the search order.

    Generators append candidates roughly low-quality (early/seed) -> high-quality
    (late/evolved). Truncating with [:k] would keep only the weak early ones, so
    we instead sample evenly across the full list to PRESERVE the skill gradient
    (and the late high-skill candidates) the pool needs."""
    if len(candidates) <= k:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, k).round().astype(int)
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(int(i)); out.append(candidates[int(i)])
    # top up with the most-evolved (last) candidates if rounding collided
    j = len(candidates) - 1
    while len(out) < k and j >= 0:
        if j not in seen:
            seen.add(j); out.append(candidates[j])
        j -= 1
    return out


def mint_candidate(task_id: str, source_method: str, prompt_text: str,
                   index: int, seed: int, parent: Optional[CandidatePrompt] = None,
                   proposal_call_ids: list[str] = None,
                   reflection_call_ids: list[str] = None,
                   behavior_profile_id: Optional[str] = None,
                   extra_meta: dict = None) -> CandidatePrompt:
    meta = dict(extra_meta or {})
    if behavior_profile_id is not None:
        meta["behavior_profile_id"] = behavior_profile_id
    return CandidatePrompt(
        prompt_id=f"{task_id}_{source_method}_{index:05d}",
        task_id=task_id,
        source_method=source_method,
        prompt_text=prompt_text,
        parent_prompt_id=parent.prompt_id if parent else None,
        proposal_call_ids=proposal_call_ids or [],
        reflection_call_ids=reflection_call_ids or [],
        token_length=len(prompt_text.split()),
        generation_seed=seed,
        metadata=meta,
    )
