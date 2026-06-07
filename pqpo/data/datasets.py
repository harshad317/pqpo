"""MVP task families + synthetic data + candidate-pool builder.

For real runs, replace `make_synthetic_task` with loaders for TREC, GSM8K-style
reasoning, and a JSON-extraction task (Sec 5.1). The synthetic generator here is
ONLY used with the simulated target so the full pipeline can be validated.

Crucially, the pool builder injects a hidden ``behavior_profile_id`` into each
candidate's metadata. This is the ground-truth phenotype that the simulator
reads and the optimisation algorithms never see. Multiple source methods are
mapped onto a shared profile bank, which is what creates *cross-lineage
behavioural redundancy* — the phenomenon the paper measures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datastructures import CandidatePrompt, TaskExample

SOURCE_METHODS = ["GEPA", "MIPROv2", "CAPO", "APE", "paraphrase", "random_mutator"]


@dataclass
class TaskSpec:
    task_id: str
    family: str                 # classification | reasoning | extraction
    label_set: list[str]
    description: str


def default_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec("trec", "classification", ["A", "B", "C", "D", "E", "F"],
                 "Classify the question into one of six coarse answer types."),
        TaskSpec("reason", "reasoning", ["A", "B", "C", "D"],
                 "Solve the multiple-choice reasoning problem; give the final letter."),
        TaskSpec("extract", "extraction", ["A", "B", "C"],
                 "Extract the requested field and return the answer label."),
    ]


def make_synthetic_examples(spec: TaskSpec, n: int, rng, split: str) -> list[TaskExample]:
    labels = spec.label_set
    examples = []
    for i in range(n):
        gold = labels[int(rng.integers(0, len(labels)))]
        examples.append(TaskExample(
            example_id=f"{spec.task_id}_{split}_{i:04d}",
            task_id=spec.task_id,
            input_text=f"[{spec.family}] item {i} :: features={rng.integers(0, 9999)}",
            target=gold,
            metadata={"split": split},
        ))
    return examples


def build_candidate_pool(spec: TaskSpec, n_prompts: int, profile_ids: list[str],
                         rng, rarity: float = 1.6) -> list[CandidatePrompt]:
    """Construct n_prompts candidates spread across source methods and profiles.

    Profiles are assigned so that several prompts from *different* source methods
    share a profile (cross-lineage redundancy). The assignment is skewed so that
    higher-index profiles (which are higher-skill in the default bank) are RARER:
    weight_i proportional to 1/(1+i)^rarity. This makes the best behavioural cell
    scarce, which is precisely the regime PQPO is designed to exploit."""
    weights = np.array([1.0 / (1.0 + i) ** rarity for i in range(len(profile_ids))])
    weights /= weights.sum()
    # Guarantee the best (last) profile a small but non-zero presence (~6%).
    k_best = max(2, n_prompts // 16)
    forced_best = set(range(n_prompts - k_best, n_prompts))
    prompts = []
    for i in range(n_prompts):
        src = SOURCE_METHODS[i % len(SOURCE_METHODS)]
        if i in forced_best:
            profile = profile_ids[-1]
        else:
            profile = profile_ids[int(rng.choice(len(profile_ids), p=weights))]
        text = (f"You are solving a {spec.family} task. {spec.description} "
                f"[style:{src}:{i}] Respond as 'Answer: <label>'.")
        prompts.append(CandidatePrompt(
            prompt_id=f"{spec.task_id}_{src}_{i:05d}",
            task_id=spec.task_id,
            source_method=src,
            prompt_text=text,
            parent_prompt_id=None,
            token_length=len(text.split()),
            generation_seed=int(rng.integers(0, 1_000_000)),
            metadata={"behavior_profile_id": profile},
        ))
    return prompts
