"""Unified candidate-pool builder (Sec 3.1 candidates/pool_builder.py, Sec 4.1).

Runs the requested generators on a task, merges their outputs into a single fixed
pool with full provenance, and reports per-method counts and the cross-lineage
behavioural redundancy that motivates PQPO. The merged pool is exactly what the
fixed-pool selectors (PQPO + baselines) then operate over.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..data.datastructures import CandidatePrompt, TaskExample
from ..logging_utils.progress import titer, log
from .ape_adapter import APEAdapter
from .base import GenerationConfig, GenerationResult
from .capo_adapter import CAPOAdapter
from .gepa_adapter import GEPAAdapter
from .miprov2_adapter import MIPROv2Adapter

GENERATORS = {"GEPA": GEPAAdapter, "MIPROv2": MIPROv2Adapter,
              "CAPO": CAPOAdapter, "APE": APEAdapter}


@dataclass
class PoolBuildResult:
    task_id: str
    candidates: list[CandidatePrompt]
    per_method_counts: dict[str, int]
    per_method_calls: dict[str, dict]
    metadata: dict = field(default_factory=dict)


def build_pool(task_id: str, task_description: str, label_set: list[str],
               seed_prompts: list[CandidatePrompt], train: list[TaskExample],
               proposal_llm, scorer, executor, rng: np.random.Generator,
               methods: list[str] = None, per_method_size: int = 60,
               generations: int = 6, population: int = 8,
               minibatch: int = 8,
               method_overrides: dict = None) -> PoolBuildResult:
    """method_overrides: {method -> {pool_size/generations/population/minibatch: v}}
    lets a single method (e.g. MIPROv2 via --mipro-auto) use its own budget."""
    methods = methods or list(GENERATORS.keys())
    method_overrides = method_overrides or {}
    all_candidates: list[CandidatePrompt] = []
    counts, calls = {}, {}
    for m in titer(methods, desc=f"generators[{task_id}]", total=len(methods)):
        ov = method_overrides.get(m, {})
        cfg = GenerationConfig(
            task_id=task_id, task_description=task_description, label_set=label_set,
            pool_size=ov.get("pool_size", per_method_size),
            minibatch=ov.get("minibatch", minibatch),
            generations=ov.get("generations", generations),
            population=ov.get("population", population),
            seed=int(rng.integers(1e9)))
        gen = GENERATORS[m](proposal_llm, scorer, executor)
        res = gen.generate(cfg, seed_prompts, train, rng)
        all_candidates.extend(res.candidates)
        counts[m] = len(res.candidates)
        calls[m] = {"proposal": res.proposal_calls, "reflection": res.reflection_calls,
                    "target": res.target_calls}
        log(f"  {m}: {len(res.candidates)} candidates "
            f"(proposal={res.proposal_calls}, reflection={res.reflection_calls}, "
            f"target={res.target_calls})")

    # dedup by exact prompt text, keeping first (preserves provenance order)
    seen, deduped = set(), []
    for c in all_candidates:
        key = c.prompt_text
        if key not in seen:
            seen.add(key); deduped.append(c)

    return PoolBuildResult(
        task_id=task_id, candidates=deduped,
        per_method_counts=counts, per_method_calls=calls,
        metadata={"n_raw": len(all_candidates), "n_deduped": len(deduped),
                  "lineage_mix": dict(Counter(c.source_method for c in deduped))})
