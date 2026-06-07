"""MIPROv2 candidate generator (Sec 4.5).

MIPROv2 (DSPy; Opsahl-Ong et al., 2024):
  1. BOOTSTRAP few-shot demonstrations: run a teacher prompt over the train set
     and keep traces that pass the metric.
  2. PROPOSE instruction candidates with a grounded proposer (conditioned on the
     bootstrapped demos and a dataset summary).
  3. SEARCH the combinatorial space of (instruction x demo-subset) with a
     TPE/Bayesian surrogate, scoring minibatches.

We emit the full set of (instruction, demo-subset) prompts produced during the
search as the candidate pool. Demo-augmented prompts are longer (realistic) and
their behaviour profile (in sim) reflects the bootstrapped-demo quality.
"""
from __future__ import annotations

import numpy as np

from ..data.datastructures import CandidatePrompt, TaskExample
from .base import CandidateGenerator, GenerationConfig, GenerationResult, mint_candidate


class MIPROv2Adapter(CandidateGenerator):
    name = "MIPROv2"

    def generate(self, cfg: GenerationConfig, seed_prompts, train, rng) -> GenerationResult:
        candidates: list[CandidatePrompt] = []
        prop_calls = target_calls = 0
        idx = 0

        # 1. Bootstrap demos from a teacher prompt.
        teacher = seed_prompts[0] if seed_prompts else None
        demo_pool, teacher_correct = [], 0
        boot_n = min(len(train), max(12, cfg.minibatch * 2))
        boot_idx = rng.choice(len(train), size=boot_n, replace=False)
        for i in boot_idx:
            ex = train[i]
            if teacher is not None:
                s = self.scorer.score_one(teacher, ex, call_type="selector_dev")
                target_calls += 1
                if s >= 0.5:
                    teacher_correct += 1
                    if len(demo_pool) < 8:
                        demo_pool.append(ex)
        demo_quality = teacher_correct / boot_n if boot_n else 0.0
        demos_summary = "; ".join(f"{e.input_text[:30]} -> {e.target}" for e in demo_pool[:4])

        # 2. Propose grounded instruction candidates.
        n_instructions = max(4, cfg.pool_size // 6)
        instructions = []
        for _ in range(n_instructions):
            out = self.proposal_llm.propose_instruction(cfg, demos_summary, demo_quality, rng)
            prop_calls += 1
            instructions.append(out)

        # 3. Search (instruction x demo-subset). Emit all combos as candidates.
        demo_counts = [0, min(2, len(demo_pool)), min(4, len(demo_pool))]
        for out in instructions:
            for k in demo_counts:
                text = out.text
                if k > 0:
                    shots = "\n".join(
                        f"Input: {e.input_text}\nAnswer: {e.target}" for e in demo_pool[:k])
                    text = f"{out.text}\nExamples:\n{shots}"
                c = mint_candidate(cfg.task_id, self.name, text, idx, cfg.seed,
                                   parent=teacher, proposal_call_ids=[out.call_id],
                                   behavior_profile_id=out.profile_id,
                                   extra_meta={"n_demos": k, "demo_quality": demo_quality})
                idx += 1
                # minibatch score (surrogate signal); counted
                acc, _, n = self.minibatch_feedback(c, train, cfg, rng); target_calls += n
                c.metadata["minibatch_acc"] = acc
                candidates.append(c)
                if len(candidates) >= cfg.pool_size:
                    break
            if len(candidates) >= cfg.pool_size:
                break

        return GenerationResult(
            self.name, cfg.task_id, candidates[: cfg.pool_size],
            proposal_calls=prop_calls, reflection_calls=0, target_calls=target_calls,
            metadata={"demo_quality": demo_quality, "n_instructions": n_instructions})
