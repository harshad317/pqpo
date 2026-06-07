"""APE candidate generator.

APE = Automatic Prompt Engineer: propose a batch of task instructions, evaluate
them on labelled minibatches, and keep the scored prompt candidates. This adapter
emits the full proposal stream so downstream selectors can compare APE directly
against GEPA/MIPROv2/CAPO or merge it into the PQPO fixed pool.
"""
from __future__ import annotations

import math

from .base import CandidateGenerator, GenerationConfig, GenerationResult, mint_candidate


class APEAdapter(CandidateGenerator):
    name = "APE"

    def generate(self, cfg: GenerationConfig, seed_prompts, train, rng) -> GenerationResult:
        candidates = []
        prop_calls = target_calls = 0
        idx = 0

        demo_n = min(len(train), max(4, cfg.minibatch))
        demo_idx = rng.choice(len(train), size=demo_n, replace=False) if train else []
        demos = [train[i] for i in demo_idx]
        demos_summary = "; ".join(f"{e.input_text[:40]} -> {e.target}" for e in demos[:4])

        # APE is proposal-heavy and simple: sample many instructions, score each,
        # and let the benchmark choose the best under the same labelled budget.
        rounds = max(1, cfg.generations)
        per_round = max(1, math.ceil(cfg.pool_size / rounds))
        best_acc = 0.0
        for gen in range(rounds):
            for _ in range(per_round):
                if len(candidates) >= cfg.pool_size:
                    break
                out = self.proposal_llm.propose_instruction(
                    cfg, demos_summary, best_acc, rng)
                prop_calls += 1
                c = mint_candidate(
                    cfg.task_id, self.name, out.text, idx, cfg.seed,
                    proposal_call_ids=[out.call_id],
                    behavior_profile_id=out.profile_id,
                    extra_meta={"generation": gen + 1},
                )
                idx += 1
                acc, _, n = self.minibatch_feedback(c, train, cfg, rng)
                target_calls += n
                best_acc = max(best_acc, acc)
                c.metadata["minibatch_acc"] = acc
                candidates.append(c)

        return GenerationResult(
            self.name, cfg.task_id, candidates,
            proposal_calls=prop_calls, reflection_calls=0, target_calls=target_calls,
            metadata={"generations_run": rounds, "n_candidates": len(candidates)},
        )
