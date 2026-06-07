"""GEPA candidate generator (Sec 4.4).

GEPA = reflective, genetic-Pareto prompt evolution (Agrawal et al., 2025):
  * run a candidate on a labelled minibatch to obtain feedback,
  * an LLM REFLECTS on the failures and proposes an improved prompt (mutation),
  * occasionally CROSSOVER two strong parents (merge lessons),
  * keep a Pareto/elite archive and re-seed from it.

We emit EVERY candidate produced along the search (not just the winner) so the
fixed pool contains the realistic, behaviourally-redundant proposal stream PQPO
is meant to compress. All proposal/reflection calls are logged; minibatch feedback
calls are counted as selector_dev cost.
"""
from __future__ import annotations

import numpy as np

from ..data.datastructures import CandidatePrompt, TaskExample
from .base import (CandidateGenerator, GenerationConfig, GenerationResult,
                   mint_candidate, subsample_spread)


class GEPAAdapter(CandidateGenerator):
    name = "GEPA"

    def generate(self, cfg: GenerationConfig, seed_prompts, train, rng) -> GenerationResult:
        candidates: list[CandidatePrompt] = []
        prop_calls = refl_calls = target_calls = 0
        idx = 0

        # Initial population: seeds + freshly proposed inits.
        population: list[tuple[CandidatePrompt, float]] = []
        for sp in seed_prompts[: cfg.population]:
            acc, fb, n = self.minibatch_feedback(sp, train, cfg, rng)
            target_calls += n
            candidates.append(sp); population.append((sp, acc))
        while len(population) < cfg.population:
            out = self.proposal_llm.propose_initial(cfg, rng); prop_calls += 1
            c = mint_candidate(cfg.task_id, self.name, out.text, idx, cfg.seed,
                               proposal_call_ids=[out.call_id],
                               behavior_profile_id=out.profile_id); idx += 1
            acc, fb, n = self.minibatch_feedback(c, train, cfg, rng); target_calls += n
            candidates.append(c); population.append((c, acc))

        # Evolutionary generations.
        for gen in range(cfg.generations):
            population.sort(key=lambda t: t[1], reverse=True)
            elites = population[: max(2, cfg.population // 2)]
            children: list[tuple[CandidatePrompt, float]] = []
            for parent, p_acc in elites:
                p_prof = parent.metadata.get("behavior_profile_id")
                _, fb, n = self.minibatch_feedback(parent, train, cfg, rng); target_calls += n
                out = self.proposal_llm.reflect_mutate(cfg, parent.prompt_text, p_prof,
                                                       p_acc, fb, rng); refl_calls += 1
                child = mint_candidate(cfg.task_id, self.name, out.text, idx, cfg.seed,
                                       parent=parent, reflection_call_ids=[out.call_id],
                                       behavior_profile_id=out.profile_id,
                                       extra_meta={"generation": gen + 1}); idx += 1
                acc, _, n = self.minibatch_feedback(child, train, cfg, rng); target_calls += n
                child.metadata["minibatch_acc"] = acc
                candidates.append(child); children.append((child, acc))

            # one crossover per generation between the two best elites
            if len(elites) >= 2:
                (a, _), (b, _) = elites[0], elites[1]
                out = self.proposal_llm.crossover(
                    cfg, a.prompt_text, a.metadata.get("behavior_profile_id"),
                    b.prompt_text, b.metadata.get("behavior_profile_id"), rng); prop_calls += 1
                xo = mint_candidate(cfg.task_id, self.name, out.text, idx, cfg.seed,
                                    parent=a, proposal_call_ids=[out.call_id],
                                    behavior_profile_id=out.profile_id,
                                    extra_meta={"generation": gen + 1, "op": "crossover"}); idx += 1
                acc, _, n = self.minibatch_feedback(xo, train, cfg, rng); target_calls += n
                xo.metadata["minibatch_acc"] = acc
                candidates.append(xo); children.append((xo, acc))

            population = elites + children
            if len(candidates) >= cfg.pool_size:
                break

        return GenerationResult(
            self.name, cfg.task_id, subsample_spread(candidates, cfg.pool_size),
            proposal_calls=prop_calls, reflection_calls=refl_calls,
            target_calls=target_calls,
            metadata={"generations_run": gen + 1, "n_candidates": len(candidates)})
