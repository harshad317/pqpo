"""CAPO candidate generator (Sec 4.6).

CAPO = Cost-Aware Prompt Optimization (Zehle et al., 2025): an evolutionary
optimizer that (a) uses LLM mutation/crossover with few-shot examples, (b) adds a
LENGTH/cost penalty to the fitness, and (c) uses RACING (successive elimination on
growing minibatches) to discard weak candidates with few evaluations.

We emit every candidate produced — including ones eliminated by racing — so the
pool reflects CAPO's true proposal stream and its characteristic short, cost-aware
prompts. Racing is what makes CAPO cheap; we count exactly the target calls it
actually spends.
"""
from __future__ import annotations

import numpy as np

from ..data.datastructures import CandidatePrompt, TaskExample
from .base import (CandidateGenerator, GenerationConfig, GenerationResult,
                   mint_candidate, subsample_spread)

LENGTH_PENALTY = 0.0008   # fitness -= penalty * token_length (cost awareness)


class CAPOAdapter(CandidateGenerator):
    name = "CAPO"

    def _fitness(self, acc: float, prompt: CandidatePrompt) -> float:
        return acc - LENGTH_PENALTY * prompt.token_length

    def _race(self, members, train, cfg, rng):
        """Successive elimination: evaluate on growing minibatches, keep top half
        each round. Returns (survivors, accuracy_map, target_calls_spent)."""
        alive = list(members)
        acc = {m.prompt_id: [] for m in alive}
        spent = 0
        rounds = 3
        while len(alive) > 1 and rounds > 0:
            bs = max(2, cfg.minibatch // 2)
            idx = rng.choice(len(train), size=min(bs, len(train)), replace=False)
            batch = [train[i] for i in idx]
            for m in alive:
                for ex in batch:
                    acc[m.prompt_id].append(
                        self.scorer.score_one(m, ex, call_type="selector_dev"))
                    spent += 1
            alive.sort(key=lambda m: self._fitness(float(np.mean(acc[m.prompt_id])), m),
                       reverse=True)
            alive = alive[: max(1, len(alive) // 2)]
            rounds -= 1
        mean_acc = {pid: float(np.mean(v)) if v else 0.0 for pid, v in acc.items()}
        return alive, mean_acc, spent

    def generate(self, cfg: GenerationConfig, seed_prompts, train, rng) -> GenerationResult:
        candidates: list[CandidatePrompt] = []
        prop_calls = refl_calls = target_calls = 0
        idx = 0

        # few-shot demo bank (short, cost-aware -> at most 2 demos)
        demo_idx = rng.choice(len(train), size=min(2, len(train)), replace=False)
        demos = [train[i] for i in demo_idx]
        shots = "\n".join(f"In: {e.input_text[:24]} -> {e.target}" for e in demos)

        # initial population
        population = []
        for sp in seed_prompts[: cfg.population]:
            candidates.append(sp); population.append(sp)
        while len(population) < cfg.population:
            out = self.proposal_llm.propose_initial(cfg, rng); prop_calls += 1
            c = mint_candidate(cfg.task_id, self.name, out.text, idx, cfg.seed,
                               proposal_call_ids=[out.call_id],
                               behavior_profile_id=out.profile_id); idx += 1
            candidates.append(c); population.append(c)

        for gen in range(cfg.generations):
            # generate offspring: mutation (reflection) + crossover, with few-shot
            offspring = []
            for parent in population[: max(2, cfg.population // 2)]:
                _, fb, n = self.minibatch_feedback(parent, train, cfg, rng); target_calls += n
                out = self.proposal_llm.reflect_mutate(
                    cfg, parent.prompt_text, parent.metadata.get("behavior_profile_id"),
                    0.5, fb, rng); refl_calls += 1
                text = f"{out.text}\nFew-shot:\n{shots}"
                child = mint_candidate(cfg.task_id, self.name, text, idx, cfg.seed,
                                       parent=parent, reflection_call_ids=[out.call_id],
                                       behavior_profile_id=out.profile_id,
                                       extra_meta={"generation": gen + 1}); idx += 1
                candidates.append(child); offspring.append(child)
            if len(population) >= 2:
                a, b = population[0], population[1]
                out = self.proposal_llm.crossover(
                    cfg, a.prompt_text, a.metadata.get("behavior_profile_id"),
                    b.prompt_text, b.metadata.get("behavior_profile_id"), rng); prop_calls += 1
                xo = mint_candidate(cfg.task_id, self.name, out.text, idx, cfg.seed,
                                    parent=a, proposal_call_ids=[out.call_id],
                                    behavior_profile_id=out.profile_id,
                                    extra_meta={"generation": gen + 1, "op": "crossover"}); idx += 1
                candidates.append(xo); offspring.append(xo)

            # RACE the combined pool; survivors seed the next generation
            survivors, mean_acc, spent = self._race(population + offspring, train, cfg, rng)
            target_calls += spent
            for c in candidates:
                if c.prompt_id in mean_acc:
                    c.metadata["raced_acc"] = mean_acc[c.prompt_id]
            population = survivors + offspring[: cfg.population - len(survivors)]
            if len(candidates) >= cfg.pool_size:
                break

        return GenerationResult(
            self.name, cfg.task_id, subsample_spread(candidates, cfg.pool_size),
            proposal_calls=prop_calls, reflection_calls=refl_calls,
            target_calls=target_calls,
            metadata={"length_penalty": LENGTH_PENALTY, "generations_run": gen + 1})
