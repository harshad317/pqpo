"""Multi-fidelity bandit baselines: successive halving, Hyperband, ASHA (Sec 4.9).

All are score-only (no behavioural quotient). They allocate the labelled budget
across prompt "arms" with increasing fidelity. ASHA here is a synchronous
rung-promotion approximation suitable for a fixed offline harness; the protocol
notes this approximation in metadata.
"""
from __future__ import annotations

import numpy as np

from .base import BudgetMeter, SelectorContext, SelectorResult


def _arm_scores_to_result(arm_scores, meter, method) -> SelectorResult:
    best = max(arm_scores, key=lambda p: np.mean(arm_scores[p]) if arm_scores[p] else -1)
    return SelectorResult(selected_prompt_id=best, labeled_evals_used=meter.used,
                          metadata={"method": method})


def run_successive_halving(ctx: SelectorContext, eta: int = 2,
                           r0: int = 1) -> SelectorResult:
    meter = BudgetMeter(ctx.budget)
    arms = [p.prompt_id for p in ctx.prompts]
    ctx.rng.shuffle(arms)
    scores: dict[str, list[float]] = {a: [] for a in arms}
    rung_evals = r0
    survivors = arms
    while len(survivors) > 1 and meter.remaining > 0:
        for a in list(survivors):
            if meter.remaining <= 0:
                break
            exs = ctx.sample_dev(min(rung_evals, meter.remaining))
            scores[a].extend(ctx.eval_prompt(a, exs, meter))
        survivors.sort(key=lambda a: np.mean(scores[a]) if scores[a] else -1, reverse=True)
        keep = max(1, len(survivors) // eta)
        survivors = survivors[:keep]
        rung_evals *= eta
    return _arm_scores_to_result(scores, meter, "successive_halving")


def run_hyperband(ctx: SelectorContext, eta: int = 3, max_r: int = 9) -> SelectorResult:
    """Run several SH brackets with different starting fidelities, share budget."""
    meter = BudgetMeter(ctx.budget)
    all_arms = [p.prompt_id for p in ctx.prompts]
    scores: dict[str, list[float]] = {a: [] for a in all_arms}
    s_max = int(np.log(max_r) / np.log(eta))
    brackets = list(range(s_max, -1, -1))
    per_bracket = max(1, ctx.budget // max(1, len(brackets)))
    for s in brackets:
        if meter.remaining <= 0:
            break
        n = int(np.ceil((s_max + 1) / (s + 1) * eta ** s))
        arms = list(ctx.rng.choice(all_arms, size=min(n, len(all_arms)), replace=False))
        rung_evals = max(1, int(max_r / eta ** s))
        bracket_meter_cap = meter.used + per_bracket
        survivors = arms
        while len(survivors) > 1 and meter.remaining > 0 and meter.used < bracket_meter_cap:
            for a in list(survivors):
                if meter.remaining <= 0 or meter.used >= bracket_meter_cap:
                    break
                exs = ctx.sample_dev(min(rung_evals, meter.remaining))
                scores[a].extend(ctx.eval_prompt(a, exs, meter))
            survivors.sort(key=lambda a: np.mean(scores[a]) if scores[a] else -1, reverse=True)
            survivors = survivors[:max(1, len(survivors) // eta)]
            rung_evals *= eta
    return _arm_scores_to_result(scores, meter, "hyperband")


def run_asha(ctx: SelectorContext, eta: int = 2, grace: int = 1,
             max_rungs: int = 5) -> SelectorResult:
    """Synchronous ASHA approximation: promote top-1/eta of each rung once enough
    arms have been evaluated at that rung."""
    meter = BudgetMeter(ctx.budget)
    arms = [p.prompt_id for p in ctx.prompts]
    ctx.rng.shuffle(arms)
    rung_of: dict[str, int] = {a: 0 for a in arms}
    scores: dict[str, list[float]] = {a: [] for a in arms}
    promoted: set[str] = set()
    while meter.remaining > 0:
        # pick the lowest-rung arm with fewest evals (async-ish scheduling)
        candidate = min(arms, key=lambda a: (rung_of[a], len(scores[a])))
        rung = rung_of[candidate]
        evals = grace * (eta ** rung)
        exs = ctx.sample_dev(min(evals, meter.remaining))
        scores[candidate].extend(ctx.eval_prompt(candidate, exs, meter))
        # promotion check within this rung
        same_rung = [a for a in arms if rung_of[a] == rung and scores[a]]
        if len(same_rung) >= eta and rung < max_rungs:
            top = max(same_rung, key=lambda a: np.mean(scores[a]))
            if top not in promoted:
                rung_of[top] = rung + 1
                promoted.add(top)
        if all(rung_of[a] >= max_rungs for a in arms):
            break
    return _arm_scores_to_result(scores, meter, "asha")
