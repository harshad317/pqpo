"""Random and stratified minibatch selectors + score-only baseline (Sec 4.11)."""
from __future__ import annotations

import numpy as np

from .base import BudgetMeter, SelectorContext, SelectorResult


def _finish(scores, meter, method):
    best = max(scores, key=lambda p: np.mean(scores[p]) if scores[p] else -1)
    return SelectorResult(selected_prompt_id=best, labeled_evals_used=meter.used,
                          metadata={"method": method, "n_arms": len(scores)})


def run_random_minibatch(ctx: SelectorContext, minibatch: int = 8) -> SelectorResult:
    meter = BudgetMeter(ctx.budget)
    arms = [p.prompt_id for p in ctx.prompts]
    ctx.rng.shuffle(arms)
    scores = {}
    for a in arms:
        if meter.remaining <= 0:
            break
        exs = ctx.sample_dev(min(minibatch, meter.remaining))
        scores[a] = ctx.eval_prompt(a, exs, meter)
    return _finish(scores, meter, "random_minibatch")


def run_stratified_minibatch(ctx: SelectorContext, minibatch: int = 8) -> SelectorResult:
    """Stratify arms by source_method, round-robin across strata."""
    meter = BudgetMeter(ctx.budget)
    strata: dict[str, list[str]] = {}
    for p in ctx.prompts:
        strata.setdefault(p.source_method, []).append(p.prompt_id)
    for s in strata:
        ctx.rng.shuffle(strata[s])
    order, cursors = list(strata.keys()), {s: 0 for s in strata}
    scores = {}
    i = 0
    while meter.remaining > 0 and any(cursors[s] < len(strata[s]) for s in order):
        s = order[i % len(order)]
        i += 1
        if cursors[s] >= len(strata[s]):
            continue
        a = strata[s][cursors[s]]; cursors[s] += 1
        exs = ctx.sample_dev(min(minibatch, meter.remaining))
        scores[a] = ctx.eval_prompt(a, exs, meter)
    return _finish(scores, meter, "stratified_minibatch")


def run_score_only(ctx: SelectorContext, minibatch: int = 4) -> SelectorResult:
    """Naive score-only: evaluate as many prompts as budget allows, pick best."""
    return run_random_minibatch(ctx, minibatch=minibatch)
