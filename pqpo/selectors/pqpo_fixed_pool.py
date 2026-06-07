"""Fixed-pool PQPO selector (Sec 2.5).

Three phases:
  1. Cluster the pool into phenotype cells; pick a medoid representative per cell.
  2. Evaluate one mini-batch per representative; then adaptively allocate the
     remaining labelled budget by cell utility.
  3. Select the final cell (LCB/mean) and the best confirmed prompt in it.

Cell utility (Sec 2.5):
  U(C) = mean_score + beta*uncertainty + lambda*novelty
         - rho*redundancy_penalty - gamma*normalized_cost
"""
from __future__ import annotations

import math

import numpy as np

from ..data.datastructures import CellState, PhenotypeCell
from ..quotient.clusterer import QuotientClusterer
from ..quotient.representatives import choose_medoid
from .base import BudgetMeter, SelectorContext, SelectorResult

BETA, LAMBDA, RHO, GAMMA = 0.5, 0.1, 0.1, 0.05


def _uncertainty(state: CellState) -> float:
    """Empirical-Bernstein-style bonus: shrinks with more evals."""
    n = max(1, state.n_labeled_evals)
    var = state.score_variance
    return math.sqrt(2.0 * var * math.log(1.0 + n) / n) + 3.0 * math.log(1.0 + n) / n


def compute_cell_utility(state: CellState, max_size: int) -> float:
    novelty = state.novelty_bonus
    redundancy = state.redundancy_penalty
    norm_cost = len(state.prompt_ids) / max(1, max_size)
    return (state.mean_score + BETA * _uncertainty(state)
            + LAMBDA * novelty - RHO * redundancy - GAMMA * norm_cost)


def _lcb(state: CellState) -> float:
    n = max(1, state.n_labeled_evals)
    return state.mean_score - math.sqrt(state.score_variance * math.log(1 + n) / n)


def run_pqpo_fixed_pool(ctx: SelectorContext, clusterer: QuotientClusterer = None,
                        tau: float = None, n_boot: int = 40) -> SelectorResult:
    clusterer = clusterer or QuotientClusterer(weights=ctx.weights)
    meter = BudgetMeter(ctx.budget)

    # 1. Cluster into phenotype cells.
    if tau is None:
        tau, tau_diag = clusterer.choose_tau_by_stability(
            ctx.fingerprints, n_boot=n_boot, rng=ctx.rng)
    else:
        tau_diag = []
    cells = clusterer.cluster(ctx.D, ctx.ids, tau, ctx.fingerprints, ctx.source_methods)
    cell_by_id = {c.cell_id: c for c in cells}
    max_size = max(c.size for c in cells)

    # novelty bonus: larger for cells far (in fingerprint) from the pool centroid.
    states: dict[str, CellState] = {}
    for c in cells:
        st = CellState(c.cell_id, list(c.prompt_ids), c.representative_prompt_id)
        st.novelty_bonus = c.intra_cell_distance_mean
        st.redundancy_penalty = max(0.0, (c.size - 1) / max_size)
        states[c.cell_id] = st

    # 2. Initial allocation: one mini-batch per representative.
    batch_size = max(1, min(8, ctx.budget // max(1, len(cells))))
    for c in cells:
        if meter.remaining <= 0:
            break
        exs = ctx.sample_dev(min(batch_size, meter.remaining))
        scores = ctx.eval_prompt(c.representative_prompt_id, exs, meter)
        for s in scores:
            states[c.cell_id].record(c.representative_prompt_id, s)

    # 3. Adaptive allocation by utility.
    while meter.remaining > 0:
        utilities = {cid: compute_cell_utility(st, max_size) for cid, st in states.items()}
        chosen = max(utilities, key=utilities.get)
        cell = cell_by_id[chosen]
        prompt_id = _choose_prompt_within_cell(cell, states[chosen], ctx)
        ex = ctx.sample_dev(1)
        scores = ctx.eval_prompt(prompt_id, ex, meter)
        for s in scores:
            states[chosen].record(prompt_id, s)

    # 4. Final selection: cell by LCB, then best confirmed prompt (or medoid).
    evaluated = {cid: st for cid, st in states.items() if st.n_labeled_evals > 0}
    final_cell_id = max(evaluated, key=lambda c: _lcb(evaluated[c]))
    final_state = states[final_cell_id]
    final_prompt = _best_prompt_in_cell(cell_by_id[final_cell_id], final_state)

    return SelectorResult(
        selected_prompt_id=final_prompt,
        selected_cell_id=final_cell_id,
        labeled_evals_used=meter.used,
        metadata={
            "tau": tau, "n_cells": len(cells),
            "compression": 1.0 - len(cells) / len(ctx.ids),
            "max_cell_frac": max_size / len(ctx.ids),
            "tau_diagnostics": tau_diag,
        },
    )


def _choose_prompt_within_cell(cell: PhenotypeCell, state: CellState, ctx) -> str:
    """Mostly confirm the representative; occasionally probe an unconfirmed member
    to guard against medoid being unrepresentative of held-out score."""
    unconfirmed = [p for p in cell.prompt_ids if p not in state.per_prompt_scores]
    if unconfirmed and ctx.rng.random() < 0.3:
        return unconfirmed[int(ctx.rng.integers(0, len(unconfirmed)))]
    return state.representative_prompt_id


def _best_prompt_in_cell(cell: PhenotypeCell, state: CellState) -> str:
    best, best_m = None, -np.inf
    for pid, scs in state.per_prompt_scores.items():
        m = float(np.mean(scs))
        if m > best_m:
            best, best_m = pid, m
    return best or cell.medoid_prompt_id
