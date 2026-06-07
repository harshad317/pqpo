"""Generic cluster-then-select allocator.

This is the *shared* machinery used by PQPO and by every clustering control
(lexical, embedding, score-bin, random-quotient, MAP-Elites). The ONLY thing
that differs between them is the partition of prompts into cells. Holding the
allocator fixed is what makes the controls fair: if a control matches PQPO, the
phenotype quotient is doing nothing; if PQPO wins, the behavioural quotient is
the cause (Sec 4.12 / 6.3).
"""
from __future__ import annotations

import numpy as np

from ..data.datastructures import CellState, PhenotypeCell
from ..quotient.representatives import choose_medoid
from .base import BudgetMeter, SelectorContext, SelectorResult
from .pqpo_fixed_pool import (_best_prompt_in_cell, _choose_prompt_within_cell,
                              _lcb, compute_cell_utility)


def cells_from_labels(labels: dict[str, int], ctx: SelectorContext,
                      use_medoid: bool = True) -> list[PhenotypeCell]:
    groups: dict[int, list[str]] = {}
    for pid, lab in labels.items():
        groups.setdefault(lab, []).append(pid)
    cells = []
    for lab, members in sorted(groups.items()):
        if use_medoid and ctx.D is not None:
            rep = choose_medoid(members, ctx.D, ctx.index_of)
        else:
            rep = members[0]
        cells.append(PhenotypeCell(
            cell_id=f"cell_{lab:04d}", task_id=ctx.prompts[0].task_id,
            prompt_ids=members, representative_prompt_id=rep, medoid_prompt_id=rep,
            size=len(members),
        ))
    return cells


def run_cluster_select(ctx: SelectorContext, cells: list[PhenotypeCell],
                       method_name: str = "cluster_select") -> SelectorResult:
    meter = BudgetMeter(ctx.budget)
    cell_by_id = {c.cell_id: c for c in cells}
    max_size = max(c.size for c in cells)

    states: dict[str, CellState] = {}
    for c in cells:
        st = CellState(c.cell_id, list(c.prompt_ids), c.representative_prompt_id)
        st.novelty_bonus = c.intra_cell_distance_mean
        st.redundancy_penalty = max(0.0, (c.size - 1) / max_size)
        states[c.cell_id] = st

    batch_size = max(1, min(8, ctx.budget // max(1, len(cells))))
    for c in cells:
        if meter.remaining <= 0:
            break
        exs = ctx.sample_dev(min(batch_size, meter.remaining))
        for s in ctx.eval_prompt(c.representative_prompt_id, exs, meter):
            states[c.cell_id].record(c.representative_prompt_id, s)

    while meter.remaining > 0:
        utilities = {cid: compute_cell_utility(st, max_size) for cid, st in states.items()}
        chosen = max(utilities, key=utilities.get)
        prompt_id = _choose_prompt_within_cell(cell_by_id[chosen], states[chosen], ctx)
        for s in ctx.eval_prompt(prompt_id, ctx.sample_dev(1), meter):
            states[chosen].record(prompt_id, s)

    evaluated = {cid: st for cid, st in states.items() if st.n_labeled_evals > 0}
    if not evaluated:
        # No budget was available to evaluate anything: fall back to the medoid of
        # the largest cell (best zero-eval guess).
        biggest = max(cells, key=lambda c: c.size)
        return SelectorResult(selected_prompt_id=biggest.representative_prompt_id,
                              selected_cell_id=biggest.cell_id,
                              labeled_evals_used=meter.used,
                              metadata={"method": method_name, "n_cells": len(cells),
                                        "fallback": "no_budget"})
    final_cell_id = max(evaluated, key=lambda c: _lcb(evaluated[c]))
    final_prompt = _best_prompt_in_cell(cell_by_id[final_cell_id], states[final_cell_id])
    return SelectorResult(
        selected_prompt_id=final_prompt, selected_cell_id=final_cell_id,
        labeled_evals_used=meter.used,
        metadata={"method": method_name, "n_cells": len(cells)},
    )
