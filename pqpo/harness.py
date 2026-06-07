"""Shared selector harness: the full method matrix + source-method baselines.

Canonical home for the selector registry so both the legacy scripts and the
unified ``benchmark.run`` entry point use exactly the same method set and
semantics. Modality-agnostic: everything operates over a SelectorContext, so the
same harness runs label / code / instruction-following tasks.
"""
from __future__ import annotations

import numpy as np

from .selectors import (bo_embedding, clustering_controls, minibatch,
                        multifidelity, pqpo_fixed_pool)
from .selectors.base import BudgetMeter, SelectorContext, SelectorResult

# Selector/control catalog (order = display order).
METHOD_NAMES = [
    "pqpo", "score_only", "random_minibatch", "stratified_minibatch",
    "successive_halving", "hyperband", "asha", "bo_embedding",
    "lexical_cluster", "embedding_cluster", "score_bins", "random_quotient",
    "map_elites", "semantic_cache",
]
# Generator / source-method baselines (also build candidate pools).
GENERATOR_METHOD_NAMES = ["GEPA", "MIPROv2", "CAPO", "APE"]


def selector_registry(ctx_factory, n_cells, cluster_sizes, embeddings, lexical_vecs,
                      tau=None):
    """method name -> zero-arg callable producing its SelectorResult."""
    return {
        "pqpo": lambda: pqpo_fixed_pool.run_pqpo_fixed_pool(ctx_factory(), tau=tau),
        "score_only": lambda: minibatch.run_score_only(ctx_factory()),
        "random_minibatch": lambda: minibatch.run_random_minibatch(ctx_factory()),
        "stratified_minibatch": lambda: minibatch.run_stratified_minibatch(ctx_factory()),
        "successive_halving": lambda: multifidelity.run_successive_halving(ctx_factory()),
        "hyperband": lambda: multifidelity.run_hyperband(ctx_factory()),
        "asha": lambda: multifidelity.run_asha(ctx_factory()),
        "bo_embedding": lambda: bo_embedding.run_bo_embedding(ctx_factory(), embeddings),
        "lexical_cluster": lambda: clustering_controls.run_lexical_cluster(
            ctx_factory(), n_cells, lexical_vecs),
        "embedding_cluster": lambda: clustering_controls.run_embedding_cluster(
            ctx_factory(), n_cells, embeddings),
        "score_bins": lambda: clustering_controls.run_score_bins(ctx_factory()),
        "random_quotient": lambda: clustering_controls.run_random_quotient(
            ctx_factory(), cluster_sizes),
        "map_elites": lambda: clustering_controls.run_map_elites(ctx_factory()),
        "semantic_cache": lambda: clustering_controls.run_semantic_cache(
            ctx_factory(), embeddings),
    }


def run_all_selectors(ctx_factory, n_cells, cluster_sizes, embeddings, lexical_vecs,
                      tau=None, methods=None):
    """Run the selected methods (default: all). Only chosen methods execute."""
    registry = selector_registry(ctx_factory, n_cells, cluster_sizes,
                                 embeddings, lexical_vecs, tau)
    chosen = [m for m in (methods or registry.keys()) if m in registry]
    return {m: registry[m]() for m in chosen}


# Internal validation-score keys the generators record during generation, in
# priority order (the prompt a generator would actually output is its best by its
# own internal feedback — already paid for during generation).
INTERNAL_SCORE_KEYS = ["raced_acc", "minibatch_acc", "internal_score",
                       "first_stage_score", "demo_quality"]


def run_source_method_baseline(ctx: SelectorContext, source_method: str):
    """The generator's *own converged output*: the candidate from this lineage
    with the best internal validation score it recorded during generation.

    This is the realistic baseline — what GEPA/MIPROv2/CAPO/APE would emit — and
    consumes no extra labelled budget (the internal selection cost is part of the
    generation cost, already counted). It avoids the unfair 'exhaustively re-search
    the lineage under the shared budget' semantics, which is near-oracle when a
    lineage happens to contain a top prompt."""
    arms = [p for p in ctx.prompts if p.source_method == source_method]
    if not arms:
        return None

    def internal_score(p):
        for k in INTERNAL_SCORE_KEYS:
            if k in p.metadata:
                return float(p.metadata[k])
        return None

    scored = [(p, internal_score(p)) for p in arms]
    have = [(p, s) for p, s in scored if s is not None]
    if have:
        best = max(have, key=lambda t: t[1])[0]
        rule = "generator_internal_best"
    else:
        # No internal score recorded (generator did no validation selection):
        # fall back to its first proposal (an honest, weak no-selection baseline).
        best = arms[0]
        rule = "generator_no_internal_selection"
    return SelectorResult(selected_prompt_id=best.prompt_id, labeled_evals_used=0,
                          metadata={"method": source_method, "n_arms": len(arms),
                                    "baseline": rule})


def run_source_method_baselines(ctx_factory, source_methods):
    out = {}
    for sm in source_methods:
        res = run_source_method_baseline(ctx_factory(), sm)
        if res is not None:
            out[sm] = res
    return out
