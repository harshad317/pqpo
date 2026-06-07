"""Quotient controls (Sec 4.12 / 6.3).

Each control produces a *different* partition of the pool, then runs the SAME
cluster-then-select allocator as PQPO. If any control matches PQPO, the
behavioural quotient adds nothing. These are the decisive negative controls.

Controls:
  * lexical_cluster   - KMeans over lexical (hashed char-ngram) vectors
  * embedding_cluster - KMeans over prompt embeddings
  * score_bins        - bin by a cheap first-stage score
  * random_quotient   - random labels with cluster sizes matched to PQPO
  * map_elites        - structural descriptors (length bucket x keyword presence)
  * semantic_cache    - dedup near-duplicates, then score-only over survivors
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .base import BudgetMeter, SelectorContext, SelectorResult
from .cluster_select import cells_from_labels, run_cluster_select
from .minibatch import run_random_minibatch


def _kmeans_labels(vectors: dict[str, np.ndarray], ids: list[str],
                   n_clusters: int, seed: int) -> dict[str, int]:
    X = np.stack([vectors[i] for i in ids])
    k = max(1, min(n_clusters, len(ids)))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    labs = km.fit_predict(X)
    return {ids[i]: int(labs[i]) for i in range(len(ids))}


def run_lexical_cluster(ctx: SelectorContext, n_clusters: int,
                        lexical_vectors: dict) -> SelectorResult:
    labels = _kmeans_labels(lexical_vectors, ctx.ids, n_clusters, int(ctx.rng.integers(1e6)))
    cells = cells_from_labels(labels, ctx)
    return run_cluster_select(ctx, cells, "lexical_cluster")


def run_embedding_cluster(ctx: SelectorContext, n_clusters: int,
                          embeddings: dict) -> SelectorResult:
    labels = _kmeans_labels(embeddings, ctx.ids, n_clusters, int(ctx.rng.integers(1e6)))
    cells = cells_from_labels(labels, ctx)
    return run_cluster_select(ctx, cells, "embedding_cluster")


def run_score_bins(ctx: SelectorContext, n_bins: int = 8) -> SelectorResult:
    """Cheap first-stage score (1 eval/prompt) -> bin -> cluster-select.

    First-stage evals are charged to the budget (honest accounting)."""
    # Reserve at least half the budget for the adaptive allocator; spend the rest
    # on cheap first-stage (1 eval/prompt) screening.
    first_stage_cap = max(1, ctx.budget // 2)
    meter = BudgetMeter(first_stage_cap)
    first = {}
    for p in ctx.prompts:
        if meter.remaining <= 0:
            first[p.prompt_id] = 0.0
            continue
        s = ctx.eval_prompt(p.prompt_id, ctx.sample_dev(1), meter)
        first[p.prompt_id] = s[0] if s else 0.0
    vals = np.array([first[p.prompt_id] for p in ctx.prompts])
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1]) if len(set(vals)) > 1 else []
    labels = {pid: int(np.digitize(first[pid], edges)) for pid in first}
    cells = cells_from_labels(labels, ctx)
    # Continue allocating with the *remaining* budget through the shared allocator.
    remaining = ctx.budget - meter.used
    ctx_remaining = SelectorContext(
        prompts=ctx.prompts, dev_examples=ctx.dev_examples, scorer=ctx.scorer,
        rng=ctx.rng, budget=remaining, fingerprints=ctx.fingerprints,
        D=ctx.D, ids=ctx.ids, source_methods=ctx.source_methods, weights=ctx.weights)
    res = run_cluster_select(ctx_remaining, cells, "score_bins")
    res.labeled_evals_used += meter.used
    return res


def run_random_quotient(ctx: SelectorContext, cluster_sizes: list[int]) -> SelectorResult:
    """Random labels with cluster sizes matched to PQPO (Sec 4.12)."""
    ids = list(ctx.ids)
    ctx.rng.shuffle(ids)
    labels, cursor = {}, 0
    for cid, size in enumerate(cluster_sizes):
        for _ in range(size):
            if cursor >= len(ids):
                break
            labels[ids[cursor]] = cid
            cursor += 1
    for pid in ids[cursor:]:
        labels[pid] = len(cluster_sizes) - 1
    cells = cells_from_labels(labels, ctx)
    return run_cluster_select(ctx, cells, "random_quotient")


def run_map_elites(ctx: SelectorContext) -> SelectorResult:
    """Structural-descriptor MAP-Elites control (Sec 4.8).

    Descriptor = (token-length bucket, has-'json' keyword, has-'step' keyword).
    Same allocator; the only difference from PQPO is the descriptor space."""
    def descriptor(p):
        tl = min(4, p.token_length // 8)
        kw1 = int("json" in p.prompt_text.lower())
        kw2 = int("step" in p.prompt_text.lower() or "reason" in p.prompt_text.lower())
        return (tl, kw1, kw2)
    cell_index, labels = {}, {}
    for p in ctx.prompts:
        d = descriptor(p)
        labels[p.prompt_id] = cell_index.setdefault(d, len(cell_index))
    cells = cells_from_labels(labels, ctx)
    return run_cluster_select(ctx, cells, "map_elites")


def run_semantic_cache(ctx: SelectorContext, embeddings: dict,
                       dup_threshold: float = 0.98) -> SelectorResult:
    """Dedup near-duplicate prompts by embedding cosine, then score-only.

    This is the "semantic caching" strawman: it removes lexical/embedding dupes
    but cannot see behavioural equivalence across dissimilar strings."""
    ids = list(ctx.ids)
    kept, kept_vecs = [], []
    for pid in ids:
        v = embeddings[pid]
        is_dup = any(float(np.dot(v, kv)) >= dup_threshold for kv in kept_vecs)
        if not is_dup:
            kept.append(pid); kept_vecs.append(v)
    survivors = [ctx.prompt_by_id[p] for p in kept]
    sub = SelectorContext(
        prompts=survivors, dev_examples=ctx.dev_examples, scorer=ctx.scorer,
        rng=ctx.rng, budget=ctx.budget, ids=kept,
        source_methods=ctx.source_methods, weights=ctx.weights)
    res = run_random_minibatch(sub, minibatch=8)
    res.metadata["method"] = "semantic_cache"
    res.metadata["n_survivors"] = len(kept)
    return res
