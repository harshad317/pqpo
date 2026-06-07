"""Held-out evaluation + quotient / mechanistic metrics (Sec 5.3)."""
from __future__ import annotations

import numpy as np

from ..data.datastructures import CandidatePrompt, TaskExample
from ..fingerprints.distances import fingerprint_distance
from .scorers import Scorer


def held_out_scores(prompt: CandidatePrompt, test_examples: list[TaskExample],
                    scorer: Scorer) -> np.ndarray:
    """Per-example 0/1 scores on the final held-out test set (call_type=final_test)."""
    return np.array([scorer.score_one(prompt, ex, call_type="final_test")
                     for ex in test_examples])


def behavioral_redundancy_rate(cells, source_methods: dict) -> float:
    """Fraction of prompts that share a cell with a prompt from a *different*
    generator lineage (Sec breakthrough condition #1)."""
    n_redundant, n_total = 0, 0
    for c in cells:
        srcs = [source_methods.get(p, "unknown") for p in c.prompt_ids]
        for i, p in enumerate(c.prompt_ids):
            n_total += 1
            if any(srcs[j] != srcs[i] for j in range(len(srcs)) if j != i):
                n_redundant += 1
    return n_redundant / n_total if n_total else 0.0


def distance_transfer_table(prompt_ids, fingerprints, embeddings, prompt_texts,
                            heldout_score: dict, weights=None,
                            max_pairs: int = 4000, rng=None) -> dict:
    """For random prompt pairs, correlate |score(p)-score(q)| with phenotype
    distance, embedding distance, edit distance and length diff (Sec 5.6)."""
    rng = rng or np.random.default_rng(0)
    ids = [p for p in prompt_ids if p in heldout_score]
    pairs = []
    for _ in range(min(max_pairs, len(ids) * (len(ids) - 1) // 2)):
        i, j = rng.integers(0, len(ids), 2)
        if i != j:
            pairs.append((ids[i], ids[j]))
    pheno, emb, edit, length, dscore = [], [], [], [], []
    for a, b in pairs:
        dscore.append(abs(heldout_score[a] - heldout_score[b]))
        pheno.append(fingerprint_distance(fingerprints[a], fingerprints[b], weights))
        va, vb = embeddings[a], embeddings[b]
        emb.append(1.0 - float(np.dot(va, vb)))
        edit.append(_norm_edit(prompt_texts[a], prompt_texts[b]))
        length.append(abs(len(prompt_texts[a]) - len(prompt_texts[b])))
    return {"phenotype": (pheno, dscore), "embedding": (emb, dscore),
            "edit": (edit, dscore), "length": (length, dscore)}


def _norm_edit(a: str, b: str) -> float:
    # token-level Jaccard distance as a cheap edit-distance proxy
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)
