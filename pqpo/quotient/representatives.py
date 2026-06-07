"""Phenotype-cell representative selection (Sec 2.5).

Default: medoid (behaviourally central). Ablation variants supported for the
representative-choice ablation in Sec 6.
"""
from __future__ import annotations

import numpy as np


def choose_medoid(prompt_ids: list[str], D: np.ndarray, index_of: dict[str, int]) -> str:
    """argmin_p mean_q d(p,q) within the cell."""
    if len(prompt_ids) == 1:
        return prompt_ids[0]
    idxs = [index_of[p] for p in prompt_ids]
    sub = D[np.ix_(idxs, idxs)]
    mean_d = sub.mean(axis=1)
    return prompt_ids[int(np.argmin(mean_d))]


def choose_representative(prompt_ids, D, index_of, rule="medoid",
                          prompt_lengths=None, first_stage_scores=None,
                          format_validity=None, rng=None) -> str:
    if rule == "medoid":
        return choose_medoid(prompt_ids, D, index_of)
    if rule == "shortest" and prompt_lengths:
        return min(prompt_ids, key=lambda p: prompt_lengths.get(p, 1e9))
    if rule == "random":
        return prompt_ids[int(rng.integers(0, len(prompt_ids)))]
    if rule == "highest_format" and format_validity:
        return max(prompt_ids, key=lambda p: format_validity.get(p, -1))
    if rule == "highest_first_stage_score" and first_stage_scores:
        return max(prompt_ids, key=lambda p: first_stage_scores.get(p, -1))
    return choose_medoid(prompt_ids, D, index_of)
