"""Fixed weighted mixed fingerprint distance (Sec 2.3).

    d = 0.55 d_answer + 0.15 d_format + 0.10 d_refusal
        + 0.10 d_shape  + 0.05 d_tokens + 0.05 d_latency

Weights are FIXED for the main MVP and must not be tuned on labels. Weight-tuned
variants may appear only as ablations / oracle upper bounds.
"""
from __future__ import annotations

import numpy as np

from ..data.datastructures import BehaviorFingerprint

DEFAULT_WEIGHTS = {
    "answer": 0.55, "format": 0.15, "refusal": 0.10,
    "shape": 0.10, "tokens": 0.05, "latency": 0.05,
}


def _mean_mismatch(a: list, b: list) -> float:
    return float(np.mean([x != y for x, y in zip(a, b)])) if a else 0.0


def _norm_abs_bucket_diff(a: list[int], b: list[int], n_buckets: int = 7) -> float:
    if not a:
        return 0.0
    denom = max(1, n_buckets - 1)
    return float(np.mean([abs(x - y) / denom for x, y in zip(a, b)]))


def fingerprint_distance(p: BehaviorFingerprint, q: BehaviorFingerprint,
                         weights: dict = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    d_answer = _mean_mismatch(p.normalized_answers, q.normalized_answers)
    d_format = _mean_mismatch(p.format_validity, q.format_validity)
    d_refusal = _mean_mismatch(p.refusal_flags, q.refusal_flags)
    # shape = parse-error-type OR length-bucket differs
    shape_p = list(zip(p.parse_error_types, p.output_length_buckets))
    shape_q = list(zip(q.parse_error_types, q.output_length_buckets))
    d_shape = _mean_mismatch(shape_p, shape_q)
    d_tokens = _norm_abs_bucket_diff(p.output_length_buckets, q.output_length_buckets)
    d_latency = _norm_abs_bucket_diff(p.latency_buckets, q.latency_buckets)
    return (
        w["answer"] * d_answer + w["format"] * d_format + w["refusal"] * d_refusal
        + w["shape"] * d_shape + w["tokens"] * d_tokens + w["latency"] * d_latency
    )


def pairwise_distance_matrix(fingerprints: dict[str, BehaviorFingerprint],
                             order: list[str] = None,
                             weights: dict = None) -> tuple[np.ndarray, list[str]]:
    ids = order or list(fingerprints.keys())
    n = len(ids)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = fingerprint_distance(fingerprints[ids[i]], fingerprints[ids[j]], weights)
            D[i, j] = D[j, i] = d
    return D, ids


def restrict_fingerprint(fp: BehaviorFingerprint, idx: list[int]) -> BehaviorFingerprint:
    """Restrict a fingerprint to a subset of sentinel indices (for bootstrap)."""
    def take(seq):
        return [seq[i] for i in idx] if seq is not None else None
    return BehaviorFingerprint(
        prompt_id=fp.prompt_id,
        task_id=fp.task_id,
        normalized_answers=take(fp.normalized_answers),
        format_validity=take(fp.format_validity),
        refusal_flags=take(fp.refusal_flags),
        parse_error_types=take(fp.parse_error_types),
        output_length_buckets=take(fp.output_length_buckets),
        token_counts=take(fp.token_counts),
        latency_buckets=take(fp.latency_buckets),
    )
