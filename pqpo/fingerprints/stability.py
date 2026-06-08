"""Cluster stability under sentinel bootstrap (Sec 2.4 / 5.6)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..data.datastructures import BehaviorFingerprint
from .distances import pairwise_distance_matrix, restrict_fingerprint


def variation_of_information(a: list[int], b: list[int]) -> float:
    """VI(a,b) = H(a|b) + H(b|a). Lower is more similar."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    def ent_cond(x, y):
        h = 0.0
        for xv in np.unique(x):
            for yv in np.unique(y):
                pxy = np.mean((x == xv) & (y == yv))
                py = np.mean(y == yv)
                if pxy > 0 and py > 0:
                    h -= pxy * np.log(pxy / py)
        return h
    return ent_cond(a, b) + ent_cond(b, a)


def bootstrap_clusterings(fingerprints, cluster_fn, n_boot: int, rng) -> list[list[int]]:
    """Recompute clusterings on bootstrapped sentinel subsets.

    cluster_fn(distance_matrix) -> labels (list[int]) aligned to sorted ids.
    """
    ids = sorted(fingerprints.keys())
    k = len(next(iter(fingerprints.values())).normalized_answers)
    clusterings = []
    for _ in range(n_boot):
        # sample sentinel indices with replacement, dedup-preserving order
        idx = sorted(set(rng.integers(0, k, size=k).tolist()))
        if len(idx) < 2:
            idx = list(range(k))
        fp_b = {i: restrict_fingerprint(fingerprints[i], idx) for i in ids}
        D_b, order = pairwise_distance_matrix(fp_b, order=ids)
        clusterings.append(list(cluster_fn(D_b)))
    return clusterings


def mean_pairwise_metric(clusterings: list[list[int]], metric: str = "ari") -> float:
    import warnings
    fn = {"ari": adjusted_rand_score, "nmi": normalized_mutual_info_score}[metric]
    vals = []
    with warnings.catch_warnings():
        # ARI/NMI warn when #clusters is high vs #samples (many singleton cells);
        # that's a valid clustering here, not a misuse — silence the noise.
        warnings.simplefilter("ignore")
        for i in range(len(clusterings)):
            for j in range(i + 1, len(clusterings)):
                vals.append(fn(clusterings[i], clusterings[j]))
    return float(np.mean(vals)) if vals else 1.0
