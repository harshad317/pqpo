"""QuotientClusterer (Sec 2.4 / 3.7).

Agglomerative clustering over the precomputed fingerprint distance matrix with a
stability-selected threshold tau. tau is chosen by sentinel-bootstrap stability
ONLY (never on labels).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..data.datastructures import BehaviorFingerprint, PhenotypeCell
from ..fingerprints.distances import pairwise_distance_matrix
from ..fingerprints.stability import bootstrap_clusterings, mean_pairwise_metric
from .representatives import choose_medoid

DEFAULT_TAU_GRID = [round(0.05 * i, 2) for i in range(1, 13)]  # 0.05 .. 0.60


def agglomerative_labels(D: np.ndarray, tau: float) -> np.ndarray:
    if D.shape[0] == 1:
        return np.array([0])
    model = AgglomerativeClustering(
        metric="precomputed", linkage="average",
        distance_threshold=tau, n_clusters=None,
    )
    return model.fit_predict(D)


class QuotientClusterer:
    """tau is accepted when bootstrap ARI >= ari_threshold AND the clustering is
    *informative*: cell count within a pool-size-aware band (>= min_cells and
    <= cell_frac_hi * n, i.e. >= ~1/cell_frac_hi prompts per cell on average)
    and no single giant cell (max_cluster_frac). A fixed compression range is
    deliberately NOT used: it implicitly assumes one pool size (0.20-0.60 was
    tuned for 240-prompt pools and rejects healthy 16-20-cell quotients on
    80-prompt pools)."""

    def __init__(self, ari_threshold: float = 0.65, weights: dict = None,
                 min_cells: int = 6, cell_frac_hi: float = 0.45,
                 max_cluster_frac: float = 0.60):
        self.ari_threshold = ari_threshold
        self.weights = weights
        self.min_cells = min_cells
        self.cell_frac_hi = cell_frac_hi
        self.max_cluster_frac = max_cluster_frac

    def cell_band(self, n_prompts: int) -> tuple[float, float]:
        lo = float(min(self.min_cells, max(2, n_prompts // 2)))
        hi = float(max(lo, self.cell_frac_hi * n_prompts))
        return lo, hi

    def pairwise_distance(self, fingerprints, order=None):
        return pairwise_distance_matrix(fingerprints, order=order, weights=self.weights)

    def choose_tau_by_stability(self, fingerprints, tau_grid=None,
                                n_boot: int = 50, rng=None) -> tuple[Optional[float], list]:
        tau_grid = tau_grid or DEFAULT_TAU_GRID
        rng = rng or np.random.default_rng(0)
        n_prompts = len(fingerprints)
        diagnostics = []
        candidates = []
        for tau in tau_grid:
            clusterings = bootstrap_clusterings(
                fingerprints, lambda D: agglomerative_labels(D, tau), n_boot, rng)
            ari = mean_pairwise_metric(clusterings, "ari")
            nmi = mean_pairwise_metric(clusterings, "nmi")
            n_clusters = float(np.mean([len(set(c)) for c in clusterings]))
            compression = 1.0 - n_clusters / n_prompts
            max_frac = float(np.mean([
                max(Counter(c).values()) / len(c) for c in clusterings]))
            rec = dict(tau=tau, ari=ari, nmi=nmi, compression=compression,
                       max_cluster_frac=max_frac, mean_n_clusters=n_clusters)
            diagnostics.append(rec)
            lo, hi = self.cell_band(n_prompts)
            if (ari >= self.ari_threshold and lo <= n_clusters <= hi
                    and max_frac <= self.max_cluster_frac):
                candidates.append(rec)
        if not candidates:
            # Fallback: among stable rows, pick the cell count closest to the
            # middle of the band (in log space, since cell counts span decades).
            lo, hi = self.cell_band(n_prompts)
            mid = float(np.sqrt(lo * hi))
            viable = [d for d in diagnostics if d["ari"] >= self.ari_threshold]
            pool = viable or diagnostics
            best = min(pool, key=lambda d: abs(np.log(max(d["mean_n_clusters"], 1.0))
                                               - np.log(mid)))
            return best["tau"], diagnostics
        best = sorted(candidates, key=lambda d: d["tau"])[0]  # smallest stable tau
        return best["tau"], diagnostics

    def cluster(self, D: np.ndarray, ids: list[str], tau: float,
                fingerprints: dict, source_methods: dict[str, str] = None) -> list[PhenotypeCell]:
        labels = agglomerative_labels(D, tau)
        index_of = {pid: i for i, pid in enumerate(ids)}
        cells: list[PhenotypeCell] = []
        for lab in sorted(set(labels.tolist())):
            members = [ids[i] for i in range(len(ids)) if labels[i] == lab]
            medoid = choose_medoid(members, D, index_of)
            idxs = [index_of[m] for m in members]
            sub = D[np.ix_(idxs, idxs)]
            intra_mean = float(sub[np.triu_indices(len(idxs), 1)].mean()) if len(idxs) > 1 else 0.0
            intra_max = float(sub.max()) if len(idxs) > 1 else 0.0
            src_counts = Counter(
                (source_methods or {}).get(m, "unknown") for m in members)
            cells.append(PhenotypeCell(
                cell_id=f"{fingerprints[members[0]].task_id}_cell_{lab:03d}",
                task_id=fingerprints[members[0]].task_id,
                prompt_ids=members,
                representative_prompt_id=medoid,
                medoid_prompt_id=medoid,
                size=len(members),
                source_methods=dict(src_counts),
                intra_cell_distance_mean=intra_mean,
                intra_cell_distance_max=intra_max,
            ))
        return cells
