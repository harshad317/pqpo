"""Bayesian optimization over prompt embeddings (Sec 4.10).

A lightweight GP-UCB-style optimiser over the fixed pool: model held-out-proxy
score as a function of prompt embedding, pick the arm with the highest UCB,
evaluate it, update. Uses a kernel-ridge surrogate to avoid heavy GP deps while
preserving the BO behaviour (exploration via predictive uncertainty proxy).
"""
from __future__ import annotations

import numpy as np

from .base import BudgetMeter, SelectorContext, SelectorResult


def _rbf(A, B, gamma):
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))


def run_bo_embedding(ctx: SelectorContext, embeddings: dict,
                     minibatch: int = 4, kappa: float = 1.5,
                     gamma: float = 1.0, ridge: float = 1e-2) -> SelectorResult:
    meter = BudgetMeter(ctx.budget)
    ids = list(ctx.ids)
    X = np.stack([embeddings[i] for i in ids])
    n = len(ids)
    observed_idx, observed_y = [], []
    scores: dict[str, list[float]] = {}

    # seed with a few random arms
    seed_arms = list(ctx.rng.choice(n, size=min(5, n), replace=False))
    for ai in seed_arms:
        if meter.remaining <= 0:
            break
        s = ctx.eval_prompt(ids[ai], ctx.sample_dev(min(minibatch, meter.remaining)), meter)
        if s:
            scores[ids[ai]] = s
            observed_idx.append(ai); observed_y.append(float(np.mean(s)))

    while meter.remaining > 0 and len(observed_idx) < n:
        Xo = X[observed_idx]
        yo = np.array(observed_y)
        K = _rbf(Xo, Xo, gamma) + ridge * np.eye(len(Xo))
        Kinv = np.linalg.inv(K)
        Ks = _rbf(X, Xo, gamma)
        mu = Ks @ Kinv @ yo
        var = 1.0 - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks)
        var = np.clip(var, 1e-6, None)
        ucb = mu + kappa * np.sqrt(var)
        for ai in observed_idx:
            ucb[ai] = -np.inf
        nxt = int(np.argmax(ucb))
        s = ctx.eval_prompt(ids[nxt], ctx.sample_dev(min(minibatch, meter.remaining)), meter)
        if s:
            scores[ids[nxt]] = s
            observed_idx.append(nxt); observed_y.append(float(np.mean(s)))

    best = max(scores, key=lambda p: np.mean(scores[p]) if scores[p] else -1)
    return SelectorResult(selected_prompt_id=best, labeled_evals_used=meter.used,
                          metadata={"method": "bo_embedding", "n_evaluated": len(scores)})
