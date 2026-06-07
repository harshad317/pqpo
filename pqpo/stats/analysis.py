"""Statistical tests (Sec 5.6 / 11.5).

  * paired bootstrap CIs over held-out examples
  * hierarchical bootstrap over (task, seed, example)
  * paired randomization win-rate test
  * area under the budget-performance curve (log budget)
  * Spearman distance-transfer correlations
  * Holm-Bonferroni multiple-comparison correction
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class BootstrapCI:
    mean: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                          alpha: float = 0.05, rng=None) -> BootstrapCI:
    """Paired bootstrap of mean(a) - mean(b) over matched examples."""
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a, float); b = np.asarray(b, float)
    diff = a - b
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value for H0: mean diff = 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return BootstrapCI(float(diff.mean()), float(lo), float(hi), float(min(1.0, p)))


def hierarchical_bootstrap(results: dict, n_boot: int = 5000, rng=None) -> BootstrapCI:
    """results: {task: {seed: np.ndarray of per-example (a-b) diffs}}.
    Resamples tasks, then seeds within task, then examples."""
    rng = rng or np.random.default_rng(0)
    tasks = list(results.keys())
    boots = []
    for _ in range(n_boot):
        t_sample = rng.choice(tasks, len(tasks), replace=True)
        vals = []
        for t in t_sample:
            seeds = list(results[t].keys())
            s_sample = rng.choice(seeds, len(seeds), replace=True)
            for s in s_sample:
                arr = results[t][s]
                if len(arr):
                    vals.append(arr[rng.integers(0, len(arr), len(arr))].mean())
        if vals:
            boots.append(np.mean(vals))
    boots = np.array(boots)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return BootstrapCI(float(boots.mean()), float(lo), float(hi), float(min(1.0, p)))


def paired_randomization_winrate(pair_diffs: list[float], n_perm: int = 10000,
                                 rng=None) -> dict:
    """pair_diffs[i] = score_PQPO - score_baseline on (task,seed) pair i.
    Null: signs are exchangeable (equal win probability)."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(pair_diffs, float)
    obs = d.mean()
    perms = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(n_perm)])
    p = (np.abs(perms) >= abs(obs)).mean()
    wins = int((d > 0).sum()); losses = int((d < 0).sum()); ties = int((d == 0).sum())
    return {"observed_mean_diff": float(obs), "p_value": float(p),
            "wins": wins, "losses": losses, "ties": ties,
            "win_rate": wins / len(d) if len(d) else 0.0}


def aubc(budgets: list[float], scores: list[float]) -> float:
    """Area under the budget-performance curve, integrated over log budget,
    normalised by the log-budget range."""
    b = np.asarray(budgets, float); s = np.asarray(scores, float)
    order = np.argsort(b)
    b, s = b[order], s[order]
    logb = np.log(np.maximum(b, 1.0))
    if logb[-1] == logb[0]:
        return float(s.mean())
    return float(np.trapz(s, logb) / (logb[-1] - logb[0]))


def spearman(x, y) -> tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0
    with np.errstate(all="ignore"):
        r, p = sps.spearmanr(x, y)
    if not np.isfinite(r):
        return 0.0, 1.0
    return float(r), float(p)


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down. Returns per-comparison adjusted p and reject flag."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - rank) * p))
        prev = adj
        out[name] = {"raw_p": p, "adjusted_p": adj, "reject_null": adj < alpha}
    return out
