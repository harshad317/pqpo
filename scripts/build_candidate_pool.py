"""Build a candidate pool with the real GEPA/MIPROv2/CAPO generators, then run
PQPO over that generated pool (Sec 4.1–4.6).

Demonstrates:
  * each generator's proposal/reflection/target call accounting,
  * cross-lineage behavioural redundancy in the *generated* pool (the PQPO premise
    measured on real proposal streams, not a synthetic pool),
  * PQPO selecting over the generated pool vs a non-behavioural baseline,
  * a full cost breakdown by call type.

Runs on the simulated target by default; set provider to a billed model and pass
LLMProposalLLM to generate real prompts.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.api.cache import APICache
from pqpo.api.executor import APIExecutor, ModelConfig
from pqpo.api.sim_target import SimTarget, build_default_profiles
from pqpo.candidates.pool_builder import build_pool
from pqpo.candidates.proposal_llm import SimProposalLLM
from pqpo.data.datasets import (build_candidate_pool, default_task_specs,
                                make_synthetic_examples)
from pqpo.evaluation.metrics import behavioral_redundancy_rate, held_out_scores
from pqpo.evaluation.scorers import Scorer
from pqpo.fingerprints.extractor import BehaviorFingerprintExtractor
from pqpo.fingerprints.normalizers import ClassificationNormalizer
from pqpo.logging_utils.cost_ledger import CostLedger
from pqpo.logging_utils.progress import metrics_table, rule, titer
from pqpo.quotient.clusterer import QuotientClusterer
from pqpo.selectors import minibatch, pqpo_fixed_pool
from pqpo.selectors.base import SelectorContext


def main():
    # A 16-profile target gives the pool genuine behavioural diversity (more
    # distinct phenotype cells, a rarer best cell) -- the regime PQPO targets.
    profiles = build_default_profiles(np.random.default_rng(7), 16)
    profile_ids = list(profiles.keys())              # ordered by ascending skill
    specs = default_task_specs(); spec = specs[0]
    ls = {s.task_id: s.label_set for s in specs}
    sim = SimTarget(profiles, ls)
    ledger = CostLedger(None)
    ex = APIExecutor(ModelConfig("sim", "sim-target"), ledger, APICache(None), "pool", sim)
    norm = ClassificationNormalizer(spec.label_set)
    extr = BehaviorFingerprintExtractor(norm); scorer = Scorer(ex, norm)
    proposal_llm = SimProposalLLM(ex, profile_ids)
    rng = np.random.default_rng(0)

    # A handful of weak seed prompts to start each optimizer from.
    seeds = build_candidate_pool(spec, 6, profile_ids, rng, rarity=0.0)
    for s in seeds:
        s.metadata["behavior_profile_id"] = profile_ids[1]   # weak start
    train = make_synthetic_examples(spec, 120, rng, "train")
    sentinels = make_synthetic_examples(spec, 12, rng, "sentinel")
    for u in sentinels: u.target = None
    test = make_synthetic_examples(spec, 200, rng, "test")

    rule(f"Building candidate pool — task '{spec.task_id}' (GEPA + MIPROv2 + CAPO)")
    res = build_pool(spec.task_id, spec.description, spec.label_set, seeds, train,
                     proposal_llm, scorer, ex, rng, per_method_size=50,
                     generations=6, population=8, minibatch=8)
    pool = res.candidates

    # Fingerprint the generated pool and measure cross-lineage redundancy.
    fps = {}
    for p in titer(pool, desc="fingerprinting pool", total=len(pool)):
        souts = [extr.parse_sentinel(ex.run_prompt_on_example(p, u, "sentinel"), f"s{j}")
                 for j, u in enumerate(sentinels)]
        fps[p.prompt_id] = extr.extract(p.prompt_id, souts)
    sm = {p.prompt_id: p.source_method for p in pool}
    cl = QuotientClusterer()
    D, ids = cl.pairwise_distance(fps)
    tau, _ = cl.choose_tau_by_stability(fps, n_boot=25, rng=np.random.default_rng(1))
    cells = cl.cluster(D, ids, tau, fps, sm)
    redundancy = behavioral_redundancy_rate(cells, sm)

    # PQPO vs random minibatch over the GENERATED pool, across budgets.
    def ctx(budget, seed=3):
        return SelectorContext(pool, list(train), scorer, np.random.default_rng(seed),
                               budget, fps, D, ids, source_methods=sm)

    def ho_of(res):
        p = next(p for p in pool if p.prompt_id == res.selected_prompt_id)
        return float(held_out_scores(p, test, scorer).mean())

    budget_sweep = [24, 48, 96]
    sweep_rows = []
    for b in budget_sweep:
        pq = np.mean([ho_of(pqpo_fixed_pool.run_pqpo_fixed_pool(ctx(b, s), tau=tau))
                      for s in range(3)])
        rm = np.mean([ho_of(minibatch.run_random_minibatch(ctx(b, s))) for s in range(3)])
        sweep_rows.append([b, f"{pq:.3f}", f"{rm:.3f}", f"{pq - rm:+.3f}"])

    cost = ledger.aggregate_all()
    metrics_table(
        f"Per-method generation — task '{spec.task_id}'",
        ["method", "candidates", "proposal", "reflection", "target calls"],
        [[m, res.per_method_counts[m], c["proposal"], c["reflection"], c["target"]]
         for m, c in res.per_method_calls.items()])
    metrics_table(
        "Generated-pool structure + PQPO over the pool",
        ["metric", "value"],
        [["raw / deduped candidates", f"{res.metadata['n_raw']} / {res.metadata['n_deduped']}"],
         ["lineage mix (deduped)", str(res.metadata['lineage_mix'])],
         ["phenotype cells (tau)", f"{len(cells)} (tau={tau})"],
         ["compression", f"{1 - len(cells)/len(pool):.2f}"],
         ["cross-lineage redundancy", f"{redundancy:.3f}"]])
    metrics_table(
        "PQPO vs random-minibatch over the generated pool (mean of 3 seeds)",
        ["budget", "PQPO held-out", "random-mb held-out", "Δ (PQPO-base)"],
        sweep_rows)
    metrics_table(
        "Cost by call type",
        ["call type", "count"],
        [[k, cost[k]] for k in ["proposal_calls", "reflection_calls", "sentinel_calls",
                                "selector_dev_calls", "final_test_calls"]]
        + [["dollar_cost", f"${cost['dollar_cost']:.4f}"]])


if __name__ == "__main__":
    main()
