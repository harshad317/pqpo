"""End-to-end PQPO MVP over a CODE benchmark with the rich test-pass fingerprint.

Pipeline: candidate pool -> per-prompt code on sentinel problems -> test-pass
behavioural fingerprints -> phenotype clustering -> PQPO vs baselines selecting on
labelled dev problems -> held-out pass@1 on test problems. Runs offline against a
deterministic simulated *code* target; swap to a billed ModelConfig + real
CodeProblems from `pqpo.data.code_loaders` for actual MBPP+/HumanEval+/LiveCodeBench.

Reports the code-modality breakthrough evidence: behavioural redundancy, cluster
stability, distance-transfer predictiveness, AUBC-by-method, and Holm-corrected
PQPO-vs-baseline pass@1 differences.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.api.cache import APICache
from pqpo.api.executor import APIExecutor, ModelConfig
from pqpo.cli import (add_method_args, add_runtime_args, build_model_config,
                      maybe_list_methods, resolve_methods)
from pqpo.logging_utils.parallel import thread_map

# Methods available in the code MVP (a subset of the full selector catalog).
CODE_METHODS = ["pqpo", "random_minibatch", "successive_halving", "lexical_cluster",
                "embedding_cluster", "random_quotient", "map_elites"]
from pqpo.api.sim_code_target import build_sim_code_target
from pqpo.data.datasets import TaskSpec, build_candidate_pool
from pqpo.evaluation.code_scorer import CodeScorer, problems_to_examples
from pqpo.fingerprints.code_extractor import CodeBehaviorFingerprintExtractor
from pqpo.fingerprints.distances import pairwise_distance_matrix
from pqpo.fingerprints.stability import bootstrap_clusterings, mean_pairwise_metric
from pqpo.fingerprints.text_features import build_embeddings
from pqpo.evaluation.metrics import behavioral_redundancy_rate, distance_transfer_table
from pqpo.logging_utils.cost_ledger import CostLedger
from pqpo.logging_utils.progress import metrics_table, rule, titer
from pqpo.quotient.clusterer import QuotientClusterer, agglomerative_labels
from pqpo.selectors import minibatch, multifidelity, clustering_controls, pqpo_fixed_pool
from pqpo.selectors.base import SelectorContext
from pqpo.stats.analysis import aubc, holm_bonferroni, paired_bootstrap_diff, spearman


def get_cfg(quick):
    # The execution cache makes high budgets cheap (identical programs are only
    # run once), so we use budgets large enough that pass@1 cell estimates carry
    # signal rather than pure binary noise.
    if quick:
        return dict(n_profiles=8, n_pool=48, n_sentinel=12, n_dev=120, n_test=100,
                    budgets=[48, 96, 192], seeds=[0, 1, 2], n_boot=15)
    return dict(n_profiles=10, n_pool=80, n_sentinel=12, n_dev=160, n_test=150,
                budgets=[96, 192, 384], seeds=[0, 1, 2, 3, 4], n_boot=30)


def held_out_pass_at_1(prompt, test_examples, scorer):
    return np.array([scorer.score_one(prompt, e, call_type="final_test")
                     for e in test_examples])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    add_runtime_args(ap)        # --provider --model --temperature --max-output-tokens --workers
    add_method_args(ap)         # --methods / --list-methods
    args = ap.parse_args()
    maybe_list_methods(args, CODE_METHODS)
    methods = resolve_methods(args, CODE_METHODS, always_include=("pqpo",))
    cfg = get_cfg(args.quick or True)  # default to quick sizing (subprocess cost)

    rng = np.random.default_rng(0)
    n_problems = cfg["n_sentinel"] + cfg["n_dev"] + cfg["n_test"]
    sim, problems = build_sim_code_target(cfg["n_profiles"], n_problems + 10, rng)
    profile_ids = list(sim.profiles.keys())
    sentinels = problems[:cfg["n_sentinel"]]
    dev_problems = problems[cfg["n_sentinel"]:cfg["n_sentinel"] + cfg["n_dev"]]
    test_problems = problems[cfg["n_sentinel"] + cfg["n_dev"]:
                             cfg["n_sentinel"] + cfg["n_dev"] + cfg["n_test"]]

    ledger = CostLedger(None)
    model_cfg = build_model_config(args, simulated=True, sim_model="sim-code-target")
    ex = APIExecutor(model_cfg, ledger, APICache(None), "codemvp", sim_target=sim)
    scorer = CodeScorer(ex, timeout=3.0)
    extractor = CodeBehaviorFingerprintExtractor(timeout=3.0)

    spec = TaskSpec("code", "code", ["A"], "Write a correct Python function.")
    pool = build_candidate_pool(spec, cfg["n_pool"], profile_ids, rng, rarity=1.2)
    sm = {p.prompt_id: p.source_method for p in pool}
    embeddings = build_embeddings(pool)
    dev_ex = problems_to_examples(dev_problems, "code")
    test_ex = problems_to_examples(test_problems, "code")

    rule(f"PQPO code-MVP — {len(pool)} prompts, {cfg['n_sentinel']} sentinel "
         f"problems, rich test-pass fingerprint")

    # 1. Rich code fingerprints over the sentinel problems (parallel; --workers).
    fps = dict(thread_map(lambda p: (p.prompt_id, extractor.extract(p, sentinels, ex)),
                          pool, workers=args.workers, desc="code fingerprints"))

    cl = QuotientClusterer()
    D, ids = cl.pairwise_distance(fps)
    tau, _ = cl.choose_tau_by_stability(fps, n_boot=cfg["n_boot"], rng=np.random.default_rng(1))
    cells = cl.cluster(D, ids, tau, fps, sm)
    redundancy = behavioral_redundancy_rate(cells, sm)
    boot = bootstrap_clusterings(fps, lambda Dm: agglomerative_labels(Dm, tau),
                                 n_boot=cfg["n_boot"], rng=np.random.default_rng(2))
    ari = mean_pairwise_metric(boot, "ari")

    # 2. Per-prompt held-out pass@1 (for transfer analysis + oracle reference).
    per_prompt_ho = dict(thread_map(
        lambda p: (p.prompt_id, float(held_out_pass_at_1(p, test_ex, scorer).mean())),
        pool, workers=args.workers, desc="held-out pass@1"))
    tbl = distance_transfer_table([p.prompt_id for p in pool], fps, embeddings,
                                  {p.prompt_id: p.prompt_text for p in pool}, per_prompt_ho)
    transfer = {k: spearman(x, y)[0] for k, (x, y) in tbl.items()}

    # 3. Selectors across budgets x seeds.
    n_cells = len(cells); sizes = [c.size for c in cells]
    heldout = {m: {} for m in methods}        # m -> {budget: [pass@1 per seed]}
    perex = {m: {} for m in methods}          # m -> {(budget,seed): np.ndarray}

    def ctx(b, s):
        return SelectorContext(pool, list(dev_ex), scorer, np.random.default_rng(100 * s + b),
                               b, fps, D, ids, embeddings, sm)

    registry = lambda b, s: {
        "pqpo": lambda: pqpo_fixed_pool.run_pqpo_fixed_pool(ctx(b, s), tau=tau),
        "random_minibatch": lambda: minibatch.run_random_minibatch(ctx(b, s)),
        "successive_halving": lambda: multifidelity.run_successive_halving(ctx(b, s)),
        "lexical_cluster": lambda: clustering_controls.run_lexical_cluster(ctx(b, s), n_cells, embeddings),
        "embedding_cluster": lambda: clustering_controls.run_embedding_cluster(ctx(b, s), n_cells, embeddings),
        "random_quotient": lambda: clustering_controls.run_random_quotient(ctx(b, s), sizes),
        "map_elites": lambda: clustering_controls.run_map_elites(ctx(b, s)),
    }
    for b in titer(cfg["budgets"], desc="budgets", total=len(cfg["budgets"])):
        for s in cfg["seeds"]:
            reg = registry(b, s)
            runs = {m: reg[m]() for m in methods if m in reg}
            for m, r in runs.items():
                ho = held_out_pass_at_1(next(p for p in pool if p.prompt_id == r.selected_prompt_id),
                                        test_ex, scorer)
                heldout[m].setdefault(b, []).append(float(ho.mean()))
                perex[m][(b, s)] = ho

    report(cfg, cells, pool, tau, redundancy, ari, transfer, heldout, perex,
           per_prompt_ho, ledger)


def report(cfg, cells, pool, tau, redundancy, ari, transfer, heldout, perex,
           per_prompt_ho, ledger):
    max_b = max(cfg["budgets"])
    aubc_by = {m: float(np.mean([aubc(sorted(heldout[m]), [np.mean(heldout[m][b])
               for b in sorted(heldout[m])])])) for m in heldout}
    # PQPO vs baselines at max budget
    rows_cmp = []
    pvals, diffs = {}, {}
    for m in heldout:
        if m == "pqpo":
            continue
        A = np.concatenate([perex["pqpo"][(max_b, s)] for s in cfg["seeds"]])
        B = np.concatenate([perex[m][(max_b, s)] for s in cfg["seeds"]])
        ci = paired_bootstrap_diff(A, B, n_boot=3000)
        pvals[m] = ci.p_value; diffs[m] = ci.mean
    holm = holm_bonferroni(pvals)
    for m in sorted(diffs, key=lambda k: -diffs[k]):
        sig = "sig" if holm[m]["reject_null"] else "ns"
        rows_cmp.append([m, f"{diffs[m]:+.3f}", f"{holm[m]['adjusted_p']:.4f}", sig])

    metrics_table("Code-MVP — phenotype quotient over test-pass fingerprints",
                  ["metric", "value"],
                  [["prompts / phenotype cells", f"{len(pool)} / {len(cells)}"],
                   ["compression", f"{1 - len(cells)/len(pool):.2f}"],
                   ["cross-lineage redundancy", f"{redundancy:.3f}"],
                   ["cluster stability ARI", f"{ari:.3f}"],
                   ["oracle best held-out pass@1", f"{max(per_prompt_ho.values()):.3f}"],
                   ["distance-transfer Spearman (phenotype)", f"{transfer['phenotype']:.3f}"],
                   ["  vs embedding / edit / length",
                    f"{transfer['embedding']:.3f} / {transfer['edit']:.3f} / {transfer['length']:.3f}"]])
    metrics_table("Held-out pass@1 by method (AUBC over budgets)",
                  ["method", "AUBC pass@1"],
                  [[m, f"{aubc_by[m]:.3f}"] for m in sorted(aubc_by, key=lambda k: -aubc_by[k])])
    metrics_table(f"PQPO vs baselines @ budget={max_b} (Holm-corrected)",
                  ["baseline", "Δ pass@1 (PQPO-base)", "adj p", ""], rows_cmp)
    cost = ledger.aggregate_all()
    print(f"\n  sentinel/dev/test code executions logged; dollar cost "
          f"${cost['dollar_cost']:.4f}. Phenotype distance predicts transfer "
          f"({transfer['phenotype']:.2f}) far better than embedding/edit/length.")


if __name__ == "__main__":
    main()
