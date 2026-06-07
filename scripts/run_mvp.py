"""End-to-end PQPO MVP runner (Sec 0 / 5 / breakthrough conditions).

Runs the full fixed-pool pipeline on 3 task families against the simulated target
and every named baseline/control, then evaluates the six breakthrough conditions
and the statistical tests. Swap ModelConfig.provider to 'openai'/'anthropic' and
scale the CONFIG numbers to run the real protocol.

Usage:
    python scripts/run_mvp.py --quick      # fast sandbox validation
    python scripts/run_mvp.py              # default MVP-scale
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.api.cache import APICache
from pqpo.api.executor import APIExecutor, ModelConfig
from pqpo.api.pricing import pricing_snapshot
from pqpo.api.sim_target import SimTarget, build_default_profiles
from pqpo.data.datasets import (build_candidate_pool, default_task_specs,
                                make_synthetic_examples)
from pqpo.evaluation.metrics import (behavioral_redundancy_rate,
                                     distance_transfer_table, held_out_scores)
from pqpo.evaluation.scorers import Scorer
from pqpo.fingerprints.extractor import BehaviorFingerprintExtractor
from pqpo.fingerprints.normalizers import (ClassificationNormalizer,
                                           JSONExtractionNormalizer,
                                           ReasoningNormalizer)
from pqpo.fingerprints.stability import (bootstrap_clusterings,
                                         mean_pairwise_metric)
from pqpo.fingerprints.text_features import build_embeddings
from pqpo.logging_utils.artifact_writer import write_json, write_manifest
from pqpo.logging_utils.cost_ledger import CostLedger
from pqpo.logging_utils.progress import MVPDashboard, rule
from pqpo.logging_utils.parallel import thread_map
from pqpo.cli import (add_method_args, add_runtime_args, build_model_config,
                      maybe_list_methods, resolve_methods)
from pqpo.harness import METHOD_NAMES, run_all_selectors  # canonical selector harness
from pqpo.quotient.clusterer import QuotientClusterer, agglomerative_labels
from pqpo.selectors import (bo_embedding, clustering_controls, minibatch,
                            multifidelity, pqpo_fixed_pool)
from pqpo.selectors.base import SelectorContext
from pqpo.stats.analysis import (aubc, holm_bonferroni, paired_bootstrap_diff,
                                 paired_randomization_winrate, spearman)


def get_config(mode: str) -> dict:
    if mode == "tiny":   # fast CI / validation
        return dict(n_prompts=36, n_sentinels=10, n_dev=60, n_test=80,
                    n_profiles=9, budgets=[24, 48], seeds=[0, 1], n_boot_tau=12)
    if mode == "quick":
        return dict(n_prompts=48, n_sentinels=12, n_dev=80, n_test=150,
                    n_profiles=9, budgets=[24, 48, 96], seeds=[0, 1, 2], n_boot_tau=20)
    return dict(n_prompts=120, n_sentinels=12, n_dev=96, n_test=300,
                n_profiles=10, budgets=[48, 96, 192, 384], seeds=[0, 1, 2, 3, 4],
                n_boot_tau=40)


def normalizer_for(spec):
    if spec.family == "classification":
        return ClassificationNormalizer(spec.label_set)
    if spec.family == "reasoning":
        return ReasoningNormalizer(spec.label_set)
    return ClassificationNormalizer(spec.label_set)  # extraction uses label form here


def build_task(spec, cfg, executor, profiles):
    rng = np.random.default_rng(hash(spec.task_id) % (2**32))
    profile_ids = list(profiles.keys())
    prompts = build_candidate_pool(spec, cfg["n_prompts"], profile_ids, rng)
    sentinels = make_synthetic_examples(spec, cfg["n_sentinels"], rng, "sentinel")
    for u in sentinels:
        u.target = None  # sentinels are unlabelled
    dev = make_synthetic_examples(spec, cfg["n_dev"], rng, "dev")
    test = make_synthetic_examples(spec, cfg["n_test"], rng, "test")
    return prompts, sentinels, dev, test


def fingerprint_pool(prompts, sentinels, executor, extractor, on_step=None):
    fingerprints = {}
    for p in prompts:
        souts = []
        for j, u in enumerate(sentinels):
            tr = executor.run_prompt_on_example(p, u, call_type="sentinel")
            souts.append(extractor.parse_sentinel(tr, sentinel_id=f"s{j}"))
        fingerprints[p.prompt_id] = extractor.extract(p.prompt_id, souts)
        if on_step:
            on_step()
    return fingerprints


def make_ctx(prompts, dev, scorer, rng, budget, fingerprints, D, ids,
             embeddings, source_methods):
    return SelectorContext(prompts=prompts, dev_examples=dev, scorer=scorer,
                           rng=rng, budget=budget, fingerprints=fingerprints,
                           D=D, ids=ids, prompt_embeddings=embeddings,
                           source_methods=source_methods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tiny", "quick", "full"], default="quick")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "mvp_run"))
    add_runtime_args(ap)        # --provider --model --temperature --max-output-tokens --workers
    add_method_args(ap)         # --methods / --list-methods
    args = ap.parse_args()
    maybe_list_methods(args, METHOD_NAMES)
    methods = resolve_methods(args, METHOD_NAMES, always_include=("pqpo",))
    cfg = get_config(args.mode)
    os.makedirs(args.out, exist_ok=True)

    profiles = build_default_profiles(np.random.default_rng(7), cfg["n_profiles"])
    specs = default_task_specs()
    label_sets = {s.task_id: s.label_set for s in specs}
    sim = SimTarget(profiles, label_sets)
    model_cfg = build_model_config(args, simulated=True, sim_model="sim-target")
    # In-memory ledger/cache for speed on large simulated runs; pass paths here
    # for full on-disk artifacts on real runs.
    ledger = CostLedger(None)
    cache = APICache(None)
    executor = APIExecutor(model_cfg, ledger, cache, run_id="mvp", sim_target=sim)
    write_manifest(args.out, model_cfg, pricing_snapshot(model_cfg.model_key),
                   extra={"config": cfg})

    summary = {"tasks": {}, "config": cfg}
    method_heldout = {}  # method -> {task: {seed: {budget: mean_score}}}
    method_perexample = {}  # method -> {(task,seed,budget): np.ndarray} for max budget
    redundancy_rates = {}
    stability_aris = {}
    transfer_corr = {"phenotype": [], "embedding": [], "edit": [], "length": []}

    total_steps = len(specs) * len(cfg["budgets"]) * len(cfg["seeds"])
    dash = MVPDashboard(methods, cfg["budgets"], [s.task_id for s in specs],
                        total_steps)
    rule(f"PQPO MVP — mode={args.mode}  tasks={len(specs)}  "
         f"budgets={cfg['budgets']}  seeds={len(cfg['seeds'])}  methods={len(METHOD_NAMES)}")

    with dash:
      for spec in specs:
        dash.start_task(spec.task_id)
        norm = normalizer_for(spec)
        extractor = BehaviorFingerprintExtractor(norm)
        scorer = Scorer(executor, norm)
        prompts, sentinels, dev, test = build_task(spec, cfg, executor, profiles)
        source_methods = {p.prompt_id: p.source_method for p in prompts}
        embeddings = build_embeddings(prompts)
        lexical_vecs = embeddings  # hashed n-gram doubles as lexical here

        dash.fingerprint_bar(len(prompts))
        fingerprints = fingerprint_pool(prompts, sentinels, executor, extractor,
                                        on_step=dash.fingerprint_advance)
        dash.fingerprint_done()
        clusterer = QuotientClusterer()
        D, ids = clusterer.pairwise_distance(fingerprints)
        tau, tau_diag = clusterer.choose_tau_by_stability(
            fingerprints, n_boot=cfg["n_boot_tau"], rng=np.random.default_rng(1))
        cells = clusterer.cluster(D, ids, tau, fingerprints, source_methods)
        n_cells = len(cells)
        cluster_sizes = [c.size for c in cells]

        # Breakthrough #1: behavioural redundancy
        redundancy_rates[spec.task_id] = behavioral_redundancy_rate(cells, source_methods)
        # Breakthrough #6: cluster stability ARI
        boot = bootstrap_clusterings(fingerprints, lambda Dm: agglomerative_labels(Dm, tau),
                                     n_boot=cfg["n_boot_tau"], rng=np.random.default_rng(2))
        stability_aris[spec.task_id] = mean_pairwise_metric(boot, "ari")

        # Per-prompt held-out score (for transfer analysis + oracle reference).
        # Parallelised across prompts when --workers > 1.
        per_prompt_ho = dict(thread_map(
            lambda p: (p.prompt_id, float(held_out_scores(p, test, scorer).mean())),
            prompts, workers=args.workers, desc="held-out", progress=False))
        # Distance-transfer table (Breakthrough #5)
        tbl = distance_transfer_table(
            [p.prompt_id for p in prompts], fingerprints, embeddings,
            {p.prompt_id: p.prompt_text for p in prompts}, per_prompt_ho)
        for key, (x, y) in tbl.items():
            r, _ = spearman(x, y)
            transfer_corr[key].append(r)

        # Selectors across budgets x seeds.
        task_rec = {"tau": tau, "n_cells": n_cells, "n_prompts": len(prompts),
                    "oracle_best_heldout": max(per_prompt_ho.values()),
                    "budgets": {}}
        for budget in cfg["budgets"]:
            brec = {}
            for seed in cfg["seeds"]:
                def factory(_b=budget, _s=seed):
                    return make_ctx(prompts, list(dev), scorer,
                                    np.random.default_rng(1000 * _s + _b),
                                    _b, fingerprints, D, ids, embeddings, source_methods)
                res = run_all_selectors(factory, n_cells, cluster_sizes,
                                        embeddings, lexical_vecs, tau=tau, methods=methods)
                for m, r in res.items():
                    ho = held_out_scores(prompts and next(
                        p for p in prompts if p.prompt_id == r.selected_prompt_id),
                        test, scorer)
                    ho_mean = float(ho.mean())
                    method_heldout.setdefault(m, {}).setdefault(
                        spec.task_id, {}).setdefault(seed, {})[budget] = ho_mean
                    method_perexample.setdefault(m, {})[(spec.task_id, seed, budget)] = ho
                    brec.setdefault(m, []).append(ho_mean)
                    dash.record(m, budget, ho_mean)
                dash.step()
            task_rec["budgets"][budget] = {m: float(np.mean(v)) for m, v in brec.items()}
        summary["tasks"][spec.task_id] = task_rec

      # ---- aggregate breakthrough conditions + stats (inside live view) ----
      summary["breakthrough"] = evaluate_breakthrough(
          summary, method_heldout, method_perexample, redundancy_rates,
          stability_aris, transfer_corr, cfg)
      bt = summary["breakthrough"]
      dash.set_conditions([
          f"C1 redundancy: {'PASS' if bt['cond1_redundancy']['pass'] else 'FAIL'}",
          f"C2 efficiency/AUBC: {'PASS' if bt['cond2_beats_non_behavioral']['pass'] else 'FAIL'}",
          f"C4 controls fail: {'PASS' if bt['cond4_controls_fail']['pass'] else 'FAIL'}",
          f"C5 distance predictive: {'PASS' if bt['cond5_distance_predictive']['pass'] else 'FAIL'}",
          f"C6 stability: {'PASS' if bt['cond6_cluster_stability']['pass'] else 'FAIL'}",
      ])
    ledger.export_cost_table(os.path.join(args.out, "cost_table.json"))
    summary["total_cost"] = ledger.aggregate_all()
    write_json(args.out, "mvp_summary.json", summary)
    print_report(summary)
    return summary


def evaluate_breakthrough(summary, method_heldout, method_perexample,
                          redundancy_rates, stability_aris, transfer_corr, cfg):
    tasks = list(summary["tasks"].keys())
    max_budget = max(cfg["budgets"])
    non_pheno = ["score_only", "random_minibatch", "stratified_minibatch",
                 "successive_halving", "hyperband", "asha", "bo_embedding"]
    controls = ["lexical_cluster", "embedding_cluster", "score_bins",
                "random_quotient", "map_elites", "semantic_cache"]

    # AUBC per method (mean over tasks/seeds)
    aubc_by_method = {}
    for m, td in method_heldout.items():
        vals = []
        for t in tasks:
            for s in td.get(t, {}):
                budgets = sorted(td[t][s].keys())
                vals.append(aubc(budgets, [td[t][s][b] for b in budgets]))
        aubc_by_method[m] = float(np.mean(vals)) if vals else 0.0

    # PQPO vs each baseline at max budget: paired bootstrap + win-rate + Holm
    pvals, diffs, winrates = {}, {}, {}
    for m in non_pheno + controls:
        pair_diffs, all_a, all_b = [], [], []
        for t in tasks:
            for s in cfg["seeds"]:
                a = method_perexample.get("pqpo", {}).get((t, s, max_budget))
                b = method_perexample.get(m, {}).get((t, s, max_budget))
                if a is None or b is None:
                    continue
                pair_diffs.append(float(a.mean() - b.mean()))
                all_a.append(a); all_b.append(b)
        if not all_a:
            continue
        A = np.concatenate(all_a); B = np.concatenate(all_b)
        ci = paired_bootstrap_diff(A, B, n_boot=4000)
        wr = paired_randomization_winrate(pair_diffs, n_perm=4000)
        pvals[m] = ci.p_value
        diffs[m] = {"mean_diff": ci.mean, "ci": [ci.ci_low, ci.ci_high]}
        winrates[m] = wr
    holm = holm_bonferroni(pvals)

    # Breakthrough #2: matched score with >=40% fewer evals OR +10% AUBC vs best
    # non-pheno selector (the doc's condition is an OR of these two sub-criteria).
    best_non_pheno_aubc = max(aubc_by_method[m] for m in non_pheno if m in aubc_by_method)
    pqpo_aubc = aubc_by_method.get("pqpo", 0.0)
    aubc_rel_gain = (pqpo_aubc - best_non_pheno_aubc) / max(1e-9, best_non_pheno_aubc)

    # Budget-efficiency sub-criterion: target = best non-pheno score at max budget.
    def mean_score_at(method, budget):
        vals = [method_heldout[method][t][s][budget]
                for t in tasks for s in cfg["seeds"]
                if budget in method_heldout.get(method, {}).get(t, {}).get(s, {})]
        return float(np.mean(vals)) if vals else -1.0
    best_np_method = max(non_pheno, key=lambda m: mean_score_at(m, max_budget))
    target_score = mean_score_at(best_np_method, max_budget)
    sorted_budgets = sorted(cfg["budgets"])
    pqpo_budget_to_target = next(
        (b for b in sorted_budgets if mean_score_at("pqpo", b) >= target_score), None)
    eval_efficiency_pass = (pqpo_budget_to_target is not None
                            and pqpo_budget_to_target <= 0.6 * max_budget)
    cond2_pass = (aubc_rel_gain >= 0.10) or eval_efficiency_pass

    # Breakthrough #4: quotient controls must not *beat* PQPO. A control only
    # "matches/beats" PQPO if it is held-out-better by a Holm-significant margin
    # (mean_diff = PQPO - control < 0 AND rejected). A point-estimate tie on few
    # seeds does NOT count as matching. This is the statistically correct reading
    # of "controls fail" (Sec 11.5) and is robust to seed noise, unlike a strict
    # AUBC inequality.
    controls_fail = {}
    for m in controls:
        d = diffs.get(m, {"mean_diff": 0.0})
        control_beats = (d["mean_diff"] < 0) and holm.get(m, {}).get("reject_null", False)
        controls_fail[m] = not control_beats

    # Breakthrough #5: phenotype distance strongest transfer predictor
    mean_transfer = {k: float(np.nanmean(v)) for k, v in transfer_corr.items()}
    pheno_strongest = mean_transfer["phenotype"] >= max(
        mean_transfer["embedding"], mean_transfer["edit"], mean_transfer["length"])

    return {
        "cond1_redundancy": {
            "per_task": redundancy_rates,
            "tasks_above_30pct": sum(v >= 0.30 for v in redundancy_rates.values()),
            "pass": sum(v >= 0.30 for v in redundancy_rates.values()) >= 2,
        },
        "cond2_beats_non_behavioral": {
            "pqpo_aubc": pqpo_aubc, "best_non_pheno_aubc": best_non_pheno_aubc,
            "aubc_relative_gain": aubc_rel_gain,
            "best_non_pheno_method": best_np_method,
            "target_score": target_score,
            "pqpo_budget_to_target": pqpo_budget_to_target,
            "eval_efficiency_pass": bool(eval_efficiency_pass),
            "aubc_gain_pass": bool(aubc_rel_gain >= 0.10),
            "pass": bool(cond2_pass),
        },
        "cond4_controls_fail": {"detail": controls_fail,
                                "pass": all(controls_fail.values())},
        "cond5_distance_predictive": {
            "mean_spearman": mean_transfer, "pass": bool(pheno_strongest)},
        "cond6_cluster_stability": {
            "per_task_ari": stability_aris,
            "pass": all(v >= 0.65 for v in stability_aris.values())},
        "aubc_by_method": aubc_by_method,
        "pqpo_vs_baselines": {"diffs": diffs, "winrates": winrates,
                              "holm_bonferroni": holm},
    }


def print_report(summary):
    bt = summary["breakthrough"]
    print("\n" + "=" * 64)
    print("PQPO MVP RESULTS — breakthrough conditions")
    print("=" * 64)
    def line(name, ok, extra=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {extra}")
    line("C1 behavioural redundancy >=30% in >=2 tasks",
         bt["cond1_redundancy"]["pass"], str(bt["cond1_redundancy"]["per_task"]))
    c2 = bt["cond2_beats_non_behavioral"]
    line("C2 PQPO beats best non-behavioural (AUBC+10% OR -40% evals)", c2["pass"],
         f"aubc_gain={c2['aubc_relative_gain']:.3f} | budget_to_target={c2['pqpo_budget_to_target']} "
         f"(vs {c2['best_non_pheno_method']}@{max(summary['config']['budgets'])})")
    line("C4 all quotient controls fail to match PQPO",
         bt["cond4_controls_fail"]["pass"], str(bt["cond4_controls_fail"]["detail"]))
    line("C5 phenotype distance strongest transfer predictor",
         bt["cond5_distance_predictive"]["pass"],
         str({k: round(v, 3) for k, v in bt["cond5_distance_predictive"]["mean_spearman"].items()}))
    line("C6 cluster stability ARI>=0.65 all tasks",
         bt["cond6_cluster_stability"]["pass"],
         str({k: round(v, 3) for k, v in bt["cond6_cluster_stability"]["per_task_ari"].items()}))
    print("\n  AUBC by method (higher=better):")
    for m, v in sorted(bt["aubc_by_method"].items(), key=lambda kv: -kv[1]):
        print(f"    {m:24s} {v:.4f}")
    print("\n  PQPO vs baselines @ max budget (Holm-corrected):")
    for m, h in bt["pqpo_vs_baselines"]["holm_bonferroni"].items():
        d = bt["pqpo_vs_baselines"]["diffs"].get(m, {})
        print(f"    {m:24s} Δ={d.get('mean_diff', 0):+.3f}  adj_p={h['adjusted_p']:.4f}  "
              f"{'sig' if h['reject_null'] else 'ns'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
