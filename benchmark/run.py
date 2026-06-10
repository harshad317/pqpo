"""Unified PQPO benchmark runner.

    python -m benchmark.run \
        --benchmarks ifbench gsm8k mbpp \
        --methods pqpo mipro gepa successive_halving \
        --model gpt-4o --provider openai \
        --mipro-auto medium \
        --workers 12

Runs the selected methods over the selected benchmarks (label / code /
instruction-following modalities), using the phenotype fingerprint appropriate to
each, and reports per-benchmark results plus a cross-benchmark summary. Defaults
to the offline simulated targets; pass --source hf + a real --model for billed runs.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# allow `python -m benchmark.run` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.candidates.pool_builder import build_pool
from pqpo.candidates.proposal_llm import LLMProposalLLM, SimProposalLLM
from pqpo.cli import add_runtime_args
from pqpo.data.datasets import TaskSpec, build_candidate_pool
from pqpo.data.datastructures import CandidatePrompt
from pqpo.evaluation.metrics import behavioral_redundancy_rate, distance_transfer_table
from pqpo.fingerprints.stability import bootstrap_clusterings, mean_pairwise_metric
from pqpo.fingerprints.text_features import build_embeddings
from pqpo.harness import (GENERATOR_METHOD_NAMES, METHOD_NAMES, run_all_selectors,
                          run_source_method_baselines)
from pqpo.logging_utils.cost_ledger import CostLedger
from pqpo.logging_utils.parallel import thread_map
from pqpo.logging_utils.progress import make_bar, metrics_table, rule, titer
from pqpo.quotient.clusterer import QuotientClusterer, agglomerative_labels
from pqpo.selectors.base import SelectorContext
from pqpo.stats.analysis import aubc, holm_bonferroni, paired_bootstrap_diff, spearman

from benchmark.modalities import BENCHMARKS, setup_modality

# ---- method aliases (so `mipro` == MIPROv2 etc.) -------------------------- #
METHOD_ALIASES = {m.lower(): m for m in METHOD_NAMES}
METHOD_ALIASES.update({
    "gepa": "GEPA", "mipro": "MIPROv2", "miprov2": "MIPROv2", "mipro_v2": "MIPROv2",
    "capo": "CAPO", "ape": "APE", "sh": "successive_halving",
    "random": "random_minibatch", "minibatch": "random_minibatch",
})
ALL_METHODS = METHOD_NAMES + GENERATOR_METHOD_NAMES

MIPRO_AUTO = {
    "light": {"pool_size": 12, "generations": 2},
    "medium": {"pool_size": 24, "generations": 3},
    "heavy": {"pool_size": 48, "generations": 4},
}


def resolve_methods_aliased(tokens):
    if not tokens:
        return list(ALL_METHODS)
    flat = []
    for t in tokens:
        flat += [x for x in t.replace(",", " ").split() if x]
    chosen, unknown = set(), []
    for r in flat:
        key = METHOD_ALIASES.get(r.lower())
        if key:
            chosen.add(key)
        elif r in ALL_METHODS:
            chosen.add(r)
        else:
            unknown.append(r)
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}\navailable: {ALL_METHODS}\n"
                         f"(aliases: mipro, gepa, capo, ape, sh, ...)")
    ordered = [m for m in ALL_METHODS if m in chosen]
    if "pqpo" not in ordered:
        ordered.insert(0, "pqpo")          # comparison anchor
    return ordered


def make_seed_prompts(spec, profile_ids):
    texts = [
        "Complete the task correctly. Read the request and answer precisely.",
        "Follow the instructions and produce only the final answer.",
        "Solve the problem step by step, then give the answer in the required format.",
        "Be accurate and literal; satisfy every requirement in the prompt.",
    ]
    meta = {"behavior_profile_id": profile_ids[min(1, len(profile_ids) - 1)]} if profile_ids else {}
    return [CandidatePrompt(f"{spec.task_id}_seed_{i:05d}", spec.task_id, "seed",
                            t, token_length=len(t.split()), metadata=dict(meta))
            for i, t in enumerate(texts)]


def get_sizes(args):
    base = dict(n_sentinel=args.n_sentinel, n_dev=args.n_dev, n_test=args.n_test,
                n_pool=args.n_pool, n_profiles=args.n_profiles, constraints=4)
    return base


def run_one_benchmark(name, args, sel_methods, gen_methods, mipro_auto):
    rng = np.random.default_rng(0)
    ledger = CostLedger(None)
    sizes = get_sizes(args)
    mod = setup_modality(name, args, sizes, ledger, rng)
    spec = TaskSpec(mod.name, mod.modality, mod.label_set, mod.description)

    # ---- candidate pool ----
    if args.pool == "generated":
        seeds = make_seed_prompts(spec, mod.profile_ids)
        proposal = SimProposalLLM(mod.executor, mod.profile_ids) if args.source == "sim" \
            else LLMProposalLLM(mod.executor)
        pool_gens = gen_methods or GENERATOR_METHOD_NAMES
        overrides = {"MIPROv2": MIPRO_AUTO[mipro_auto]} if mipro_auto else None
        built = build_pool(spec.task_id, spec.description, spec.label_set, seeds,
                           mod.dev_ex, proposal, mod.scorer, mod.executor, rng,
                           methods=pool_gens, per_method_size=args.per_method_size,
                           generations=args.generations, method_overrides=overrides)
        pool = built.candidates
    else:
        pool = build_candidate_pool(spec, sizes["n_pool"],
                                    mod.profile_ids or ["p0"], rng, rarity=1.2)
    sm = {p.prompt_id: p.source_method for p in pool}
    embeddings = build_embeddings(pool)

    rule(f"[{name}] modality={mod.modality} | {len(pool)} prompts | "
         f"source={args.source} pool={args.pool} | metric={mod.metric_name}")
    if min(args.budgets) >= len(pool):
        print(f"  [regime note] ALL budgets ({args.budgets}) >= pool size "
              f"({len(pool)}): selectors can evaluate ~the entire pool and converge "
              f"to the oracle prompt, so selection differences vanish. Include "
              f"budgets << pool size to measure selection *efficiency*.")

    # ---- fingerprints + quotient ----
    fps = dict(thread_map(lambda p: (p.prompt_id, mod.fingerprint(p)), pool,
                          workers=args.workers, desc=f"{name} fingerprints"))
    cl = QuotientClusterer()
    D, ids = cl.pairwise_distance(fps)
    tau, tau_diag = cl.choose_tau_by_stability(fps, n_boot=args.n_boot,
                                               rng=np.random.default_rng(1))
    cells = cl.cluster(D, ids, tau, fps, sm)
    quotient_diag = report_quotient_diagnostics(name, cl, D, ids, cells, tau, tau_diag)
    redundancy = behavioral_redundancy_rate(cells, sm)
    boot = bootstrap_clusterings(fps, lambda Dm: agglomerative_labels(Dm, tau),
                                 n_boot=args.n_boot, rng=np.random.default_rng(2))
    ari = mean_pairwise_metric(boot, "ari")

    def held(p):
        # Parallelise across test items (the dominant real-model cost). Cached
        # results return instantly; --workers controls concurrency.
        return np.array(thread_map(
            lambda e: mod.scorer.score_one(p, e, "final_test"),
            mod.test_ex, workers=args.workers, progress=False))

    # Per-prompt held-out (for the oracle reference + distance-transfer table) scores
    # EVERY pool prompt on the full test set — costly on real models. --skip-transfer
    # omits it; the method comparison (which scores only selected prompts) is unaffected.
    if args.skip_transfer:
        per_prompt_ho, transfer = {}, {k: float("nan") for k in
                                       ("phenotype", "embedding", "edit", "length")}
    else:
        per_prompt_ho = dict(thread_map(lambda p: (p.prompt_id, float(held(p).mean())),
                                        pool, workers=args.workers, desc=f"{name} held-out"))
        tbl = distance_transfer_table([p.prompt_id for p in pool], fps, embeddings,
                                      {p.prompt_id: p.prompt_text for p in pool}, per_prompt_ho)
        transfer = {k: spearman(x, y)[0] for k, (x, y) in tbl.items()}

    # Optional cache pre-warm: evaluate the pool x dev matrix in parallel up front
    # (Sec 5.5 precompute) so the sequential selector dev-evals become cache hits.
    # Trades dollar cost (evaluates more pairs than any single budget uses) for a
    # large wall-clock speedup; the budget meter still limits algorithm-visible evals.
    if args.prewarm:
        pairs = [(p, e) for p in pool for e in mod.dev_ex]
        thread_map(lambda pe: mod.scorer.score_one(pe[0], pe[1], "selector_dev"),
                   pairs, workers=args.workers, desc=f"{name} prewarm dev-matrix")

    # ---- run methods ----
    n_cells, sizes_c = len(cells), [c.size for c in cells]
    heldout = {m: {} for m in sel_methods + gen_methods}
    perex = {m: {} for m in sel_methods + gen_methods}

    def factory(b, s):
        return SelectorContext(pool, list(mod.dev_ex), mod.scorer,
                               np.random.default_rng(100 * s + b), b, fps, D, ids,
                               embeddings, sm)

    # Live metric in the progress bar: running PQPO score + current leader.
    bar = make_bar(len(args.budgets) * len(args.seeds),
                   f"{name} sweep [{mod.metric_name}]")
    for b in args.budgets:
        for s in args.seeds:
            res = run_all_selectors(lambda: factory(b, s), n_cells, sizes_c,
                                    embeddings, embeddings, tau=tau, methods=sel_methods)
            res.update(run_source_method_baselines(lambda: factory(b, s), gen_methods))
            for m, r in res.items():
                ho = held(next(p for p in pool if p.prompt_id == r.selected_prompt_id))
                heldout[m].setdefault(b, []).append(float(ho.mean()))
                perex[m][(b, s)] = ho
            # running means across everything seen so far
            means = {m: float(np.mean([v for bb in heldout[m] for v in heldout[m][bb]]))
                     for m in heldout if heldout[m]}
            leader = max(means, key=means.get) if means else "-"
            bar.update(1)
            bar.set_postfix(budget=b, pqpo=f"{means.get('pqpo', float('nan')):.3f}",
                            leader=f"{leader}:{means.get(leader, 0):.3f}")
    bar.close()

    summary = report_one(name, args, mod, pool, cells, redundancy, ari, transfer,
                         heldout, perex, per_prompt_ho, ledger)
    summary["quotient_diagnostics"] = quotient_diag
    return summary


def report_quotient_diagnostics(name, cl, D, ids, cells, tau, tau_diag):
    """Print the tau stability grid + cell-size distribution and flag a degenerate
    quotient (e.g. one giant cell / compression outside the target range), which
    makes PQPO structurally indistinguishable from its clustering controls."""
    n = len(ids)
    n_unique = len(set(agglomerative_labels(D, 1e-9).tolist())) if n > 1 else 1
    cell_sizes = sorted((c.size for c in cells), reverse=True)
    max_frac = cell_sizes[0] / n if n else 0.0
    comp = 1 - len(cells) / n if n else 0.0
    if tau_diag:
        metrics_table(
            f"[{name}] tau stability grid (chosen tau={tau})",
            ["tau", "ARI", "compression", "max cell frac", "mean #cells"],
            [[f"{d['tau']:.2f}", f"{d['ari']:.2f}", f"{d['compression']:.2f}",
              f"{d['max_cluster_frac']:.2f}", f"{d['mean_n_clusters']:.1f}"]
             for d in tau_diag])
    top = ", ".join(str(s) for s in cell_sizes[:10])
    print(f"  cell sizes (top 10): [{top}] | unique fingerprints (d=0): "
          f"{n_unique}/{n} | off-diag distance: median="
          f"{np.median(D[np.triu_indices(n, 1)]):.3f} "
          f"max={D.max():.3f}")
    lo, hi = cl.compression_range
    degenerate = not (lo <= comp <= hi) or max_frac > cl.max_cluster_frac
    if degenerate:
        print(f"  [quotient WARNING] compression {comp:.2f} outside target "
              f"[{lo:.2f},{hi:.2f}] or max cell frac {max_frac:.2f} > "
              f"{cl.max_cluster_frac:.2f}: tau selection fell back. The fingerprint "
              f"is likely too coarse (near-identical vectors -> giant cells; raise "
              f"--n-sentinel or enrich the fingerprint) or too fine. Selector "
              f"comparisons in this regime are NOT a valid test of PQPO.")
    return {"tau": tau, "tau_grid": tau_diag, "cell_sizes": cell_sizes,
            "n_unique_fingerprints": n_unique, "max_cell_frac": max_frac,
            "degenerate": degenerate}


def report_one(name, args, mod, pool, cells, redundancy, ari, transfer,
               heldout, perex, per_prompt_ho, ledger):
    max_b = max(args.budgets)
    aubc_by = {m: float(aubc(sorted(heldout[m]),
               [np.mean(heldout[m][b]) for b in sorted(heldout[m])]))
               for m in heldout if heldout[m]}
    pheno_rows = [["prompts / cells", f"{len(pool)} / {len(cells)}"],
                  ["compression", f"{1 - len(cells)/len(pool):.2f}"],
                  ["cross-lineage redundancy", f"{redundancy:.3f}"],
                  ["cluster stability ARI", f"{ari:.3f}"]]
    if per_prompt_ho:
        pheno_rows += [["oracle best held-out", f"{max(per_prompt_ho.values()):.3f}"],
                       ["distance-transfer phenotype", f"{transfer['phenotype']:.3f}"],
                       ["  vs embed/edit/len",
                        f"{transfer['embedding']:.2f}/{transfer['edit']:.2f}/{transfer['length']:.2f}"]]
    metrics_table(f"[{name}] phenotype quotient", ["metric", "value"], pheno_rows)
    metrics_table(f"[{name}] {mod.metric_name} by method (AUBC over budgets)",
                  ["method", f"AUBC {mod.metric_name}"],
                  [[m, f"{aubc_by[m]:.3f}"] for m in sorted(aubc_by, key=lambda k: -aubc_by[k])])

    # Budget-efficiency: smallest PQPO budget that matches the best generator's
    # (fixed) output quality. This is the efficiency claim, independent of regime.
    gens_present = [m for m in heldout if m in GENERATOR_METHOD_NAMES and heldout[m]]
    if gens_present and heldout.get("pqpo"):
        best_gen_m = max(gens_present,
                         key=lambda m: np.mean(heldout[m][max(heldout[m])]))
        best_gen = float(np.mean(heldout[best_gen_m][max(heldout[best_gen_m])]))
        budgets_sorted = sorted(heldout["pqpo"])
        hit = next((b for b in budgets_sorted
                    if np.mean(heldout["pqpo"][b]) >= best_gen), None)
        msg = (f"budget {hit} (= {hit/len(pool):.0%} of pool)" if hit is not None
               else "not reached in tested budgets")
        metrics_table(f"[{name}] PQPO budget-efficiency vs best generator output",
                      ["quantity", "value"],
                      [["best generator output", f"{best_gen:.3f} ({best_gen_m})"],
                       ["PQPO matches it at", msg],
                       ["pool size", str(len(pool))]])
    eff = None
    if gens_present and heldout.get("pqpo"):
        eff = {"best_generator": best_gen_m, "best_generator_score": best_gen,
               "pqpo_matches_at_budget": hit,
               "fraction_of_pool": (hit / len(pool)) if hit is not None else None}
    # PQPO vs every other method @ max budget (Holm)
    pvals, diffs = {}, {}
    for m in heldout:
        if m == "pqpo" or not perex[m]:
            continue
        seeds = [s for s in args.seeds if (max_b, s) in perex["pqpo"] and (max_b, s) in perex[m]]
        if not seeds:
            continue
        A = np.concatenate([perex["pqpo"][(max_b, s)] for s in seeds])
        B = np.concatenate([perex[m][(max_b, s)] for s in seeds])
        ci = paired_bootstrap_diff(A, B, n_boot=3000)
        pvals[m] = ci.p_value; diffs[m] = ci.mean
    holm = holm_bonferroni(pvals)
    metrics_table(f"[{name}] PQPO vs methods @ budget={max_b} (Holm)",
                  ["method", f"Δ {mod.metric_name}", "adj p", ""],
                  [[m, f"{diffs[m]:+.3f}", f"{holm[m]['adjusted_p']:.4f}",
                    "sig" if holm[m]["reject_null"] else "ns"]
                   for m in sorted(diffs, key=lambda k: -diffs[k])])
    return {
        "benchmark": name, "modality": mod.modality, "metric": mod.metric_name,
        "n_prompts": len(pool), "n_cells": len(cells),
        "compression": 1 - len(cells) / len(pool),
        "redundancy": redundancy, "cluster_stability_ari": ari,
        "oracle_best_heldout": (max(per_prompt_ho.values()) if per_prompt_ho else None),
        "distance_transfer": transfer,
        "aubc_by_method": aubc_by,
        "heldout_by_method_budget": {m: {int(b): float(np.mean(heldout[m][b]))
                                         for b in heldout[m]} for m in heldout if heldout[m]},
        "pqpo_vs_methods_at_max_budget": {
            m: {"delta": diffs[m], "adjusted_p": holm[m]["adjusted_p"],
                "significant": bool(holm[m]["reject_null"])} for m in diffs},
        "budget_efficiency": eff,
        "dollar_cost": ledger.aggregate_all()["dollar_cost"],
    }


def main():
    ap = argparse.ArgumentParser(prog="benchmark.run",
                                 description="Unified PQPO benchmark runner")
    ap.add_argument("--benchmarks", nargs="+", required=True,
                    choices=list(BENCHMARKS), metavar="B",
                    help=f"one or more of: {list(BENCHMARKS)}")
    ap.add_argument("--source", choices=["sim", "hf"], default="sim")
    ap.add_argument("--pool", choices=["synthetic", "generated"], default="synthetic")
    ap.add_argument("--mipro-auto", choices=["light", "medium", "heavy"], default=None,
                    help="MIPROv2 search budget (DSPy-style) when --pool generated")
    ap.add_argument("--per-method-size", type=int, default=12)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--budgets", type=str, default="48,96,192",
                    help="comma-separated labelled budgets")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..n-1)")
    ap.add_argument("--n-sentinel", type=int, default=12,
                    help="number of unlabelled sentinel probes (longer fingerprint "
                         "= finer phenotype discrimination; raise if the pool "
                         "collapses to very few cells)")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-pool", type=int, default=48)
    ap.add_argument("--n-profiles", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=15)
    ap.add_argument("--if-shape", dest="if_shape", action="store_true", default=True,
                    help="IFBench: include coarse response-length in the fingerprint")
    ap.add_argument("--no-if-shape", dest="if_shape", action="store_false",
                    help="IFBench: constraint-satisfaction only (no response shape)")
    ap.add_argument("--report-json", nargs="?", const="__auto__", default=None,
                    metavar="PATH",
                    help="write all results to a JSON file (give a path, or pass the "
                         "flag alone for artifacts/benchmark_report_<timestamp>.json)")
    ap.add_argument("--prewarm", action="store_true",
                    help="parallel-evaluate the pool x dev matrix up front to warm "
                         "the cache, so the selector sweep is cache hits (big "
                         "wall-clock win on real models; raises dollar cost)")
    ap.add_argument("--skip-transfer", action="store_true",
                    help="skip the per-prompt held-out oracle/distance-transfer "
                         "analysis (scores every pool prompt on the full test set); "
                         "big cost saver on real models")
    add_runtime_args(ap)        # --provider --model --temperature --max-output-tokens --workers
    ap.add_argument("--methods", nargs="*", default=None, metavar="M",
                    help="methods/aliases (e.g. pqpo mipro gepa). Default: all.")
    ap.add_argument("--list-methods", action="store_true")
    args = ap.parse_args()

    if args.list_methods:
        print("Methods:", ", ".join(ALL_METHODS))
        print("Aliases:", ", ".join(sorted(set(METHOD_ALIASES) - {m.lower() for m in METHOD_NAMES})))
        return
    args.budgets = [int(x) for x in args.budgets.split(",")]
    args.seeds = list(range(args.seeds))
    chosen = resolve_methods_aliased(args.methods)
    sel_methods = [m for m in chosen if m in METHOD_NAMES]
    gen_methods = [m for m in chosen if m in GENERATOR_METHOD_NAMES]

    rule("PQPO unified benchmark run")
    print(f"  benchmarks : {args.benchmarks}")
    print(f"  selectors  : {sel_methods}")
    print(f"  generators : {gen_methods}")
    print(f"  source={args.source} pool={args.pool} model={args.model or '(sim)'} "
          f"temp={args.temperature} workers={args.workers} "
          f"budgets={args.budgets} seeds={len(args.seeds)}"
          + (f" mipro_auto={args.mipro_auto}" if args.mipro_auto else ""))

    summaries = []
    for name in args.benchmarks:
        summaries.append(run_one_benchmark(name, args, sel_methods, gen_methods,
                                           args.mipro_auto))

    if len(summaries) > 1:
        metrics_table("Cross-benchmark summary — PQPO AUBC and best baseline",
                      ["benchmark", "metric", "PQPO AUBC", "best-baseline AUBC", "PQPO rank"],
                      [_summary_row(s) for s in summaries])

    if args.report_json is not None:
        _write_json_report(args, summaries)


def _write_json_report(args, summaries):
    import datetime
    import json
    path = args.report_json
    if path == "__auto__":
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(here, "artifacts", f"benchmark_report_{ts}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": {
            "benchmarks": args.benchmarks, "source": args.source, "pool": args.pool,
            "provider": args.provider, "model": args.model,
            "temperature": args.temperature, "budgets": args.budgets,
            "seeds": len(args.seeds), "n_sentinel": args.n_sentinel,
            "n_dev": args.n_dev, "n_test": args.n_test, "n_pool": args.n_pool,
            "per_method_size": args.per_method_size, "generations": args.generations,
            "mipro_auto": args.mipro_auto, "skip_transfer": args.skip_transfer,
        },
        "results": summaries,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  [report] wrote JSON results to {path}")


def _summary_row(s):
    ab = s["aubc_by_method"]
    pqpo = ab.get("pqpo", 0.0)
    others = {m: v for m, v in ab.items() if m != "pqpo"}
    best = max(others.values()) if others else 0.0
    rank = 1 + sum(1 for v in others.values() if v > pqpo)
    return [s["benchmark"], s["metric"], f"{pqpo:.3f}", f"{best:.3f}",
            f"{rank}/{len(ab)}"]


if __name__ == "__main__":
    main()
