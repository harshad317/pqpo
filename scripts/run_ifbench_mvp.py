"""Run ALL methods on IFBench (instruction-following with verifiable constraints).

PQPO + every baseline/control select over a candidate-prompt pool using the
per-constraint satisfaction fingerprint; held-out metric is IFEval strict
prompt-level accuracy (all constraints satisfied). Runs offline against a
deterministic simulated IF target; set --source hf and a billed ModelConfig for
real IFEval/IFBench.

Usage:
    python scripts/run_ifbench_mvp.py                 # offline, simulated target
    python scripts/run_ifbench_mvp.py --source hf     # real IFEval (needs `datasets`)
    python scripts/run_ifbench_mvp.py --source hf --pool generated
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.api.cache import APICache
from pqpo.api.executor import APIExecutor, ModelConfig
from pqpo.cli import (add_method_args, add_runtime_args, build_model_config,
                      maybe_list_methods, resolve_methods)
from pqpo.logging_utils.parallel import parallel_map
from pqpo.candidates.pool_builder import build_pool
from pqpo.candidates.proposal_llm import LLMProposalLLM, SimProposalLLM
from pqpo.data.datasets import TaskSpec, build_candidate_pool
from pqpo.data.datastructures import CandidatePrompt
from pqpo.evaluation.ifbench import (IFBehaviorFingerprintExtractor, IFScorer,
                                     build_sim_if_target, if_items_to_examples,
                                     load_ifbench)
from pqpo.evaluation.metrics import behavioral_redundancy_rate, distance_transfer_table
from pqpo.fingerprints.stability import bootstrap_clusterings, mean_pairwise_metric
from pqpo.fingerprints.text_features import build_embeddings
from pqpo.logging_utils.cost_ledger import CostLedger
from pqpo.logging_utils.progress import metrics_table, rule, titer
from pqpo.quotient.clusterer import QuotientClusterer, agglomerative_labels
from pqpo.selectors.base import BudgetMeter, SelectorContext, SelectorResult
from pqpo.stats.analysis import aubc, holm_bonferroni, paired_bootstrap_diff, spearman

# reuse the full 14-method matrix + runner from the text MVP
from run_mvp import METHOD_NAMES, run_all_selectors


GENERATOR_METHOD_NAMES = ["GEPA", "MIPROv2", "CAPO", "APE"]


def get_cfg(quick):
    if quick:
        return dict(n_profiles=8, n_pool=48, n_sentinel=12, n_dev=120, n_test=100,
                    constraints=4, budgets=[48, 96, 192], seeds=[0, 1, 2], n_boot=15)
    return dict(n_profiles=10, n_pool=80, n_sentinel=12, n_dev=160, n_test=150,
                constraints=5, budgets=[96, 192, 384], seeds=[0, 1, 2, 3, 4], n_boot=30)


def make_seed_prompts(spec: TaskSpec, profile_ids: list[str] | None = None) -> list[CandidatePrompt]:
    texts = [
        "Follow every instruction in the user request exactly. Answer directly.",
        "Satisfy all formatting, length, keyword, and style constraints in the prompt.",
        "Read the full request, check each constraint, then produce only the final answer.",
        "Comply with every visible instruction while staying concise and literal.",
        "Before answering, account for required words, case, bullets, and endings.",
        "Return a response that obeys all constraints in the prompt.",
    ]
    meta = {}
    if profile_ids:
        meta["behavior_profile_id"] = profile_ids[min(1, len(profile_ids) - 1)]
    return [
        CandidatePrompt(
            prompt_id=f"{spec.task_id}_seed_{i:05d}",
            task_id=spec.task_id,
            source_method="seed",
            prompt_text=text,
            token_length=len(text.split()),
            metadata=dict(meta),
        )
        for i, text in enumerate(texts)
    ]


def run_source_method_baseline(ctx: SelectorContext, source_method: str) -> SelectorResult | None:
    """Select the best prompt emitted by one generator/source under the same
    labelled dev-eval budget used by the other selectors."""
    arms = [p.prompt_id for p in ctx.prompts if p.source_method == source_method]
    if not arms:
        return None
    ctx.rng.shuffle(arms)
    scores = {a: [] for a in arms}
    meter = BudgetMeter(ctx.budget)
    i = 0
    while meter.remaining > 0:
        arm = arms[i % len(arms)]
        scores[arm].extend(ctx.eval_prompt(arm, ctx.sample_dev(1), meter))
        i += 1
    best = max(scores, key=lambda p: np.mean(scores[p]) if scores[p] else -1)
    return SelectorResult(
        selected_prompt_id=best,
        labeled_evals_used=meter.used,
        metadata={"method": source_method, "n_arms": len(arms),
                  "baseline": "source_method_best_of_pool"},
    )


def run_source_method_baselines(ctx_factory, source_methods: list[str]):
    out = {}
    for source_method in source_methods:
        res = run_source_method_baseline(ctx_factory(), source_method)
        if res is not None:
            out[source_method] = res
    return out


def _if_fingerprint_worker(payload):
    prompt, sentinels, executor, extractor = payload
    before = len(executor.ledger.traces)
    fp = extractor.extract(prompt, sentinels, executor)
    return prompt.prompt_id, fp, executor.ledger.traces[before:]


def _if_heldout_mean_worker(payload):
    prompt, examples, scorer = payload
    before = len(scorer.executor.ledger.traces)
    scores = [scorer.score_one(prompt, e, "final_test") for e in examples]
    return prompt.prompt_id, float(np.mean(scores)), scorer.executor.ledger.traces[before:]


def _merge_child_traces(ledger, records, enabled: bool) -> None:
    if not enabled:
        return
    for rec in records:
        for trace in rec[-1]:
            ledger.record_api_call(trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "hf"], default="sim")
    ap.add_argument("--pool", choices=["synthetic", "generated"], default="synthetic",
                    help=("synthetic uses the fixed labelled source pool; generated runs "
                          "GEPA/MIPROv2/CAPO/APE candidate generators first"))
    ap.add_argument("--per-method-size", type=int, default=12,
                    help="candidates per generator when --pool generated")
    ap.add_argument("--generations", type=int, default=3,
                    help="generator rounds when --pool generated")
    ap.add_argument("--population", type=int, default=6,
                    help="generator population when --pool generated")
    ap.add_argument("--minibatch", type=int, default=6,
                    help="generator feedback minibatch when --pool generated")
    ap.add_argument("--quick", action="store_true", default=True)
    add_runtime_args(ap)        # --provider --model --temperature --max-output-tokens --workers
    add_method_args(ap)         # --methods / --list-methods
    args = ap.parse_args()

    available = METHOD_NAMES + GENERATOR_METHOD_NAMES
    maybe_list_methods(args, available)
    chosen = resolve_methods(args, available, always_include=("pqpo",))
    sel_methods = [m for m in chosen if m in METHOD_NAMES]
    gen_methods = [m for m in chosen if m in GENERATOR_METHOD_NAMES]
    cfg = get_cfg(args.quick)

    rng = np.random.default_rng(0)
    ledger = CostLedger(None)

    if args.source == "hf":
        sentinels, dev_items, test_items = load_ifbench(
            cfg["n_sentinel"], cfg["n_dev"], cfg["n_test"], seed=0, source="hf")
        model_cfg = build_model_config(args, simulated=False)   # --provider/--model/keys
        ex = APIExecutor(model_cfg, ledger, APICache(None), "ifbench")
        profile_ids = []
    else:
        n_items = cfg["n_sentinel"] + cfg["n_dev"] + cfg["n_test"] + 20
        sim, items = build_sim_if_target(cfg["n_profiles"], n_items, rng, cfg["constraints"])
        sentinels = items[:cfg["n_sentinel"]]
        dev_items = items[cfg["n_sentinel"]:cfg["n_sentinel"] + cfg["n_dev"]]
        test_items = items[cfg["n_sentinel"] + cfg["n_dev"]:
                           cfg["n_sentinel"] + cfg["n_dev"] + cfg["n_test"]]
        model_cfg = build_model_config(args, simulated=True, sim_model="sim-if-target")
        ex = APIExecutor(model_cfg, ledger, APICache(None), "ifbench", sim_target=sim)
        profile_ids = list(sim.profiles.keys())

    scorer = IFScorer(ex, strict=True)
    extractor = IFBehaviorFingerprintExtractor()
    spec = TaskSpec("ifbench", "instruction_following", ["A"],
                    "Follow every instruction exactly.")
    dev_ex = if_items_to_examples(dev_items)
    test_ex = if_items_to_examples(test_items)
    pool_info = {"mode": args.pool}
    if args.pool == "generated":
        seeds = make_seed_prompts(spec, profile_ids if args.source == "sim" else None)
        proposal_llm = (
            SimProposalLLM(ex, profile_ids)
            if args.source == "sim"
            else LLMProposalLLM(ex)
        )
        # Need generators to build a pool; if the user selected only selectors,
        # fall back to the full generator set for pool construction.
        pool_gen_methods = gen_methods or GENERATOR_METHOD_NAMES
        built = build_pool(
            spec.task_id, spec.description, spec.label_set, seeds, dev_ex,
            proposal_llm, scorer, ex, rng, methods=pool_gen_methods,
            per_method_size=args.per_method_size, generations=args.generations,
            population=args.population, minibatch=args.minibatch,
        )
        pool = built.candidates
        pool_info.update({
            "per_method_counts": built.per_method_counts,
            "per_method_calls": built.per_method_calls,
            "lineage_mix": built.metadata["lineage_mix"],
            "n_raw": built.metadata["n_raw"],
            "n_deduped": built.metadata["n_deduped"],
        })
    else:
        pool = build_candidate_pool(spec, cfg["n_pool"],
                                    profile_ids if args.source == "sim" else ["p0"],
                                    rng, rarity=1.2)
        pool_info["lineage_mix"] = dict(Counter(p.source_method for p in pool))
    sm = {p.prompt_id: p.source_method for p in pool}
    embeddings = build_embeddings(pool)

    rule(f"PQPO IFBench — selector methods + GEPA/MIPROv2/CAPO/APE baselines | "
         f"pool={args.pool} | {len(pool)} prompts, {len(sentinels)} sentinel items")

    if args.parallel_backend == "process" and args.source != "sim":
        print("  [parallel note] process backend runs separate API clients/caches; "
              "thread backend is usually safer for real providers and rate limits.")

    # 1. Per-constraint fingerprints (parallelised across prompts; --workers).
    fp_records = parallel_map(
        _if_fingerprint_worker,
        [(p, sentinels, ex, extractor) for p in pool],
        workers=args.workers,
        backend=args.parallel_backend,
        desc="IF fingerprints",
    )
    _merge_child_traces(ledger, fp_records,
                        args.parallel_backend == "process" and args.workers > 1)
    fps = {pid: fp for pid, fp, _ in fp_records}
    cl = QuotientClusterer()
    D, ids = cl.pairwise_distance(fps)
    tau, _ = cl.choose_tau_by_stability(fps, n_boot=cfg["n_boot"], rng=np.random.default_rng(1))
    cells = cl.cluster(D, ids, tau, fps, sm)
    redundancy = behavioral_redundancy_rate(cells, sm)
    boot = bootstrap_clusterings(fps, lambda Dm: agglomerative_labels(Dm, tau),
                                 n_boot=cfg["n_boot"], rng=np.random.default_rng(2))
    ari = mean_pairwise_metric(boot, "ari")

    def strict_acc(p, examples):
        return np.array([scorer.score_one(p, e, "final_test") for e in examples])

    ho_records = parallel_map(
        _if_heldout_mean_worker,
        [(p, test_ex, scorer) for p in pool],
        workers=args.workers,
        backend=args.parallel_backend,
        desc="held-out strict acc",
    )
    _merge_child_traces(ledger, ho_records,
                        args.parallel_backend == "process" and args.workers > 1)
    per_prompt_ho = {pid: mean_score for pid, mean_score, _ in ho_records}
    tbl = distance_transfer_table([p.prompt_id for p in pool], fps, embeddings,
                                  {p.prompt_id: p.prompt_text for p in pool}, per_prompt_ho)
    transfer = {k: spearman(x, y)[0] for k, (x, y) in tbl.items()}

    # 2. All methods across budgets x seeds.
    n_cells = len(cells); sizes = [c.size for c in cells]
    heldout = {m: {} for m in sel_methods}
    perex = {m: {} for m in sel_methods}
    generator_heldout = {m: {} for m in gen_methods}
    generator_perex = {m: {} for m in gen_methods}

    def factory(b, s):
        return SelectorContext(pool, list(dev_ex), scorer, np.random.default_rng(100 * s + b),
                               b, fps, D, ids, embeddings, sm)

    for b in titer(cfg["budgets"], desc="budgets", total=len(cfg["budgets"])):
        for s in cfg["seeds"]:
            res = run_all_selectors(lambda: factory(b, s), n_cells, sizes,
                                    embeddings, embeddings, tau=tau, methods=sel_methods)
            for m, r in res.items():
                ho = strict_acc(next(p for p in pool if p.prompt_id == r.selected_prompt_id), test_ex)
                heldout[m].setdefault(b, []).append(float(ho.mean()))
                perex[m][(b, s)] = ho
            gen_res = run_source_method_baselines(lambda: factory(b, s), gen_methods)
            for m, r in gen_res.items():
                ho = strict_acc(next(p for p in pool if p.prompt_id == r.selected_prompt_id), test_ex)
                generator_heldout[m].setdefault(b, []).append(float(ho.mean()))
                generator_perex[m][(b, s)] = ho

    report(cfg, pool, cells, redundancy, ari, transfer, heldout, perex,
           generator_heldout, generator_perex, per_prompt_ho, ledger, pool_info)


def report(cfg, pool, cells, redundancy, ari, transfer, heldout, perex,
           generator_heldout, generator_perex, per_prompt_ho, ledger, pool_info):
    max_b = max(cfg["budgets"])
    aubc_by = {m: float(aubc(sorted(heldout[m]),
               [np.mean(heldout[m][b]) for b in sorted(heldout[m])])) for m in heldout}
    pvals, diffs = {}, {}
    for m in heldout:
        if m == "pqpo":
            continue
        A = np.concatenate([perex["pqpo"][(max_b, s)] for s in cfg["seeds"]])
        B = np.concatenate([perex[m][(max_b, s)] for s in cfg["seeds"]])
        ci = paired_bootstrap_diff(A, B, n_boot=3000)
        pvals[m] = ci.p_value; diffs[m] = ci.mean
    holm = holm_bonferroni(pvals)

    metrics_table("IFBench — phenotype quotient over per-constraint fingerprints",
                  ["metric", "value"],
                  [["prompts / phenotype cells", f"{len(pool)} / {len(cells)}"],
                   ["compression", f"{1 - len(cells)/len(pool):.2f}"],
                   ["cross-lineage redundancy", f"{redundancy:.3f}"],
                   ["cluster stability ARI", f"{ari:.3f}"],
                   ["oracle best held-out strict acc", f"{max(per_prompt_ho.values()):.3f}"],
                   ["distance-transfer Spearman (phenotype)", f"{transfer['phenotype']:.3f}"],
                   ["  vs embedding / edit / length",
                    f"{transfer['embedding']:.3f} / {transfer['edit']:.3f} / {transfer['length']:.3f}"]])
    if pool_info.get("per_method_calls"):
        metrics_table("Generated candidate pool",
                      ["generator", "candidates", "proposal", "reflection", "target calls"],
                      [[m, pool_info["per_method_counts"].get(m, 0),
                        pool_info["per_method_calls"].get(m, {}).get("proposal", 0),
                        pool_info["per_method_calls"].get(m, {}).get("reflection", 0),
                        pool_info["per_method_calls"].get(m, {}).get("target", 0)]
                       for m in GENERATOR_METHOD_NAMES])
    else:
        metrics_table("Candidate source mix",
                      ["source", "candidates"],
                      [[m, pool_info.get("lineage_mix", {}).get(m, 0)]
                       for m in GENERATOR_METHOD_NAMES])
    metrics_table("Held-out strict accuracy by method (AUBC over budgets)",
                  ["method", "AUBC strict acc"],
                  [[m, f"{aubc_by[m]:.3f}"] for m in sorted(aubc_by, key=lambda k: -aubc_by[k])])
    active_generators = [m for m in GENERATOR_METHOD_NAMES if generator_heldout.get(m)]
    if active_generators:
        gen_aubc = {
            m: float(aubc(sorted(generator_heldout[m]),
                          [np.mean(generator_heldout[m][b])
                           for b in sorted(generator_heldout[m])]))
            for m in active_generators
        }
        metrics_table("GEPA / MIPROv2 / CAPO / APE baselines (best-of-source pool)",
                      ["generator", "AUBC strict acc"],
                      [[m, f"{gen_aubc[m]:.3f}"]
                       for m in sorted(gen_aubc, key=lambda k: -gen_aubc[k])])
        gen_rows = []
        gen_pvals, gen_diffs = {}, {}
        for m in active_generators:
            seeds = [s for s in cfg["seeds"]
                     if (max_b, s) in perex["pqpo"] and (max_b, s) in generator_perex[m]]
            if not seeds:
                continue
            A = np.concatenate([perex["pqpo"][(max_b, s)] for s in seeds])
            B = np.concatenate([generator_perex[m][(max_b, s)] for s in seeds])
            ci = paired_bootstrap_diff(A, B, n_boot=3000)
            gen_pvals[m] = ci.p_value; gen_diffs[m] = ci.mean
        gen_holm = holm_bonferroni(gen_pvals) if gen_pvals else {}
        for m in sorted(gen_diffs, key=lambda k: -gen_diffs[k]):
            gen_rows.append([m, f"{gen_diffs[m]:+.3f}",
                             f"{gen_holm[m]['adjusted_p']:.4f}",
                             "sig" if gen_holm[m]["reject_null"] else "ns"])
        metrics_table(f"PQPO vs generator baselines @ budget={max_b} (Holm-corrected)",
                      ["generator", "Δ strict acc (PQPO-gen)", "adj p", ""], gen_rows)
    rows = []
    for m in sorted(diffs, key=lambda k: -diffs[k]):
        rows.append([m, f"{diffs[m]:+.3f}", f"{holm[m]['adjusted_p']:.4f}",
                     "sig" if holm[m]["reject_null"] else "ns"])
    metrics_table(f"PQPO vs all baselines @ budget={max_b} (Holm-corrected)",
                  ["baseline", "Δ strict acc (PQPO-base)", "adj p", ""], rows)
    print(f"\n  cost ${ledger.aggregate_all()['dollar_cost']:.4f}. Per-constraint "
          f"satisfaction fingerprint -> phenotype distance predicts transfer "
          f"({transfer['phenotype']:.2f}) far above embedding/edit/length.")


if __name__ == "__main__":
    main()
