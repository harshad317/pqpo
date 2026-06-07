"""Smoke + correctness tests for the PQPO pipeline.

Run: python -m pytest tests/ -q     (or: python tests/test_pipeline.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.api.cache import APICache, request_hash
from pqpo.api.executor import APIExecutor, ModelConfig
from pqpo.api.sim_target import SimTarget, build_default_profiles
from pqpo.data.datasets import (build_candidate_pool, default_task_specs,
                                make_synthetic_examples)
from pqpo.evaluation.scorers import Scorer
from pqpo.fingerprints.distances import fingerprint_distance, pairwise_distance_matrix
from pqpo.fingerprints.extractor import BehaviorFingerprintExtractor
from pqpo.fingerprints.normalizers import ClassificationNormalizer
from pqpo.logging_utils.cost_ledger import CostLedger
from pqpo.quotient.clusterer import QuotientClusterer
from pqpo.selectors import minibatch, pqpo_fixed_pool
from pqpo.selectors.base import SelectorContext
from pqpo.stats.analysis import holm_bonferroni, paired_bootstrap_diff


def _setup(n_prompts=24, n_sent=8):
    profiles = build_default_profiles(np.random.default_rng(7), 9)
    specs = default_task_specs(); spec = specs[0]
    sim = SimTarget(profiles, {s.task_id: s.label_set for s in specs})
    ex = APIExecutor(ModelConfig("sim", "sim-target"), CostLedger(None), APICache(None), "t", sim)
    norm = ClassificationNormalizer(spec.label_set)
    extr = BehaviorFingerprintExtractor(norm); scorer = Scorer(ex, norm)
    rng = np.random.default_rng(0)
    prompts = build_candidate_pool(spec, n_prompts, list(profiles.keys()), rng)
    sent = make_synthetic_examples(spec, n_sent, rng, "sentinel")
    for u in sent: u.target = None
    dev = make_synthetic_examples(spec, 40, rng, "dev")
    fps = {}
    for p in prompts:
        so = [extr.parse_sentinel(ex.run_prompt_on_example(p, u, "sentinel"), f"s{j}")
              for j, u in enumerate(sent)]
        fps[p.prompt_id] = extr.extract(p.prompt_id, so)
    return spec, prompts, sent, dev, fps, scorer


def test_request_hash_deterministic():
    a = request_hash("m", "v", "in", 0.0, 64, None)
    b = request_hash("m", "v", "in", 0.0, 64, None)
    c = request_hash("m", "v", "in2", 0.0, 64, None)
    assert a == b and a != c


def test_cache_replays_zero_cost():
    profiles = build_default_profiles(np.random.default_rng(1), 9)
    specs = default_task_specs()
    sim = SimTarget(profiles, {s.task_id: s.label_set for s in specs})
    ledger = CostLedger(None)
    ex = APIExecutor(ModelConfig("sim", "sim-target"), ledger, APICache(None), "t", sim)
    rng = np.random.default_rng(0)
    p = build_candidate_pool(specs[0], 1, list(profiles.keys()), rng)[0]
    e = make_synthetic_examples(specs[0], 1, rng, "dev")[0]
    t1 = ex.run_prompt_on_example(p, e, "selector_dev")
    t2 = ex.run_prompt_on_example(p, e, "selector_dev")
    assert not t1.cache_hit and t2.cache_hit
    assert t2.dollar_cost == t1.dollar_cost  # same priced tokens
    assert ledger.aggregate_all()["cache_hits"] == 1


def test_fingerprint_distance_bounds():
    _, _, _, _, fps, _ = _setup()
    ids = list(fps)
    for a in ids[:5]:
        assert fingerprint_distance(fps[a], fps[a]) == 0.0
        for b in ids[:5]:
            d = fingerprint_distance(fps[a], fps[b])
            assert 0.0 <= d <= 1.0 + 1e-9


def test_clustering_compresses_and_recovers_structure():
    _, prompts, _, _, fps, _ = _setup(n_prompts=36)
    sm = {p.prompt_id: p.source_method for p in prompts}
    cl = QuotientClusterer()
    D, ids = cl.pairwise_distance(fps)
    tau, _ = cl.choose_tau_by_stability(fps, n_boot=10, rng=np.random.default_rng(1))
    cells = cl.cluster(D, ids, tau, fps, sm)
    assert 1 < len(cells) < len(prompts)              # genuine compression
    assert sum(c.size for c in cells) == len(prompts)  # partition is complete


def test_pqpo_runs_within_budget():
    spec, prompts, sent, dev, fps, scorer = _setup(n_prompts=24)
    sm = {p.prompt_id: p.source_method for p in prompts}
    cl = QuotientClusterer(); D, ids = cl.pairwise_distance(fps)
    tau, _ = cl.choose_tau_by_stability(fps, n_boot=8, rng=np.random.default_rng(1))
    ctx = SelectorContext(prompts, list(dev), scorer, np.random.default_rng(3),
                          budget=32, fingerprints=fps, D=D, ids=ids,
                          source_methods=sm)
    res = pqpo_fixed_pool.run_pqpo_fixed_pool(ctx, tau=tau)
    assert res.labeled_evals_used <= 32
    assert res.selected_prompt_id in {p.prompt_id for p in prompts}


def test_candidate_generators_produce_pools_with_provenance():
    from pqpo.candidates.pool_builder import build_pool
    from pqpo.candidates.proposal_llm import SimProposalLLM
    profiles = build_default_profiles(np.random.default_rng(7), 10)
    pids = list(profiles.keys())
    specs = default_task_specs(); spec = specs[0]
    sim = SimTarget(profiles, {s.task_id: s.label_set for s in specs})
    ex = APIExecutor(ModelConfig("sim", "sim-target"), CostLedger(None), APICache(None), "g", sim)
    norm = ClassificationNormalizer(spec.label_set); scorer = Scorer(ex, norm)
    rng = np.random.default_rng(0)
    seeds = build_candidate_pool(spec, 5, pids, rng, rarity=0.0)
    for s in seeds: s.metadata["behavior_profile_id"] = pids[1]
    train = make_synthetic_examples(spec, 60, rng, "train")
    res = build_pool(spec.task_id, spec.description, spec.label_set, seeds, train,
                     SimProposalLLM(ex, pids), scorer, ex, rng,
                     per_method_size=20, generations=3, population=6, minibatch=6)
    methods = {c.source_method for c in res.candidates}
    assert {"GEPA", "MIPROv2", "CAPO"} <= methods           # all three present
    assert len(res.candidates) > 20                          # meaningful pool
    # provenance: at least some candidates carry proposal/reflection call ids
    assert any(c.proposal_call_ids or c.reflection_call_ids for c in res.candidates)
    # cost ledger recorded proposal/reflection calls
    agg = ex.ledger.aggregate_all()
    assert agg["proposal_calls"] > 0 and agg["reflection_calls"] > 0


def test_gepa_behavior_hillclimbs():
    """Later GEPA generations should reach higher-skill behaviour profiles."""
    from pqpo.candidates.gepa_adapter import GEPAAdapter
    from pqpo.candidates.base import GenerationConfig
    from pqpo.candidates.proposal_llm import SimProposalLLM
    profiles = build_default_profiles(np.random.default_rng(7), 10)
    pids = list(profiles.keys())
    specs = default_task_specs(); spec = specs[0]
    sim = SimTarget(profiles, {s.task_id: s.label_set for s in specs})
    ex = APIExecutor(ModelConfig("sim", "sim-target"), CostLedger(None), APICache(None), "g", sim)
    norm = ClassificationNormalizer(spec.label_set); scorer = Scorer(ex, norm)
    rng = np.random.default_rng(1)
    seeds = build_candidate_pool(spec, 4, pids, rng, rarity=0.0)
    for s in seeds: s.metadata["behavior_profile_id"] = pids[1]
    train = make_synthetic_examples(spec, 60, rng, "train")
    cfg = GenerationConfig(spec.task_id, spec.description, spec.label_set,
                           pool_size=40, generations=6, population=8, minibatch=6)
    res = GEPAAdapter(SimProposalLLM(ex, pids), scorer, ex).generate(cfg, seeds, train, rng)
    def gen_of(c): return c.metadata.get("generation", 0)
    early = [pids.index(c.metadata["behavior_profile_id"]) for c in res.candidates if gen_of(c) <= 1 and "behavior_profile_id" in c.metadata]
    late = [pids.index(c.metadata["behavior_profile_id"]) for c in res.candidates if gen_of(c) >= 4 and "behavior_profile_id" in c.metadata]
    assert early and late
    assert np.mean(late) > np.mean(early)                    # behaviour improved


def test_code_exec_classifies_outcomes():
    from pqpo.evaluation.code_exec import CodeProblem, run_tests
    prob = CodeProblem("p", "add", ["assert add(2,3)==5", "assert add(0,0)==0"])
    assert run_tests("def add(a,b):\n return a+b", prob, 2).all_pass
    assert run_tests("def add(a,b):\n return a-b", prob, 2).n_pass == 1  # wrong on first
    assert run_tests("def add(a,b)\n return a+b", prob, 2).dominant_error == "syntax"
    assert run_tests("def add(a,b):\n return zzz", prob, 2).dominant_error == "runtime"
    assert run_tests("def add(a,b):\n while True: pass", prob, 1).dominant_error == "timeout"


def test_code_fingerprint_clusters_by_behavior():
    """Lexically different but behaviourally identical programs share a cell;
    buggy programs separate."""
    from pqpo.data.datastructures import APITrace, CandidatePrompt
    from pqpo.evaluation.code_exec import CodeProblem
    from pqpo.fingerprints.code_extractor import CodeBehaviorFingerprintExtractor
    from pqpo.quotient.clusterer import QuotientClusterer

    class Stub:
        def run_prompt_on_example(self, prompt, example, call_type="sentinel"):
            prog = prompt.metadata["program"]
            return APITrace("c", "r", "stub", "stub", prompt.task_id, call_type,
                            "h", example.input_text, prog, 0.0, 64, 1,
                            max(1, len(prog) // 4), 2, 10.0, 0.0, False, "rh")

    sentinels = [CodeProblem("add", "add(a,b)",
                             ["assert add(2,3)==5", "assert add(1,1)==2", "assert add(0,0)==0"])]
    progs = {
        "ok_terse": "def add(a,b):\n return a+b",
        "ok_verbose": "def add(a,b):\n s = a\n s += b\n return s",   # same behaviour
        "buggy": "def add(a,b):\n return a-b",
    }
    prompts = [CandidatePrompt(k, "code", "gen", "p", metadata={"program": v})
               for k, v in progs.items()]
    ext = CodeBehaviorFingerprintExtractor(timeout=2.0)
    fps = {p.prompt_id: ext.extract(p, sentinels, Stub()) for p in prompts}
    cl = QuotientClusterer()
    D, ids = cl.pairwise_distance(fps)
    cells = cl.cluster(D, ids, 0.05, fps, {p.prompt_id: "gen" for p in prompts})
    cell_of = {pid: c.cell_id for c in cells for pid in c.prompt_ids}
    assert cell_of["ok_terse"] == cell_of["ok_verbose"]       # behaviourally equal
    assert cell_of["buggy"] != cell_of["ok_terse"]            # bug separates


def test_mvp_normalizers():
    from pqpo.fingerprints.normalizers import (Banking77Normalizer, GSM8KNormalizer,
                                               HoVerNormalizer)
    b = Banking77Normalizer(["card_arrival", "lost_or_stolen_card"])
    assert b.parse("Answer: card_arrival").normalized_answer == "card_arrival"
    assert b.parse("i think it's lost or stolen card").normalized_answer == "lost_or_stolen_card"
    g = GSM8KNormalizer()
    assert g.parse("...\n#### 42").normalized_answer == "42"
    assert g.parse("the answer is 18 dollars").normalized_answer == "18"
    h = HoVerNormalizer()
    assert h.parse("NOT SUPPORTED").normalized_answer == "NOT_SUPPORTED"
    assert h.parse("Answer: supported").normalized_answer == "SUPPORTED"


def test_loaders_synthetic_fallback():
    from pqpo.data.loaders import load_task
    t = load_task("gsm8k", n_sentinel=5, n_dev=10, n_test=10, source="synthetic")
    assert len(t.sentinels) == 5 and len(t.dev) == 10 and len(t.test) == 10
    assert all(u.target is None for u in t.sentinels)         # sentinels unlabelled
    assert all(d.target is not None for d in t.dev)


def test_sim_code_target_and_scorer():
    from pqpo.api.sim_code_target import build_sim_code_target
    from pqpo.data.datastructures import CandidatePrompt
    from pqpo.evaluation.code_scorer import CodeScorer, problems_to_examples
    sim, problems = build_sim_code_target(8, 20, np.random.default_rng(0))
    pids = list(sim.profiles.keys())
    ex = APIExecutor(ModelConfig("sim", "sim-code-target"), CostLedger(None),
                     APICache(None), "c", sim_target=sim)
    scorer = CodeScorer(ex, timeout=3.0)
    examples = problems_to_examples(problems, "code")
    # Distinct prompt text per candidate (as in any real pool) so the executor's
    # content-addressed cache does not collide the two.
    best = CandidatePrompt("p_best", "code", "gen", "high-skill instruction",
                           metadata={"behavior_profile_id": pids[-1]})
    worst = CandidatePrompt("p_worst", "code", "gen", "low-skill instruction",
                            metadata={"behavior_profile_id": pids[0]})
    best_acc = np.mean([scorer.score_one(best, e, "final_test") for e in examples])
    worst_acc = np.mean([scorer.score_one(worst, e, "final_test") for e in examples])
    assert best_acc > worst_acc           # higher-skill profile -> higher pass@1
    assert 0.0 <= worst_acc <= best_acc <= 1.0


def test_code_mvp_end_to_end():
    """Tiny PQPO-over-code run: fingerprints cluster the pool and PQPO returns a
    valid prompt whose held-out pass@1 is at least the pool mean."""
    from pqpo.api.sim_code_target import build_sim_code_target
    from pqpo.data.datasets import TaskSpec, build_candidate_pool
    from pqpo.evaluation.code_scorer import CodeScorer, problems_to_examples
    from pqpo.fingerprints.code_extractor import CodeBehaviorFingerprintExtractor
    from pqpo.quotient.clusterer import QuotientClusterer
    from pqpo.selectors import pqpo_fixed_pool
    from pqpo.selectors.base import SelectorContext

    rng = np.random.default_rng(0)
    sim, problems = build_sim_code_target(8, 60, rng)
    pids = list(sim.profiles.keys())
    sentinels, dev, test = problems[:8], problems[8:38], problems[38:58]
    ex = APIExecutor(ModelConfig("sim", "sim-code-target"), CostLedger(None),
                     APICache(None), "c", sim_target=sim)
    scorer = CodeScorer(ex, timeout=3.0)
    ext = CodeBehaviorFingerprintExtractor(timeout=3.0)
    spec = TaskSpec("code", "code", ["A"], "Write a correct function.")
    pool = build_candidate_pool(spec, 24, pids, rng, rarity=1.0)
    fps = {p.prompt_id: ext.extract(p, sentinels, ex) for p in pool}
    cl = QuotientClusterer(); D, ids = cl.pairwise_distance(fps)
    tau, _ = cl.choose_tau_by_stability(fps, n_boot=8, rng=np.random.default_rng(1))
    cells = cl.cluster(D, ids, tau, fps, {p.prompt_id: p.source_method for p in pool})
    assert 1 < len(cells) < len(pool)
    dev_ex = problems_to_examples(dev, "code")
    test_ex = problems_to_examples(test, "code")
    pool_by_id = {p.prompt_id: p for p in pool}

    def acc(p):
        return float(np.mean([scorer.score_one(p, e, "final_test") for e in test_ex]))

    # Average PQPO over a few seeds vs the pool mean (robust to single-seed noise).
    sm = {p.prompt_id: p.source_method for p in pool}
    pqpo_accs = []
    for seed in range(3):
        ctx = SelectorContext(pool, list(dev_ex), scorer, np.random.default_rng(seed),
                              192, fps, D, ids, source_methods=sm)
        res = pqpo_fixed_pool.run_pqpo_fixed_pool(ctx, tau=tau)
        assert res.selected_prompt_id in pool_by_id
        pqpo_accs.append(acc(pool_by_id[res.selected_prompt_id]))
    pool_mean = float(np.mean([acc(p) for p in pool]))
    assert np.mean(pqpo_accs) >= pool_mean    # PQPO beats the pool average in expectation


def test_holm_bonferroni_monotone():
    out = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.5})
    assert out["a"]["adjusted_p"] <= out["b"]["adjusted_p"] <= out["c"]["adjusted_p"]


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.binomial(1, 0.8, 400).astype(float)
    b = rng.binomial(1, 0.5, 400).astype(float)
    ci = paired_bootstrap_diff(a, b, n_boot=2000, rng=rng)
    assert ci.mean > 0 and ci.ci_low > 0 and ci.p_value < 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
