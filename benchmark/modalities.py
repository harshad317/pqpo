"""Modality glue for the unified runner.

Three behavioural modalities, each providing a uniform interface (sentinels, dev
/test examples, a scorer, and a per-prompt fingerprint function):

  * label   - classification / reasoning / verification (Banking77, GSM8K, HoVer).
              Fingerprint = normalized-answer vector over the sentinel panel.
  * code    - MBPP/HumanEval/LiveCodeBench. Fingerprint = test-pass vector.
  * ifbench - instruction following. Fingerprint = per-constraint satisfaction.

For ``--source sim`` a deterministic simulated target makes everything run offline;
for ``--source hf`` a billed provider + the real dataset loaders are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pqpo.api.cache import APICache
from pqpo.api.executor import APIExecutor
from pqpo.api.sim_target import SimTarget, build_default_profiles
from pqpo.api.sim_code_target import build_sim_code_target
from pqpo.cli import build_model_config
from pqpo.data.loaders import load_task
from pqpo.data.code_loaders import load_code_task
from pqpo.evaluation.scorers import Scorer
from pqpo.evaluation.code_scorer import CodeScorer, problems_to_examples
from pqpo.evaluation.ifbench import (IFBehaviorFingerprintExtractor, IFScorer,
                                     build_sim_if_target, if_items_to_examples,
                                     load_ifbench)
from pqpo.fingerprints.code_extractor import CodeBehaviorFingerprintExtractor
from pqpo.fingerprints.extractor import BehaviorFingerprintExtractor
from pqpo.fingerprints.normalizers import ClassificationNormalizer

# benchmark name -> modality
BENCHMARKS = {
    "banking77": "label", "gsm8k": "label", "hover": "label",
    "ifbench": "ifbench",
    "mbpp": "code", "humaneval": "code", "livecodebench": "code",
}


@dataclass
class Modality:
    name: str
    modality: str
    executor: APIExecutor
    sentinels: list
    dev_ex: list
    test_ex: list
    scorer: object
    fingerprint: Callable
    profile_ids: list
    label_set: list
    description: str
    metric_name: str


def _profiles_or_placeholder(simulated, n_profiles, rng):
    if simulated:
        return build_default_profiles(rng, n_profiles), None
    return {}, None


def setup_label(name, args, sizes, ledger, rng) -> Modality:
    simulated = args.source == "sim"
    load = load_task(name, n_sentinel=sizes["n_sentinel"], n_dev=sizes["n_dev"],
                     n_test=sizes["n_test"], seed=0,
                     source=("synthetic" if simulated else "hf"))
    label_set = load.label_set or ["A", "B", "C", "D"]
    # Sim emits 'Answer: <label>'; use a classification normalizer over the label
    # set. Real (hf) runs use the dataset's own normalizer.
    normalizer = ClassificationNormalizer(label_set) if simulated else load.normalizer
    extractor = BehaviorFingerprintExtractor(normalizer)

    if simulated:
        profiles = build_default_profiles(rng, sizes["n_profiles"])
        sim = SimTarget(profiles, {load.spec.task_id: label_set})
        model_cfg = build_model_config(args, simulated=True, sim_model="sim-target")
        ex = APIExecutor(model_cfg, ledger, APICache(None), name, sim_target=sim)
        profile_ids = list(profiles.keys())
    else:
        model_cfg = build_model_config(args, simulated=False)
        ex = APIExecutor(model_cfg, ledger, APICache(None), name)
        profile_ids = []

    scorer = Scorer(ex, normalizer)
    sentinels = load.sentinels

    def fingerprint(prompt):
        souts = [extractor.parse_sentinel(
            ex.run_prompt_on_example(prompt, u, "sentinel"), f"s{j}")
            for j, u in enumerate(sentinels)]
        return extractor.extract(prompt.prompt_id, souts)

    return Modality(name, "label", ex, sentinels, load.dev, load.test, scorer,
                    fingerprint, profile_ids, label_set, load.spec.description,
                    "accuracy")


def setup_code(name, args, sizes, ledger, rng) -> Modality:
    simulated = args.source == "sim"
    extractor = CodeBehaviorFingerprintExtractor(timeout=3.0)
    if simulated:
        n_problems = sizes["n_sentinel"] + sizes["n_dev"] + sizes["n_test"] + 20
        sim, problems = build_sim_code_target(sizes["n_profiles"], n_problems, rng)
        model_cfg = build_model_config(args, simulated=True, sim_model="sim-code-target")
        ex = APIExecutor(model_cfg, ledger, APICache(None), name, sim_target=sim)
        profile_ids = list(sim.profiles.keys())
    else:
        load = load_code_task(name, n_sentinel=sizes["n_sentinel"], n_dev=sizes["n_dev"],
                              n_test=sizes["n_test"], seed=0, source="hf")
        problems = load.sentinels + load.dev + load.test
        model_cfg = build_model_config(args, simulated=False)
        ex = APIExecutor(model_cfg, ledger, APICache(None), name)
        profile_ids = []
    s, d, t = (problems[:sizes["n_sentinel"]],
               problems[sizes["n_sentinel"]:sizes["n_sentinel"] + sizes["n_dev"]],
               problems[sizes["n_sentinel"] + sizes["n_dev"]:
                        sizes["n_sentinel"] + sizes["n_dev"] + sizes["n_test"]])
    scorer = CodeScorer(ex, timeout=3.0)
    fingerprint = lambda prompt: extractor.extract(prompt, s, ex)
    return Modality(name, "code", ex, s, problems_to_examples(d, name),
                    problems_to_examples(t, name), scorer, fingerprint,
                    profile_ids, ["A"], "Write a correct Python function.", "pass@1")


def setup_ifbench(name, args, sizes, ledger, rng) -> Modality:
    simulated = args.source == "sim"
    extractor = IFBehaviorFingerprintExtractor(use_shape=getattr(args, "if_shape", True))
    if simulated:
        n_items = sizes["n_sentinel"] + sizes["n_dev"] + sizes["n_test"] + 20
        sim, items = build_sim_if_target(sizes["n_profiles"], n_items, rng,
                                         sizes.get("constraints", 4))
        model_cfg = build_model_config(args, simulated=True, sim_model="sim-if-target")
        ex = APIExecutor(model_cfg, ledger, APICache(None), name, sim_target=sim)
        profile_ids = list(sim.profiles.keys())
        s = items[:sizes["n_sentinel"]]
        d = items[sizes["n_sentinel"]:sizes["n_sentinel"] + sizes["n_dev"]]
        t = items[sizes["n_sentinel"] + sizes["n_dev"]:
                  sizes["n_sentinel"] + sizes["n_dev"] + sizes["n_test"]]
    else:
        s, d, t = load_ifbench(sizes["n_sentinel"], sizes["n_dev"], sizes["n_test"],
                               seed=0, source="hf")
        model_cfg = build_model_config(args, simulated=False)
        ex = APIExecutor(model_cfg, ledger, APICache(None), name)
        profile_ids = []
    scorer = IFScorer(ex, strict=True)
    fingerprint = lambda prompt: extractor.extract(prompt, s, ex)
    return Modality(name, "ifbench", ex, s, if_items_to_examples(d),
                    if_items_to_examples(t), scorer, fingerprint, profile_ids,
                    ["A"], "Follow every instruction exactly.", "strict acc")


def setup_modality(name, args, sizes, ledger, rng) -> Modality:
    modality = BENCHMARKS[name]
    if modality == "label":
        return setup_label(name, args, sizes, ledger, rng)
    if modality == "code":
        return setup_code(name, args, sizes, ledger, rng)
    return setup_ifbench(name, args, sizes, ledger, rng)
