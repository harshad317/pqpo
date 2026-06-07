"""Closed-loop PQPO demo + non-scheduler evidence (Sec 2.6 / 2.7).

Runs closed-loop PQPO with phenotype-targeted mutation and reports the evidence
that distinguishes it from "Hyperband/ASHA + clustering":
  * target-cell hit rate (proposals land in the intended phenotype region)
  * new-cell entry rate
  * proposal distribution shift vs an UNGUIDED (random-target) proposal stream
    measured as total-variation distance over phenotype-cell occupancy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.api.cache import APICache
from pqpo.api.executor import APIExecutor, ModelConfig
from pqpo.api.sim_target import SimTarget, build_default_profiles
from pqpo.data.datasets import (build_candidate_pool, default_task_specs,
                                make_synthetic_examples)
from pqpo.evaluation.scorers import Scorer
from pqpo.fingerprints.extractor import BehaviorFingerprintExtractor
from pqpo.fingerprints.normalizers import ClassificationNormalizer
from pqpo.logging_utils.cost_ledger import CostLedger
from pqpo.logging_utils.progress import make_bar, metrics_table, rule
from pqpo.selectors.pqpo_closed_loop import (MutationTarget, ProposalModel,
                                             SimProposalModel,
                                             run_pqpo_closed_loop)


class UnguidedProposalModel(ProposalModel):
    """Baseline proposal stream: ignores the target, samples profiles uniformly."""
    def __init__(self, profile_ids):
        self.profile_ids = profile_ids; self._c = 0
    def generate_prompts(self, task_desc, target, rng):
        from pqpo.data.datastructures import CandidatePrompt
        out = []
        for _ in range(target.num_proposals):
            self._c += 1
            prof = self.profile_ids[int(rng.integers(0, len(self.profile_ids)))]
            t = f"{task_desc} [unguided:{self._c}] Respond as 'Answer: <label>'."
            tid = target.anchor_prompts[0].task_id if target.anchor_prompts else "task"
            out.append(CandidatePrompt(
                prompt_id=f"{tid}_ug_{self._c:05d}", task_id=tid,
                source_method="unguided", prompt_text=t, token_length=len(t.split()),
                metadata={"behavior_profile_id": prof}))
        return out


def total_variation(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def main():
    profiles = build_default_profiles(np.random.default_rng(7), 9)
    specs = default_task_specs(); spec = specs[0]
    ls = {s.task_id: s.label_set for s in specs}
    sim = SimTarget(profiles, ls)
    ex = APIExecutor(ModelConfig("sim", "sim-target"), CostLedger(None), APICache(None), "cl", sim)
    norm = ClassificationNormalizer(spec.label_set)
    extr = BehaviorFingerprintExtractor(norm); scorer = Scorer(ex, norm)
    rng = np.random.default_rng(0)
    seeds = build_candidate_pool(spec, 8, list(profiles.keys()), rng)
    sent = make_synthetic_examples(spec, 12, rng, "sentinel")
    for u in sent: u.target = None
    dev = make_synthetic_examples(spec, 80, rng, "dev")

    rule("Closed-loop PQPO — phenotype-targeted mutation")
    gbar = make_bar(20, "guided (phenotype-targeted)")
    guided = run_pqpo_closed_loop(
        spec.description, seeds, sent, dev, ex, extr, scorer,
        SimProposalModel(list(profiles.keys())), tau=0.10,
        total_labeled_budget=160, rng=np.random.default_rng(1),
        n_iters=20, proposals_per_iter=3, dev_per_eval=4,
        on_iter=lambda s: (gbar.update(1), gbar.set_postfix(
            cells=s["n_cells"], hit=f"{s['target_hit_rate']:.2f}",
            labeled=s["labeled"])))
    gbar.close()
    ubar = make_bar(20, "unguided (random targets)")
    unguided = run_pqpo_closed_loop(
        spec.description, seeds, sent, dev, ex, extr, scorer,
        UnguidedProposalModel(list(profiles.keys())), tau=0.10,
        total_labeled_budget=160, rng=np.random.default_rng(1),
        n_iters=20, proposals_per_iter=3, dev_per_eval=4,
        on_iter=lambda s: ubar.update(1))
    ubar.close()

    tv = total_variation(guided.occupancy, unguided.occupancy)
    metrics_table(
        "Closed-loop PQPO — non-scheduler evidence (Sec 2.7)",
        ["metric", "value"],
        [["target-cell hit rate (guided)", f"{guided.target_cell_hit_rate:.3f}"],
         ["new-cell entry rate (guided)", f"{guided.new_cell_entry_rate:.3f}"],
         ["#cells guided / unguided",
          f"{guided.metadata['n_cells']} / {unguided.metadata['n_cells']}"],
         ["proposal-distribution TV shift", f"{tv:.3f}"],
         ["labeled / sentinel calls (guided)",
          f"{guided.n_labeled_evals} / {guided.n_sentinel_calls}"],
         ["best prompt (guided)", guided.best_prompt_id]])
    print("  A high target-hit rate + non-zero TV shift show the mutation stream")
    print("  is steered by phenotype state, not a fixed schedule.")


if __name__ == "__main__":
    main()
