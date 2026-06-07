"""Closed-loop PQPO optimiser (Sec 2.6 / 2.7).

Maintains a PhenotypeArchive and uses cell state to *target* mutation. Produces
the evidence required to refute "this is just Hyperband/ASHA with clustering":
new-cell entry rate, target-cell hit rate, proposal distribution shift vs an
unguided stream, and a selector-only-vs-targeted comparison.

Proposal model
  A ProposalModel turns a target behaviour summary into new candidate prompts. It
  may see prompt text, sentinel inputs, sentinel outputs and behavioural
  summaries, but NEVER labels/held-out scores (Sec 2.6). The SimProposalModel
  realises targeting against the simulated profile bank, using only observable
  behaviour (the anchor prompts' fingerprints) as its guide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..api.executor import APIExecutor
from ..data.datastructures import CandidatePrompt, TaskExample
from ..fingerprints.extractor import BehaviorFingerprintExtractor
from ..quotient.archive import PhenotypeArchive


# --------------------------------------------------------------------------- #
# Proposal models
# --------------------------------------------------------------------------- #
@dataclass
class MutationTarget:
    mode: str                       # empty_cell_exploration|high_utility_refinement|...
    anchor_prompts: list[CandidatePrompt]
    behavior_summary: dict
    intended_cell: Optional[str]
    num_proposals: int = 3


class ProposalModel:
    def generate_prompts(self, task_desc, target: MutationTarget, rng) -> list[CandidatePrompt]:  # pragma: no cover
        raise NotImplementedError


class SimProposalModel(ProposalModel):
    """Generates new candidates whose latent profile follows the target.

    For refinement it copies the anchor profile (lands in the same cell -> high
    target-hit rate). For exploration it samples an unused profile (-> new cell).
    The choice uses ONLY observed anchor behaviour, never labels."""

    def __init__(self, profile_ids: list[str]):
        self.profile_ids = profile_ids
        self._counter = 0

    def generate_prompts(self, task_desc, target: MutationTarget, rng) -> list[CandidatePrompt]:
        out = []
        for _ in range(target.num_proposals):
            if target.mode == "empty_cell_exploration":
                profile = self.profile_ids[int(rng.integers(0, len(self.profile_ids)))]
            else:  # refinement / boundary / repair: follow anchors
                anchor = target.anchor_prompts[0]
                profile = anchor.metadata["behavior_profile_id"]
            self._counter += 1
            text = (f"{task_desc} [cl_mut:{target.mode}:{self._counter}] "
                    f"Respond as 'Answer: <label>'.")
            out.append(CandidatePrompt(
                prompt_id=f"{anchor_task(target)}_clmut_{self._counter:05d}",
                task_id=anchor_task(target),
                source_method="pqpo_closed_loop",
                prompt_text=text,
                parent_prompt_id=target.anchor_prompts[0].prompt_id if target.anchor_prompts else None,
                token_length=len(text.split()),
                generation_seed=int(rng.integers(0, 1_000_000)),
                metadata={"behavior_profile_id": profile},
            ))
        return out


def anchor_task(target: MutationTarget) -> str:
    if target.anchor_prompts:
        return target.anchor_prompts[0].task_id
    return "task"


# --------------------------------------------------------------------------- #
# Closed-loop optimiser
# --------------------------------------------------------------------------- #
@dataclass
class ClosedLoopResult:
    best_prompt_id: str
    archive: PhenotypeArchive
    new_cell_entry_rate: float
    target_cell_hit_rate: float
    occupancy: dict
    n_labeled_evals: int
    n_sentinel_calls: int
    metadata: dict = field(default_factory=dict)


def run_sentinels_and_fingerprint(prompt, sentinels, executor, extractor):
    sent_outputs = []
    for j, u in enumerate(sentinels):
        tr = executor.run_prompt_on_example(prompt, u, call_type="sentinel")
        sent_outputs.append(extractor.parse_sentinel(tr, sentinel_id=f"s{j}"))
    return extractor.extract(prompt.prompt_id, sent_outputs)


def run_pqpo_closed_loop(
    task_desc: str,
    seed_prompts: list[CandidatePrompt],
    sentinels: list[TaskExample],
    dev_examples: list[TaskExample],
    executor: APIExecutor,
    extractor: BehaviorFingerprintExtractor,
    scorer,
    proposal_model: ProposalModel,
    tau: float,
    total_labeled_budget: int,
    rng: np.random.Generator,
    n_iters: int = 20,
    proposals_per_iter: int = 3,
    dev_per_eval: int = 4,
    weights: dict = None,
    on_iter=None,
) -> ClosedLoopResult:
    archive = PhenotypeArchive(tau=tau, weights=weights)
    registry: dict[str, CandidatePrompt] = {}    # prompt_id -> CandidatePrompt
    n_sentinel = 0
    n_labeled = 0

    # Seed the archive.
    for p in seed_prompts:
        registry[p.prompt_id] = p
        fp = run_sentinels_and_fingerprint(p, sentinels, executor, extractor)
        n_sentinel += len(sentinels)
        cid = archive.assign_or_create_cell(p.prompt_id, fp, p.prompt_text)
        # initial representative eval
        if n_labeled < total_labeled_budget:
            exs = dev_examples[:dev_per_eval]
            scs = [scorer.score_one(p, e) for e in exs]
            n_labeled += len(exs)
            archive.update_scores(cid, p.prompt_id, scs)

    for it in range(n_iters):
        if n_labeled >= total_labeled_budget:
            break
        archive.iteration = it
        target = choose_mutation_target(archive, registry, rng, proposals_per_iter)
        proposals = proposal_model.generate_prompts(task_desc, target, rng)
        for p_new in proposals:
            registry[p_new.prompt_id] = p_new
            fp = run_sentinels_and_fingerprint(p_new, sentinels, executor, extractor)
            n_sentinel += len(sentinels)
            cid = archive.assign_or_create_cell(
                p_new.prompt_id, fp, p_new.prompt_text, intended_cell=target.intended_cell)
            if should_evaluate(archive, cid, target.mode) and n_labeled < total_labeled_budget:
                exs = dev_examples[(n_labeled) % len(dev_examples):][:dev_per_eval] or dev_examples[:dev_per_eval]
                scs = [scorer.score_one(p_new, e) for e in exs]
                n_labeled += len(exs)
                archive.update_scores(cid, p_new.prompt_id, scs)
        if it % 5 == 0:
            archive.recluster_if_needed()
        if on_iter is not None:
            on_iter({"iter": it + 1, "n_cells": len(archive.cell_members),
                     "labeled": n_labeled, "sentinel": n_sentinel,
                     "target_hit_rate": archive.target_cell_hit_rate,
                     "new_cell_rate": archive.new_cell_entry_rate})

    return ClosedLoopResult(
        best_prompt_id=archive.best_prompt(),
        archive=archive,
        new_cell_entry_rate=archive.new_cell_entry_rate,
        target_cell_hit_rate=archive.target_cell_hit_rate,
        occupancy=archive.occupancy_distribution(),
        n_labeled_evals=n_labeled,
        n_sentinel_calls=n_sentinel,
        metadata={"n_cells": len(archive.cell_members), "tau": tau},
    )


def choose_mutation_target(archive: PhenotypeArchive, registry: dict,
                           rng, n_proposals) -> MutationTarget:
    modes = ["empty_cell_exploration", "high_utility_refinement",
             "boundary_split", "failure_repair"]
    # weight toward refinement of best cells, with exploration pressure
    scored = [(cid, st.mean_score, st.n_labeled_evals)
              for cid, st in archive.cell_state.items() if st.n_labeled_evals > 0]
    if scored and rng.random() < 0.6:
        best_cid = max(scored, key=lambda t: t[1])[0]
        anchors = [_anchor_prompt(archive, registry, best_cid)]
        return MutationTarget("high_utility_refinement", anchors,
                              {"mean_score": "hidden"}, best_cid, n_proposals)
    mode = modes[int(rng.integers(0, len(modes)))]
    if mode == "empty_cell_exploration" or not scored:
        anchor_cid = next(iter(archive.cell_members))
        return MutationTarget("empty_cell_exploration",
                              [_anchor_prompt(archive, registry, anchor_cid)],
                              {}, "__new__", n_proposals)
    cid = scored[int(rng.integers(0, len(scored)))][0]
    return MutationTarget(mode, [_anchor_prompt(archive, registry, cid)],
                          {}, cid, n_proposals)


def _anchor_prompt(archive, registry: dict, cid) -> CandidatePrompt:
    pid = archive.cell_members[cid][0]
    return registry[pid]


def should_evaluate(archive: PhenotypeArchive, cell_id: str, mode: str) -> bool:
    st = archive.cell_state.get(cell_id)
    if st is None or st.n_labeled_evals == 0:
        return True  # always evaluate a freshly occupied cell once
    # spend more on promising cells, less on redundant ones
    return mode in ("high_utility_refinement", "failure_repair")
