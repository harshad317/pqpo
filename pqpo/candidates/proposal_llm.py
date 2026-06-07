"""Proposal/reflection LLM layer for the candidate generators.

Two backends behind one interface:

  * SimProposalLLM  - deterministic; emits prompt text AND a hidden behaviour
    profile. Reflection/crossover move the profile UP the skill ordering with
    probability tied to feedback, so simulated GEPA/MIPROv2/CAPO pools exhibit
    realistic behavioural hill-climbing and, because all methods draw from the
    same profile bank, heavy CROSS-LINEAGE behavioural redundancy (the PQPO
    premise). Paraphrase stays in the same cell (pure redundancy).
  * LLMProposalLLM  - calls a billed provider via executor.complete and returns
    real text; no profile (the real target defines behaviour).

Every operation logs its meta-call cost (proposal/reflection) to the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..api.executor import APIExecutor


@dataclass
class ProposalOutput:
    text: str
    profile_id: Optional[str]
    call_id: str
    call_type: str


def _est_tokens(s: str) -> int:
    return max(1, len(s) // 4)


class ProposalLLM:
    def propose_initial(self, cfg, rng) -> ProposalOutput: ...
    def reflect_mutate(self, cfg, parent_text, parent_profile, feedback_acc, feedback, rng) -> ProposalOutput: ...
    def crossover(self, cfg, a_text, a_profile, b_text, b_profile, rng) -> ProposalOutput: ...
    def propose_instruction(self, cfg, demos_summary, demo_quality, rng) -> ProposalOutput: ...
    def paraphrase(self, cfg, parent_text, parent_profile, rng) -> ProposalOutput: ...


# --------------------------------------------------------------------------- #
# Simulated proposal model (profile-aware)
# --------------------------------------------------------------------------- #
class SimProposalLLM(ProposalLLM):
    def __init__(self, executor: APIExecutor, profile_ids: list[str]):
        # profile_ids must be ordered by ascending skill (build_default_profiles).
        self.executor = executor
        self.profile_ids = profile_ids
        self.n = len(profile_ids)

    def _idx(self, profile_id: Optional[str]) -> int:
        if profile_id in self.profile_ids:
            return self.profile_ids.index(profile_id)
        return self.n // 3  # default: low-mid

    def _log(self, cfg, in_text, out_text, call_type) -> str:
        tr = self.executor.record_meta_call(
            call_type=call_type, task_id=cfg.task_id, input_text=in_text,
            output_text=out_text, input_tokens=_est_tokens(in_text),
            output_tokens=_est_tokens(out_text), latency_ms=120.0)
        return tr.call_id

    def propose_initial(self, cfg, rng) -> ProposalOutput:
        idx = int(rng.integers(0, max(1, self.n // 3)))
        text = (f"You are solving a task. {cfg.task_description} "
                f"Think briefly, then answer as 'Answer: <label>'. [init:{int(rng.integers(1e6))}]")
        cid = self._log(cfg, f"INIT {cfg.task_description}", text, "proposal")
        return ProposalOutput(text, self.profile_ids[idx], cid, "proposal")

    def reflect_mutate(self, cfg, parent_text, parent_profile, feedback_acc, feedback, rng):
        pidx = self._idx(parent_profile)
        room = 1.0 - feedback_acc                       # poor accuracy -> more headroom
        p_improve = 0.45 + 0.45 * room
        step = 1 if rng.random() < p_improve else 0
        if rng.random() < 0.12:                          # occasional regression/noise
            step -= 1
        cidx = int(np.clip(pidx + step, 0, self.n - 1))
        text = (f"{parent_text.split(' [')[0]} "
                f"Focus on the cases you previously got wrong; be precise about the "
                f"label definition. Answer as 'Answer: <label>'. [refl:{int(rng.integers(1e6))}]")
        in_text = f"REFLECT feedback={feedback}"
        cid = self._log(cfg, in_text, text, "reflection")
        return ProposalOutput(text[:cfg.max_prompt_tokens * 6], self.profile_ids[cidx], cid, "reflection")

    def crossover(self, cfg, a_text, a_profile, b_text, b_profile, rng):
        hi = max(self._idx(a_profile), self._idx(b_profile))
        cidx = int(np.clip(hi + (1 if rng.random() < 0.4 else 0), 0, self.n - 1))
        text = (f"You are solving a task. {cfg.task_description} "
                f"Combine: state the label definition, check edge cases, then answer "
                f"as 'Answer: <label>'. [xover:{int(rng.integers(1e6))}]")
        cid = self._log(cfg, "CROSSOVER", text, "proposal")
        return ProposalOutput(text, self.profile_ids[cidx], cid, "proposal")

    def propose_instruction(self, cfg, demos_summary, demo_quality, rng):
        # MIPROv2: instruction grounded on bootstrapped demos; better demos -> better.
        base = int(round(demo_quality * (self.n - 1)))
        cidx = int(np.clip(base + int(rng.integers(-1, 2)), 0, self.n - 1))
        text = (f"{cfg.task_description} Use the following style demonstrated by "
                f"examples to guide your answer. Answer as 'Answer: <label>'. "
                f"[instr:{int(rng.integers(1e6))}]")
        cid = self._log(cfg, f"PROPOSE_INSTR demos={demos_summary[:60]}", text, "proposal")
        return ProposalOutput(text, self.profile_ids[cidx], cid, "proposal")

    def paraphrase(self, cfg, parent_text, parent_profile, rng):
        # lateral move: same behaviour cell (deliberate redundancy)
        text = (f"{parent_text.split(' [')[0]} Respond concisely as 'Answer: <label>'. "
                f"[para:{int(rng.integers(1e6))}]")
        cid = self._log(cfg, "PARAPHRASE", text, "proposal")
        return ProposalOutput(text, parent_profile or self.profile_ids[self._idx(parent_profile)],
                              cid, "proposal")


# --------------------------------------------------------------------------- #
# Real proposal model (billed provider)
# --------------------------------------------------------------------------- #
class LLMProposalLLM(ProposalLLM):
    """Calls a billed provider for real text. profile is always None."""

    SYS = ("You are an expert prompt engineer. Improve task instructions so a "
           "downstream model answers correctly. Output ONLY the new instruction.")

    def __init__(self, executor: APIExecutor):
        self.executor = executor

    def _call(self, cfg, user, call_type, rng) -> ProposalOutput:
        tr = self.executor.complete(self.SYS, user, call_type=call_type,
                                    task_id=cfg.task_id, temperature=0.8,
                                    seed=int(rng.integers(1e9)))
        return ProposalOutput(tr.output_text.strip(), None, tr.call_id, call_type)

    def propose_initial(self, cfg, rng):
        return self._call(cfg, f"Write an initial instruction for: {cfg.task_description}",
                          "proposal", rng)

    def reflect_mutate(self, cfg, parent_text, parent_profile, feedback_acc, feedback, rng):
        user = (f"Current instruction:\n{parent_text}\n\nObserved performance: {feedback}\n"
                f"Revise the instruction to fix the failures. Output only the instruction.")
        return self._call(cfg, user, "reflection", rng)

    def crossover(self, cfg, a_text, a_profile, b_text, b_profile, rng):
        user = (f"Merge the strengths of these two instructions into one:\n"
                f"A:\n{a_text}\nB:\n{b_text}\nOutput only the merged instruction.")
        return self._call(cfg, user, "proposal", rng)

    def propose_instruction(self, cfg, demos_summary, demo_quality, rng):
        user = (f"Task: {cfg.task_description}\nExample demonstrations:\n{demos_summary}\n"
                f"Write an instruction that elicits this behaviour. Output only the instruction.")
        return self._call(cfg, user, "proposal", rng)

    def paraphrase(self, cfg, parent_text, parent_profile, rng):
        user = f"Paraphrase without changing meaning:\n{parent_text}\nOutput only the instruction."
        return self._call(cfg, user, "proposal", rng)
