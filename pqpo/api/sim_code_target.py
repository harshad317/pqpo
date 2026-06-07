"""Deterministic simulated *code* target model.

Mirrors `sim_target.py` but emits real, executable Python programs instead of
classification labels, so the whole code path (execution -> test-pass fingerprint
-> clustering -> selection -> pass@1) runs offline and deterministically.

Mechanism
  Each synthetic problem implements ``f(x) = x + N`` for a hidden N, with unit
  tests over a few inputs (including 0). A prompt's behaviour profile decides, per
  problem, whether the model writes the correct program (probability = skill) or a
  profile-specific *buggy archetype* with a characteristic test-pass signature:

    * correct  -> def f(x): return x + N            passes [1,1,1]
    * const    -> def f(x): return N                passes only the x==0 test [1,0,0]
    * offbyone -> def f(x): return x + N + 1         passes nothing [0,0,0]
    * negate   -> def f(x): return -(x) + N          passes only x==0 [1,0,0]-ish

  Prompts sharing a profile id produce identical programs per problem -> identical
  test-pass fingerprints -> behavioural redundancy (cross-lineage when different
  source methods share a profile). Skill governs held-out pass@1.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data.datastructures import CandidatePrompt, TaskExample
from ..evaluation.code_exec import CodeProblem
from .sim_target import BehaviorProfile, _u01, approx_tokens, build_default_profiles

ARCHETYPES = ["const", "offbyone", "negate", "double"]


@dataclass
class SimCodeTarget:
    """Frozen deterministic simulated code generator."""
    profiles: dict[str, BehaviorProfile]
    problem_N: dict[str, int]                      # problem_id -> hidden N
    problem_archetype_inputs: dict[str, list[int]] = None

    model_provider = "sim"
    model_name = "sim-code-target"
    model_version = "v1"

    def _archetype_for(self, profile_id: str) -> str:
        idx = int(hashlib.sha256(profile_id.encode()).hexdigest()[:6], 16)
        return ARCHETYPES[idx % len(ARCHETYPES)]

    def generate(self, prompt: CandidatePrompt, example: TaskExample,
                 temperature: float = 0.0, max_output_tokens: int = 256,
                 seed: Optional[int] = None) -> tuple[str, int, int, float]:
        profile = self.profiles[prompt.metadata["behavior_profile_id"]]
        N = self.problem_N[example.example_id]
        key = f"{profile.profile_id}|{example.example_id}"

        if _u01("skill", key) < profile.skill:
            body = f"    return x + {N}"
        else:
            arch = self._archetype_for(profile.profile_id)
            if arch == "const":
                body = f"    return {N}"
            elif arch == "offbyone":
                body = f"    return x + {N + 1}"
            elif arch == "negate":
                body = f"    return -x + {N}"
            else:  # double
                body = f"    return 2 * x + {N}"
        # lightweight stylistic variation (comment) so prompt strings differ but
        # behaviour (the executed result) does not.
        code = f"```python\ndef f(x):\n{body}\n```"
        in_tok = approx_tokens(prompt.prompt_text + example.input_text)
        out_tok = approx_tokens(code)
        latency = profile.latency_base + 40.0 * _u01("lat", key)
        return code, in_tok, out_tok, latency


def build_synthetic_code_problems(n_problems: int, rng: np.random.Generator,
                                  task_id: str = "code") -> tuple[list[CodeProblem], dict[str, int]]:
    """Create n synthetic ``f(x)=x+N`` problems with tests over inputs incl. 0."""
    problems, problem_N = [], {}
    inputs = [0, 1, 2]
    for i in range(n_problems):
        N = int(rng.integers(1, 50))
        pid = f"{task_id}_p{i:04d}"
        tests = [f"assert f({x}) == {x + N}" for x in inputs]
        problems.append(CodeProblem(problem_id=pid,
                                    prompt=f"Implement f(x) that returns x plus {N}.",
                                    tests=tests, entry_point="f"))
        problem_N[pid] = N
    return problems, problem_N


def build_sim_code_target(n_profiles: int, n_problems: int,
                          rng: np.random.Generator) -> tuple[SimCodeTarget, list[CodeProblem]]:
    profiles = build_default_profiles(rng, n_profiles)
    problems, problem_N = build_synthetic_code_problems(n_problems, rng)
    return SimCodeTarget(profiles, problem_N), problems
