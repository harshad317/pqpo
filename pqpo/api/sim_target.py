"""Deterministic *simulated* black-box target model.

Why this exists
---------------
To validate the PQPO machinery end to end without billed API calls, we need a
target model whose behaviour has the structure the method assumes:

  * many distinct prompt *strings* induce the *same* observable behaviour
    (behavioural redundancy across generator lineages), and
  * a prompt's behaviour on unlabelled sentinels predicts its held-out score
    (so phenotype distance is mechanistically useful).

We realise this with latent **behaviour profiles**. Each candidate prompt is
assigned (by the pool builder, in metadata) a hidden profile id. The simulator
maps (profile, input) -> output deterministically at temperature 0. The
optimisation algorithms NEVER see the profile id; they only see outputs. So any
recovery of the profile structure is earned from black-box behaviour alone.

Swap-in contract
-----------------
A real provider implements the same `generate(...)` signature in
``executor.py``. Nothing downstream depends on the simulator.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from ..data.datastructures import CandidatePrompt, TaskExample


def _u01(*parts: str) -> float:
    """Deterministic pseudo-uniform in [0,1) from string parts."""
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(h[:13], 16) / float(16 ** 13)


def approx_tokens(text: str) -> int:
    # Cheap, stable token estimate (~4 chars/token), min 1.
    return max(1, round(len(text) / 4.0))


@dataclass
class BehaviorProfile:
    """A latent behaviour archetype. Held-out skill ~= `skill`."""
    profile_id: str
    skill: float            # P(emit correct label) on a typical example
    bias_label: str         # fallback label when not "skillful" on an example
    format_quality: float   # P(format valid)
    refusal_rate: float     # P(refuse)
    verbosity: int          # base output length in tokens
    latency_base: float     # base latency ms


class SimTarget:
    """A frozen, deterministic simulated target model."""

    model_provider = "sim"
    model_name = "sim-target"
    model_version = "v1"

    def __init__(self, profiles: dict[str, BehaviorProfile], label_sets: dict[str, list[str]]):
        # profiles keyed by profile_id; label_sets keyed by task_id
        self.profiles = profiles
        self.label_sets = label_sets

    # -- core black-box call ------------------------------------------------ #
    def generate(
        self,
        prompt: CandidatePrompt,
        example: TaskExample,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> tuple[str, int, int, float]:
        profile = self.profiles[prompt.metadata["behavior_profile_id"]]
        labels = self.label_sets[example.task_id]
        key = f"{profile.profile_id}|{example.example_id}"

        # Refusal short-circuit.
        if _u01("refuse", key) < profile.refusal_rate:
            out = "I'm sorry, but I can't help with that request."
            in_tok = approx_tokens(prompt.prompt_text + example.input_text)
            return out, in_tok, approx_tokens(out), profile.latency_base + 5 * _u01("lat", key)

        # Decide the (latent) answer this profile would give.
        gold = example.target
        if gold is not None and _u01("skill", key) < profile.skill:
            answer = gold
        else:
            # A deterministic, profile-specific "guess": usually the bias label,
            # occasionally a hash-chosen other label so distinct profiles differ.
            if _u01("useblias", key) < 0.7:
                answer = profile.bias_label
            else:
                answer = labels[int(_u01("guess", key) * len(labels)) % len(labels)]

        # Format validity: if invalid, wrap the answer so the normalizer fails.
        format_valid = _u01("fmt", key) < profile.format_quality
        if format_valid:
            body = f"Answer: {answer}"
        else:
            body = f"Well, it could be {answer} but I'm not fully sure honestly."

        # Verbosity padding (affects token/length fingerprint dims).
        pad_words = max(0, profile.verbosity + int(8 * _u01("verb", key)) - 6)
        padding = " ".join(["context"] * pad_words)
        out = (body + (" " + padding if padding else "")).strip()

        in_tok = approx_tokens(prompt.prompt_text + example.input_text)
        out_tok = approx_tokens(out)
        latency = profile.latency_base + 40.0 * _u01("lat", key)
        return out, in_tok, out_tok, latency


# --------------------------------------------------------------------------- #
# Factory: build a structured profile bank for the MVP tasks.
# --------------------------------------------------------------------------- #
def build_default_profiles(rng, n_profiles: int = 10) -> dict[str, BehaviorProfile]:
    """Create a bank of profiles with a spread of skill levels.

    The spread guarantees that (a) some profiles are clearly better than others
    (so selection matters) and (b) several share near-identical behaviour (so
    clustering compresses the pool)."""
    profiles: dict[str, BehaviorProfile] = {}
    # Skill grid: a crowd of mediocre profiles plus a clearly-best one at the END
    # (paired, in the pool builder, with rarity). This mirrors the paper's
    # hypothesis: behaviour is highly redundant, quality varies across cells, and
    # the best behavioural region is rare -- so covering one representative per
    # cell (PQPO) finds it where uniform sampling tends to miss it.
    skill_grid = [0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.62, 0.70, 0.90]
    bias_options = ["A", "B", "C"]
    for i in range(n_profiles):
        # Always interpolate across the FULL skill range so the spread is wide
        # regardless of n_profiles (the best/rarest cell stays clearly best).
        gi = 0 if n_profiles == 1 else round(i * (len(skill_grid) - 1) / (n_profiles - 1))
        skill = skill_grid[gi]
        profiles[f"prof_{i:02d}"] = BehaviorProfile(
            profile_id=f"prof_{i:02d}",
            skill=skill,
            bias_label=bias_options[i % len(bias_options)],
            format_quality=0.85 + 0.13 * rng.random(),
            refusal_rate=0.0,
            verbosity=6 + (i % 4) * 4,
            latency_base=80.0 + 15.0 * (i % 5),
        )
    return profiles
