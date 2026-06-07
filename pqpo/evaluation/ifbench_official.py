"""Bridge to the official IFBench / IFEval instruction verifiers.

IFBench (allenai/IFBench_test) scores instruction following with 58 out-of-domain
*verifiable* constraints, implemented in the allenai/IFBench repo (IFEval-style
``instructions_registry``). To score the FULL benchmark correctly you must use
those official verifiers — this module imports them if available and exposes a
uniform ``check_following_official(...)`` returning the per-instruction
satisfaction vector.

Install for a real run (one-time):

    git clone https://github.com/allenai/IFBench
    # add its instruction-following eval module to PYTHONPATH, e.g.:
    export PYTHONPATH="$PYTHONPATH:/path/to/IFBench/IFBench"   # dir with instructions_registry.py

If the registry is not importable, callers fall back to the built-in verifier
subset in ``ifbench.py`` (approximate; only covers a handful of constraint types)
and should treat scores as a lower-bound proxy, not official IFBench numbers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def load_registry():
    """Return the official instruction registry module, or None if unavailable.

    Tries the common import paths used by the IFEval/IFBench harnesses."""
    candidates = [
        "instruction_following_eval.instructions_registry",
        "instructions_registry",
        "ifbench.instructions_registry",
        "IFBench.instructions_registry",
        "eval.instruction_following_eval.instructions_registry",
    ]
    for path in candidates:
        try:
            mod = __import__(path, fromlist=["INSTRUCTION_DICT"])
            if hasattr(mod, "INSTRUCTION_DICT"):
                return mod
        except Exception:
            continue
    return None


def check_following_official(registry, prompt: str, instruction_id_list: list[str],
                             kwargs_list: list[dict], response: str) -> Optional[list[bool]]:
    """Per-instruction satisfaction using the official verifiers.

    Mirrors the canonical IFEval/IFBench harness loop: build each instruction from
    its kwargs (and the prompt when the instruction requires it), then
    check_following on the model response. Returns None if registry is None."""
    if registry is None:
        return None
    results = []
    for iid, kw in zip(instruction_id_list, kwargs_list or [{}] * len(instruction_id_list)):
        try:
            cls = registry.INSTRUCTION_DICT[iid]
            inst = cls(iid)
            clean = {k: v for k, v in (kw or {}).items() if v is not None}
            inst.build_description(**clean)
            args = inst.get_instruction_args() if hasattr(inst, "get_instruction_args") else None
            if args and "prompt" in args:
                inst.build_description(prompt=prompt)
            ok = bool(response.strip()) and bool(inst.check_following(response))
        except Exception:
            ok = False
        results.append(ok)
    return results


def official_available() -> bool:
    return load_registry() is not None
