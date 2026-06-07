"""Bridge to the official IFBench / IFEval instruction verifiers.

IFBench (allenai/IFBench_test) scores instruction following with 58 out-of-domain
*verifiable* constraints, implemented in the allenai/IFBench repo (IFEval-style
``instructions_registry``). To score the FULL benchmark correctly you must use
those official verifiers — this module imports them if available and exposes a
uniform ``check_following_official(...)`` returning the per-instruction
satisfaction vector.

Install for a real run (one-time):

    git clone https://github.com/allenai/IFBench
    pip install -r IFBench/requirements.txt        # checker deps (spacy, nltk, ...)
    export IFBENCH_DIR=/abs/path/to/IFBench        # the REPO ROOT (where
                                                   # instructions_registry.py lives)

``instructions_registry.py`` is at the repo ROOT and does ``import instructions``
(also root), so the repo root — not a subdirectory — must be importable. Setting
``IFBENCH_DIR`` is the most robust option; ``export PYTHONPATH=$PYTHONPATH:/abs/
path/to/IFBench`` also works.

If the registry is not importable, callers fall back to the built-in verifier
subset in ``ifbench.py`` (approximate; only covers a handful of constraint types)
and should treat scores as a lower-bound proxy, not official IFBench numbers.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def load_registry():
    """Return the official instruction registry module, or None if unavailable.

    Honors $IFBENCH_DIR (the cloned repo root) by inserting it on sys.path, then
    tries the import paths used by the IFBench / IFEval harnesses."""
    d = os.environ.get("IFBENCH_DIR")
    if d and os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
    candidates = [
        "instructions_registry",                       # IFBench repo root
        "instruction_following_eval.instructions_registry",
        "ifbench.instructions_registry",
        "IFBench.instructions_registry",
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
