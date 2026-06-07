"""Code-benchmark loaders: MBPP/MBPP+, HumanEval/HumanEval+, LiveCodeBench.

Each loader returns a ``LoadedCodeTask`` with sentinel/dev/test splits of
``CodeProblem`` (prompt + unit tests + entry point). HuggingFace ``datasets`` is
optional; with ``source="synthetic"`` (or when offline) the loaders fall back to
the deterministic synthetic ``f(x)=x+N`` problem family so the whole code path
runs without network.

Reproducibility: LiveCodeBench is a *moving, contamination-controlled* benchmark —
pin a release_version (date window) and record it in the manifest, or results are
not reproducible.

Test-harness note: real datasets ship tests in varying shapes (assert lists,
``check(candidate)`` functions, stdin/stdout). The adapters below normalise to a
list of ``assert`` strings that reference the entry point; extend
``_normalise_tests`` per dataset as needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..evaluation.code_exec import CodeProblem


@dataclass
class LoadedCodeTask:
    task_id: str
    sentinels: list[CodeProblem]
    dev: list[CodeProblem]
    test: list[CodeProblem]
    metadata: dict


def _have_datasets() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except Exception:
        return False


def _split_problems(problems, n_sentinel, n_dev, n_test, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(problems))
    take = lambda a, b: [problems[i] for i in idx[a:b]]
    return (take(0, n_sentinel),
            take(n_sentinel, n_sentinel + n_dev),
            take(n_sentinel + n_dev, n_sentinel + n_dev + n_test))


def _synthetic(n, seed, task_id):
    from ..api.sim_code_target import build_synthetic_code_problems
    probs, _ = build_synthetic_code_problems(n, np.random.default_rng(seed), task_id)
    return probs


# --------------------------------------------------------------------------- #
def load_mbpp(n_sentinel=12, n_dev=60, n_test=100, seed=0, source="hf",
              plus=True) -> LoadedCodeTask:
    task_id = "mbpp_plus" if plus else "mbpp"
    if source == "hf" and _have_datasets():
        from datasets import load_dataset
        ds = load_dataset("mbpp", "sanitized", split="test")
        problems = []
        for r in ds:
            entry = _entry_from_code(r.get("code", ""))
            tests = _normalise_tests(r.get("test_list", []), entry)
            if tests:
                problems.append(CodeProblem(
                    problem_id=f"mbpp_{r['task_id']}", prompt=r["prompt"],
                    tests=tests, entry_point=entry,
                    setup="\n".join(r.get("test_imports", []))))
    else:
        problems = _synthetic(n_sentinel + n_dev + n_test + 20, seed, task_id)
    s, d, t = _split_problems(problems, n_sentinel, n_dev, n_test, seed)
    return LoadedCodeTask(task_id, s, d, t, {"plus": plus, "source": source})


def load_humaneval(n_sentinel=12, n_dev=40, n_test=80, seed=0, source="hf",
                   plus=True) -> LoadedCodeTask:
    task_id = "humaneval_plus" if plus else "humaneval"
    if source == "hf" and _have_datasets():
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        problems = []
        for r in ds:
            # HumanEval ships a check(candidate) harness + entry_point.
            tests = [r["test"] + f"\ncheck({r['entry_point']})"]
            problems.append(CodeProblem(
                problem_id=r["task_id"].replace("/", "_"), prompt=r["prompt"],
                tests=tests, entry_point=r["entry_point"]))
    else:
        problems = _synthetic(n_sentinel + n_dev + n_test + 20, seed, task_id)
    s, d, t = _split_problems(problems, n_sentinel, n_dev, n_test, seed)
    return LoadedCodeTask(task_id, s, d, t, {"plus": plus, "source": source})


def load_livecodebench(n_sentinel=12, n_dev=40, n_test=80, seed=0, source="hf",
                       release_version: str = "release_v5") -> LoadedCodeTask:
    """LiveCodeBench: pin `release_version` (a dated window) for reproducibility."""
    if source == "hf" and _have_datasets():
        from datasets import load_dataset
        # LiveCodeBench code-generation split; field names vary by release.
        ds = load_dataset("livecodebench/code_generation_lite",
                          version_tag=release_version, split="test",
                          trust_remote_code=True)
        problems = []
        for i, r in enumerate(ds):
            entry = r.get("entry_point") or _entry_from_code(r.get("starter_code", ""))
            tests = _normalise_tests(r.get("public_test_cases", []), entry)
            if tests:
                problems.append(CodeProblem(
                    problem_id=f"lcb_{i}", prompt=r.get("question_content", ""),
                    tests=tests, entry_point=entry))
    else:
        problems = _synthetic(n_sentinel + n_dev + n_test + 20, seed, "livecodebench")
    s, d, t = _split_problems(problems, n_sentinel, n_dev, n_test, seed)
    return LoadedCodeTask("livecodebench", s, d, t,
                          {"release_version": release_version, "source": source})


CODE_LOADERS = {"mbpp": load_mbpp, "humaneval": load_humaneval,
                "livecodebench": load_livecodebench}


def load_code_task(name: str, **kw) -> LoadedCodeTask:
    if name not in CODE_LOADERS:
        raise ValueError(f"unknown code task '{name}'; choose {list(CODE_LOADERS)}")
    return CODE_LOADERS[name](**kw)


# --------------------------------------------------------------------------- #
def _entry_from_code(code: str) -> Optional[str]:
    import re
    m = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code or "")
    return m.group(1) if m else None


def _normalise_tests(raw_tests, entry: Optional[str]) -> list[str]:
    """Coerce dataset test specs into a list of `assert` strings."""
    out = []
    for t in raw_tests or []:
        if isinstance(t, str) and t.strip().startswith("assert"):
            out.append(t)
        elif isinstance(t, dict) and "input" in t and "output" in t and entry:
            out.append(f"assert {entry}({t['input']}) == {t['output']}")
    return out
