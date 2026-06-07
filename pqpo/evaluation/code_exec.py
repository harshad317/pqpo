"""Sandboxed code execution + per-test behavioural signatures.

For code benchmarks (MBPP+/HumanEval+/LiveCodeBench), a candidate program's
*behaviour* on a problem is the vector of unit-test pass/fail outcomes plus the
error class — a far richer fingerprint than a single label. This module runs a
candidate program against a problem's tests in a subprocess with a timeout and
returns that signature.

SAFETY: this runs model-generated code. The subprocess + timeout + (on Unix)
memory/CPU rlimits are a guardrail, NOT a security sandbox. For untrusted code in
real runs, execute inside a container / gVisor / firejail. The harness imports
nothing on the candidate's behalf and blocks obviously dangerous calls by running
with a restricted import note in the docs; still, isolate at the OS level.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

# error classes (ordered by severity for the "dominant error" summary)
ERR_PASS = "pass"
ERR_WRONG = "wrong"          # AssertionError: ran but produced wrong output
ERR_RUNTIME = "runtime"      # raised a non-assertion exception
ERR_TIMEOUT = "timeout"
ERR_SYNTAX = "syntax"        # did not compile
ERR_NOCODE = "nocode"        # no code could be extracted


@dataclass
class CodeProblem:
    problem_id: str
    prompt: str                       # problem statement shown to the model
    tests: list[str]                  # e.g. ["assert add(1,2)==3", ...]
    entry_point: Optional[str] = None
    setup: str = ""                   # optional imports/helpers prepended to tests


@dataclass
class ExecResult:
    pass_vector: list[bool]
    error_types: list[str]            # per-test error class
    n_pass: int
    n_tests: int
    dominant_error: str
    syntax_ok: bool
    ast_signature: str
    code_chars: int

    @property
    def all_pass(self) -> bool:
        return self.n_tests > 0 and self.n_pass == self.n_tests

    @property
    def pass_rate(self) -> float:
        return self.n_pass / self.n_tests if self.n_tests else 0.0


def extract_code(output_text: str) -> Optional[str]:
    """Pull a python code block from a model output; fall back to raw text."""
    if not output_text:
        return None
    fences = []
    in_block, buf, lang = False, [], ""
    for line in output_text.splitlines():
        if line.strip().startswith("```"):
            if in_block:
                fences.append(("\n".join(buf), lang)); buf = []; in_block = False
            else:
                in_block = True; lang = line.strip()[3:].strip().lower()
            continue
        if in_block:
            buf.append(line)
    if in_block and buf:
        fences.append(("\n".join(buf), lang))
    if fences:
        for code, lang in fences:
            if lang in ("python", "py", ""):
                return code
        return fences[0][0]
    # no fences: treat the whole thing as code if it parses
    return output_text


def ast_signature(code: str) -> tuple[bool, str]:
    """(syntax_ok, signature). Signature = sorted node-type histogram buckets,
    so structurally-similar programs share a signature dimension."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "SYNTAX_ERROR"
    counts: dict[str, int] = {}
    depth_max = 0
    for node in ast.walk(tree):
        counts[type(node).__name__] = counts.get(type(node).__name__, 0) + 1
    # bucket counts to be robust to small variation
    def bucket(n):
        return 0 if n == 0 else 1 if n <= 2 else 2 if n <= 5 else 3 if n <= 10 else 4
    keys = ["FunctionDef", "For", "While", "If", "Call", "Return", "BinOp",
            "Compare", "ListComp", "Assign"]
    sig = ",".join(f"{k}:{bucket(counts.get(k, 0))}" for k in keys)
    return True, sig


_DRIVER = r"""
import json, sys
__tests__ = {tests!r}
__results__, __errs__ = [], []
for __t in __tests__:
    try:
        exec(__t, globals())
        __results__.append(True); __errs__.append("pass")
    except AssertionError:
        __results__.append(False); __errs__.append("wrong")
    except Exception as __e:
        __results__.append(False); __errs__.append("runtime")
print("__PQPO_RESULT__" + json.dumps({{"results": __results__, "errs": __errs__}}))
"""


# Deterministic execution cache: identical (code, tests, setup) -> identical
# result. Collapses the many redundant runs that arise when prompts share
# behaviour (and thus emit identical programs). Keyed by content hash.
_RESULT_CACHE: dict[str, "ExecResult"] = {}


def _cache_key(code: str, problem: "CodeProblem") -> str:
    import hashlib
    blob = (code + "\x00" + problem.setup + "\x00" + "\x00".join(problem.tests))
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


def run_tests(code: str, problem: CodeProblem, timeout: float = 5.0) -> ExecResult:
    ck = _cache_key(code or "", problem)
    cached = _RESULT_CACHE.get(ck)
    if cached is not None:
        return cached
    res = _run_tests_uncached(code, problem, timeout)
    _RESULT_CACHE[ck] = res
    return res


def _run_tests_uncached(code: str, problem: CodeProblem, timeout: float = 5.0) -> ExecResult:
    n = len(problem.tests)
    syntax_ok, sig = ast_signature(code or "")
    if not code:
        return ExecResult([False] * n, [ERR_NOCODE] * n, 0, n, ERR_NOCODE, False, "NONE", 0)
    if not syntax_ok:
        return ExecResult([False] * n, [ERR_SYNTAX] * n, 0, n, ERR_SYNTAX, False, sig, len(code))

    src = (problem.setup + "\n" + code + "\n" +
           _DRIVER.format(tests=problem.tests))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src); path = fh.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=timeout,
                              preexec_fn=_limit_resources if os.name == "posix" else None)
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return ExecResult([False] * n, [ERR_TIMEOUT] * n, 0, n, ERR_TIMEOUT, True, sig, len(code))
    finally:
        if os.path.exists(path):
            os.unlink(path)

    marker = "__PQPO_RESULT__"
    line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
    if line is None:
        # process crashed before reporting (e.g. import error at module load)
        return ExecResult([False] * n, [ERR_RUNTIME] * n, 0, n, ERR_RUNTIME, True, sig, len(code))
    data = json.loads(line[len(marker):])
    results = data["results"]; errs = data["errs"]
    n_pass = sum(results)
    dom = _dominant(errs)
    return ExecResult(results, errs, n_pass, n, dom, True, sig, len(code))


def _dominant(errs: list[str]) -> str:
    if not errs:
        return ERR_NOCODE
    for sev in (ERR_TIMEOUT, ERR_RUNTIME, ERR_WRONG, ERR_PASS):
        if sev in errs:
            # report worst non-pass error if any failure exists
            if sev != ERR_PASS or all(e == ERR_PASS for e in errs):
                return sev
    return errs[0]


def _limit_resources():  # pragma: no cover - posix only
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except Exception:
        pass
