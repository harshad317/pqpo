"""Rich code-behaviour fingerprint extractor.

For a code task, a prompt's behaviour on the sentinel panel is the concatenation
of its generated programs' per-test outcomes. We flatten to *(problem, test)*
granularity so the fingerprint's answer dimension is the full test-pass vector
(plus error class per test) — not a single label. Two prompts whose generated
code passes/fails the same individual tests across the panel are behaviourally
equivalent and land in the same phenotype cell.

This reuses the standard BehaviorFingerprint / fingerprint_distance machinery:
  * normalized_answers[k]   = per-test outcome class (pass/wrong/runtime/...)
  * format_validity[k]      = code compiled (repeated per test of a problem)
  * refusal_flags[k]        = no code produced
  * parse_error_types[k]    = per-test error class (drives the shape term)
  * length / token / latency = problem-level, repeated to align lengths
"""
from __future__ import annotations

from typing import Optional

from ..data.datastructures import BehaviorFingerprint, CandidatePrompt, TaskExample
from ..evaluation.code_exec import (CodeProblem, ExecResult, extract_code,
                                    run_tests)
from .extractor import bucket_latency, bucket_output_length


class CodeBehaviorFingerprintExtractor:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def run_problem(self, prompt: CandidatePrompt, problem: CodeProblem,
                    executor) -> tuple[ExecResult, int, float]:
        ex = TaskExample(problem.problem_id, prompt.task_id, problem.prompt, None)
        trace = executor.run_prompt_on_example(prompt, ex, call_type="sentinel")
        code = extract_code(trace.output_text)
        result = run_tests(code or "", problem, timeout=self.timeout)
        return result, trace.output_tokens, trace.latency_ms

    def extract(self, prompt: CandidatePrompt, sentinel_problems: list[CodeProblem],
                executor) -> BehaviorFingerprint:
        norm_answers, fmt, refusal, errs = [], [], [], []
        len_buckets, tok_counts, lat_buckets = [], [], []
        for problem in sentinel_problems:
            result, out_tokens, latency = self.run_problem(prompt, problem, executor)
            for k in range(result.n_tests):
                # Behavioural equivalence for code = same per-test PASS/FAIL pattern
                # (answer dim) AND same per-test error class (shape dim). Verbosity
                # is deliberately excluded: length/token/latency are held constant
                # so two programs that pass the same tests cluster together
                # regardless of how they are written.
                norm_answers.append("pass" if result.pass_vector[k] else "fail")
                fmt.append(result.syntax_ok)
                refusal.append(result.dominant_error == "nocode")
                errs.append(result.error_types[k])             # pass/wrong/runtime/...
                len_buckets.append(0)
                tok_counts.append(0)
                lat_buckets.append(0)
        return BehaviorFingerprint(
            prompt_id=prompt.prompt_id, task_id=prompt.task_id,
            normalized_answers=norm_answers, format_validity=fmt,
            refusal_flags=refusal, parse_error_types=errs,
            output_length_buckets=len_buckets, token_counts=tok_counts,
            latency_buckets=lat_buckets)


def score_code_on_problem(prompt: CandidatePrompt, problem: CodeProblem,
                          executor, timeout: float = 5.0) -> float:
    """pass@1-style score on one problem (1.0 iff all tests pass)."""
    ex = TaskExample(problem.problem_id, prompt.task_id, problem.prompt, None)
    trace = executor.run_prompt_on_example(prompt, ex, call_type="selector_dev")
    result = run_tests(extract_code(trace.output_text) or "", problem, timeout)
    return 1.0 if result.all_pass else 0.0
