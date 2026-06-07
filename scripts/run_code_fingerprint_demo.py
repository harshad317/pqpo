"""Rich code-behaviour fingerprint demo.

Shows that PQPO's phenotype clustering recovers *behavioural* equivalence for code
from the test-pass vector alone: lexically different programs that pass the same
tests land in the same cell, while a program that fails (wrong / runtime / syntax)
lands in a different cell. This is the regime where the behavioural quotient is
strictly richer than lexical / embedding clustering.

No LLM is needed: a stub executor returns each candidate's stored program, so the
demo is deterministic and offline. The same CodeBehaviorFingerprintExtractor is
used with a real APIExecutor for MBPP+/HumanEval+/LiveCodeBench.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqpo.data.datastructures import APITrace, CandidatePrompt
from pqpo.evaluation.code_exec import CodeProblem
from pqpo.fingerprints.code_extractor import CodeBehaviorFingerprintExtractor
from pqpo.fingerprints.distances import pairwise_distance_matrix
from pqpo.logging_utils.progress import metrics_table, rule, titer
from pqpo.quotient.clusterer import QuotientClusterer


class StubCodeExecutor:
    """Returns the program stored in prompt.metadata['program'] as the output."""
    def run_prompt_on_example(self, prompt, example, call_type="sentinel"):
        program = prompt.metadata["program"]
        return APITrace(
            call_id="c", run_id="demo", model_provider="stub", model_name="stub",
            task_id=prompt.task_id, call_type=call_type, input_text_hash="h",
            full_input_text=example.input_text, output_text=program,
            temperature=0.0, max_output_tokens=256, input_tokens=10,
            output_tokens=max(1, len(program) // 4), total_tokens=20,
            latency_ms=50.0, dollar_cost=0.0, cache_hit=False, request_hash="r")


def main():
    # Sentinel panel: 3 small problems, each with several unit tests.
    sentinels = [
        CodeProblem("add", "add(a,b)", ["assert add(2,3)==5", "assert add(-1,1)==0",
                                        "assert add(0,0)==0"]),
        CodeProblem("mul", "mul(a,b)", ["assert mul(2,3)==6", "assert mul(0,9)==0",
                                        "assert mul(-2,3)==-6"]),
        CodeProblem("mx", "mx(xs)", ["assert mx([1,2,3])==3", "assert mx([5])==5",
                                     "assert mx([-1,-5])==-1"]),
    ]
    # Candidate programs (a program must define add, mul, mx). Several correct ones
    # are lexically different but behaviourally identical; others are buggy.
    correct_a = ("def add(a,b):\n return a+b\n"
                 "def mul(a,b):\n return a*b\n"
                 "def mx(xs):\n return max(xs)\n")
    correct_b = ("def add(a,b):\n s=a+b\n return s\n"               # same behaviour
                 "def mul(a,b):\n p=0\n for _ in range(abs(b)):\n  p+=a\n return p if b>=0 else -p\n"
                 "def mx(xs):\n m=xs[0]\n for v in xs:\n  m=v if v>m else m\n return m\n")
    wrong_mul = ("def add(a,b):\n return a+b\n"
                 "def mul(a,b):\n return a+b\n"                       # bug: + not *
                 "def mx(xs):\n return max(xs)\n")
    wrong_mx = ("def add(a,b):\n return a+b\n"
                "def mul(a,b):\n return a*b\n"
                "def mx(xs):\n return min(xs)\n")                     # bug: min not max
    runtime_bug = ("def add(a,b):\n return a+b\n"
                   "def mul(a,b):\n return a*b\n"
                   "def mx(xs):\n return xs[99]\n")                   # IndexError
    syntax_bug = "def add(a,b)\n return a+b\n"                        # no colon

    programs = {
        "p_correct_1": correct_a, "p_correct_2": correct_b, "p_correct_3": correct_a,
        "p_wrong_mul": wrong_mul, "p_wrong_mx": wrong_mx,
        "p_runtime": runtime_bug, "p_syntax": syntax_bug,
    }
    prompts = [CandidatePrompt(pid, "code", "gen", f"<prompt for {pid}>",
                               metadata={"program": prog})
               for pid, prog in programs.items()]

    rule("Rich code-behaviour fingerprint demo (test-pass vectors)")
    extractor = CodeBehaviorFingerprintExtractor(timeout=3.0)
    stub = StubCodeExecutor()
    fps = {}
    for p in titer(prompts, desc="executing candidates", total=len(prompts)):
        fps[p.prompt_id] = extractor.extract(p, sentinels, stub)

    # Cluster by behavioural fingerprint.
    cl = QuotientClusterer()
    D, ids = cl.pairwise_distance(fps)
    cells = cl.cluster(D, ids, tau=0.05, fingerprints=fps,
                       source_methods={p.prompt_id: "gen" for p in prompts})

    rows = []
    for p in prompts:
        fp = fps[p.prompt_id]
        passv = "".join("1" if a == "pass" else "0" for a in fp.normalized_answers)
        cell = next(c.cell_id for c in cells if p.prompt_id in c.prompt_ids)
        rows.append([p.prompt_id, passv, cell])
    metrics_table("Per-candidate test-pass vector and assigned phenotype cell",
                  ["candidate", "test-pass vector (9 tests)", "phenotype cell"], rows)

    cell_members = {c.cell_id: c.prompt_ids for c in cells}
    metrics_table("Phenotype cells (behavioural equivalence classes)",
                  ["cell", "members"],
                  [[cid, ", ".join(m)] for cid, m in cell_members.items()])
    print(f"\n  {len(prompts)} programs -> {len(cells)} behavioural cells. "
          "Lexically-different correct programs share a cell; each bug class separates.")


if __name__ == "__main__":
    main()
