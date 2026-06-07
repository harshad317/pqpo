"""pass@1 scorer for code tasks (duck-compatible with evaluation.Scorer).

Reuses the APIExecutor (so cost is logged) to generate a program for a problem,
then executes it against the problem's unit tests. Score = 1.0 iff all tests pass.
The CodeProblem is carried on ``example.metadata['problem']`` so the existing
SelectorContext / selectors work unchanged over code tasks.
"""
from __future__ import annotations

from ..api.executor import APIExecutor
from ..data.datastructures import CandidatePrompt, TaskExample
from .code_exec import CodeProblem, extract_code, run_tests


class CodeScorer:
    def __init__(self, executor: APIExecutor, timeout: float = 5.0):
        self.executor = executor
        self.timeout = timeout

    def score_one(self, prompt: CandidatePrompt, example: TaskExample,
                  call_type: str = "selector_dev") -> float:
        problem: CodeProblem = example.metadata["problem"]
        trace = self.executor.run_prompt_on_example(prompt, example, call_type=call_type)
        result = run_tests(extract_code(trace.output_text) or "", problem, self.timeout)
        return 1.0 if result.all_pass else 0.0

    def score_many(self, prompt, examples, call_type="selector_dev") -> list[float]:
        return [self.score_one(prompt, e, call_type) for e in examples]


def problems_to_examples(problems: list[CodeProblem], task_id: str) -> list[TaskExample]:
    """Wrap CodeProblems as TaskExamples (problem carried in metadata)."""
    return [TaskExample(example_id=p.problem_id, task_id=task_id,
                        input_text=p.prompt, target=None,
                        metadata={"problem": p}) for p in problems]
